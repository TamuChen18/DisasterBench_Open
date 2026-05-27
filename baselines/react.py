import json
import re

from utils.common_utils import read_yaml, read_json


class ReActPlanning:
    """
    ReAct baseline: allow the model to reason in Thought/Action/Observation style,
    but still require the final output to contain the structured JSON plan.
    """

    def __init__(self, io, evaluator, use_fewshot: bool = True, num_samples: int = 1):
        self.io = io
        self.evaluator = evaluator
        self.use_fewshot = use_fewshot
        self.num_samples = num_samples
        self.prompt = read_yaml("baselines/baseline_prompts/react_prompt.yaml")["REACT"]
        self.tools_desc = read_json("interfaces/tools/tools_manifest.json")
        self._generated_ref_pattern = re.compile(r"^<GENERATED>-(\d+)-<?([^<>]+)>?$")
        self._agent_outputs = {
            agent_name: list(agent_spec.get("output", {}).keys())
            for agent_name, agent_spec in self.tools_desc.items()
        }
        self.last_all_completions = None
        self.last_selected_completion = None

    def _normalize_generated_ref(self, value):
        if not isinstance(value, str):
            return value, None
        m = self._generated_ref_pattern.match(value.strip())
        if not m:
            return value, None
        step_idx = int(m.group(1))
        output_key = m.group(2).strip()
        normalized = f"<GENERATED>-{step_idx}-<{output_key}>"
        return normalized, (step_idx, output_key)

    def _extract_json_array_from_text(self, text: str):
        """
        Best-effort extraction of a JSON ARRAY from a model completion.
        ReAct is supposed to end with: "The structured task plan is: [ ... ]"
        but models sometimes omit the prefix and/or include extra reasoning text.
        Returns the parsed Python object (usually list) or None.
        """
        if not isinstance(text, str) or not text.strip():
            return None

        s = text.strip()

        prefix = "The structured task plan is:"
        if prefix in s:
            s = s.split(prefix, 1)[1].strip()

        # Strip common markdown fences.
        if s.startswith("```"):
            lines = s.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()

        try:
            obj = json.loads(s)
            return obj
        except Exception:
            pass

        # Fallback: find the best JSON array substring.
        starts = [m.start() for m in re.finditer(r"\[", s)]
        if not starts:
            return None

        def find_matching_bracket(src: str, open_idx: int):
            depth = 0
            for i in range(open_idx, len(src)):
                ch = src[i]
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        return i
            return None

        best = None
        best_score = -1
        for open_idx in starts[:50]:
            close_idx = find_matching_bracket(s, open_idx)
            if close_idx is None:
                continue
            candidate = s[open_idx : close_idx + 1].strip()
            if len(candidate) < 2:
                continue
            try:
                obj = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(obj, list):
                continue

            score = 0
            if obj and all(isinstance(x, dict) for x in obj):
                score = 100 + len(obj)
            elif len(obj) == 1 and isinstance(obj[0], list) and obj[0] and all(isinstance(x, dict) for x in obj[0]):
                score = 90 + len(obj[0])
            elif obj and any(isinstance(x, dict) for x in obj):
                score = 50 + sum(1 for x in obj if isinstance(x, dict))
            else:
                score = 1

            if re.search(r"\[\s*\{", candidate):
                score += 20

            if score > best_score:
                best_score = score
                best = obj

        return best

    def _is_canonical_plan_json(self, plan_text: str) -> bool:
        if not isinstance(plan_text, str):
            return False
        try:
            obj = json.loads(plan_text)
        except Exception:
            return False
        if not isinstance(obj, list) or not obj:
            return False
        for step in obj:
            if not isinstance(step, dict):
                return False
            if not isinstance(step.get("step"), int):
                return False
            if not isinstance(step.get("agent"), str) or not step.get("agent"):
                return False
            if not isinstance(step.get("inputs"), dict):
                return False
            if not isinstance(step.get("outputs"), list):
                return False
            if not (isinstance(step.get("dependence"), list) and len(step["dependence"]) == 1 and isinstance(step["dependence"][0], int)):
                return False
            dc = step.get("dependence_content", None)
            if dc is not None and not isinstance(dc, dict):
                return False
        return True

    def _postprocess_structured_plan(self, answer_text: str):
        # Same postprocess behavior as CoT/DP: normalize GENERATED refs, fill missing outputs, infer dependence.
        if not isinstance(answer_text, str):
            return answer_text
        try:
            plan = json.loads(answer_text)
        except Exception:
            return answer_text
        if not isinstance(plan, list):
            return answer_text

        # Drop non-object entries that sometimes leak from reasoning formats.
        plan = [x for x in plan if isinstance(x, dict)]
        if not plan:
            return answer_text

        # Normalize step indices: enforce 0-based sequential steps.
        # Some models start from 1 even when only one step exists.
        try:
            plan.sort(key=lambda x: x.get("step") if isinstance(x.get("step"), int) else 10**9)
            for i, item in enumerate(plan):
                item["step"] = i
        except Exception:
            pass

        # Build step -> outputs map, filling missing outputs from tool manifest when possible.
        step_to_outputs = {}
        for item in plan:
            step_idx = item.get("step")
            if not isinstance(step_idx, int):
                continue
            outputs = item.get("outputs", [])
            if not isinstance(outputs, list) or len(outputs) == 0:
                agent_name = item.get("agent")
                outputs = self._agent_outputs.get(agent_name, []) if isinstance(agent_name, str) else []
                if outputs:
                    item["outputs"] = outputs
            step_to_outputs[step_idx] = item.get("outputs", []) if isinstance(item.get("outputs"), list) else []

        for item in plan:
            inputs = item.get("inputs", {})
            if not isinstance(inputs, dict):
                inputs = {}
                item["inputs"] = inputs

            predecessor = None
            consumed_keys = []
            for k, v in list(inputs.items()):
                norm_v, parsed = self._normalize_generated_ref(v)
                if parsed is None:
                    continue
                ref_step, ref_key = parsed
                valid_outputs = step_to_outputs.get(ref_step, [])
                if isinstance(valid_outputs, list) and valid_outputs and ref_key not in valid_outputs and len(valid_outputs) == 1:
                    ref_key = valid_outputs[0]
                    norm_v = f"<GENERATED>-{ref_step}-<{ref_key}>"
                inputs[k] = norm_v
                if predecessor is None:
                    predecessor = ref_step
                consumed_keys.append(ref_key)

            if predecessor is None:
                item["dependence"] = [-1]
                item["dependence_content"] = None
            else:
                dedup_keys = []
                for ck in consumed_keys:
                    if ck not in dedup_keys:
                        dedup_keys.append(ck)
                item["dependence"] = [predecessor]
                item["dependence_content"] = {str(predecessor): dedup_keys}

        return json.dumps(plan, ensure_ascii=False)

    def generate(self, user_question: str) -> str:
        examples_str = ""
        if self.use_fewshot and self.prompt.get("examples"):
            examples_str = "Examples: "
            for example in self.prompt["examples"]:
                examples_str += "\n" + f"{example['input']}"
                examples_str += "\n" + f"{example['output']}"

        # NOTE: system_prompt contains many literal `{}` from JSON in few-shot examples.
        # Using Python `.format()` would treat them as placeholders and crash.
        # We only need to substitute `{agents_desc}` and `{examples}`.
        system_prompt_template = self.prompt["system_prompt"]
        system_prompt = (
            system_prompt_template
            .replace("{agents_desc}", json.dumps(self.tools_desc, ensure_ascii=False))
            .replace("{examples}", examples_str)
        )
        user_prompt = self.prompt["user_prompt"].format(task_desc=user_question)
        io_input = system_prompt + "\n" + user_prompt

        io_output_list = self.io.generate(io_input, num_return=self.num_samples)
        if not isinstance(io_output_list, (list, tuple)):
            io_output_list = [io_output_list]
        self.last_all_completions = io_output_list
        self.last_selected_completion = io_output_list[0] if io_output_list else None

        raw = (io_output_list[0] or "").strip() if io_output_list else ""
        extracted = self.evaluator.extract_answer_from_model_completion(raw)
        if extracted is None:
            obj = self._extract_json_array_from_text(raw)
            if isinstance(obj, list):
                extracted = json.dumps(obj, ensure_ascii=False)
            else:
                extracted = None

        if isinstance(extracted, str):
            obj2 = self._extract_json_array_from_text(extracted)
            if isinstance(obj2, list):
                extracted = json.dumps(obj2, ensure_ascii=False)

        if isinstance(extracted, str):
            post = self._postprocess_structured_plan(extracted)
            if self._is_canonical_plan_json(post):
                return post
        else:
            post = raw

        # One-shot format repair: enforce that the model outputs ONLY the JSON array.
        repair_prompt = (
            "You previously produced an invalid plan output.\n"
            "Your task is STRICTLY to FIX OUTPUT FORMAT ONLY.\n"
            "Do NOT change the plan content: do NOT add/remove steps, do NOT change agent choices, do NOT change step order, and do NOT change any input values.\n"
            "Only convert the plan you already intended into the canonical JSON array schema.\n\n"
            "Output EXACTLY a valid JSON array of step objects with keys:\n"
            "agent (string), step (int), dependence (list with one int), dependence_content (null or object), inputs (object), outputs (list of strings).\n"
            "Rules:\n"
            "- If a step uses any <GENERATED>-k-<OutputKey> input, set dependence=[k] and dependence_content={\"k\": [\"OutputKey\", ...]}.\n"
            "- If all inputs are user-provided paths/strings, set dependence=[-1] and dependence_content=null.\n"
            "- Output ONLY the JSON array. No prose, no markdown.\n\n"
            f"Task:\n{user_question}\n\n"
            f"Agents manifest (JSON):\n{json.dumps(self.tools_desc, ensure_ascii=False)}\n\n"
            f"Your previous output:\n{raw}\n"
        )
        repaired_list = self.io.generate(repair_prompt, num_return=1)
        repaired_raw = (repaired_list[0] or "").strip() if isinstance(repaired_list, (list, tuple)) and repaired_list else ""
        repaired_obj = self._extract_json_array_from_text(repaired_raw)
        if isinstance(repaired_obj, list):
            repaired_text = json.dumps(repaired_obj, ensure_ascii=False)
            post2 = self._postprocess_structured_plan(repaired_text)
            if self._is_canonical_plan_json(post2):
                return post2

        return post

