import json
import re
from utils.common_utils import read_yaml, read_json


class DirectPlanning:
    """
    Direct Planning baseline: directly output the structured JSON plan without chain-of-thought.
    """

    def __init__(self, io, evaluator, use_fewshot: bool = True):
        self.io = io
        self.evaluator = evaluator
        self.use_fewshot = use_fewshot
        self.prompt = read_yaml("baselines/baseline_prompts/dp_prompt.yaml")["DIRECT_PLANNING"]
        self.tools_desc = read_json("interfaces/tools/tools_manifest.json")
        self._generated_ref_pattern = re.compile(r"^<GENERATED>-(\d+)-<?([^<>]+)>?$")
        self._agent_outputs = {
            agent_name: list(agent_spec.get("output", {}).keys())
            for agent_name, agent_spec in self.tools_desc.items()
        }
        self.last_all_completions = None
        self.last_selected_completion = None

    def _extract_json_array_from_text(self, text: str):
        """
        Best-effort extraction of a JSON ARRAY from a model completion.
        DP is supposed to output: "The structured task plan is: [ ... ]"
        but small models often include extra text, code fences, or prefixes.
        Returns the parsed Python object (usually list) or None.
        """
        if not isinstance(text, str) or not text.strip():
            return None

        s = text.strip()

        # Prefer extracting the substring after the strict prefix if present.
        prefix = "The structured task plan is:"
        if prefix in s:
            s = s.split(prefix, 1)[1].strip()

        # Strip common markdown fences around JSON.
        s = s.strip()
        if s.startswith("```"):
            # Remove first fence line and trailing fence if present.
            # Works for ```json\n...\n``` and ```\n...\n```
            lines = s.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()

        # If it's already valid JSON, parse directly.
        try:
            obj = json.loads(s)
            return obj
        except Exception:
            pass

        # Fallback: try to find the "best" JSON array substring in text.
        # Prefer arrays of objects (plans), and avoid accidentally grabbing
        # list fragments like [-1] or [[0], ["x"], ...].
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
        # Hard cap attempts to keep this cheap.
        for open_idx in starts[:50]:
            close_idx = find_matching_bracket(s, open_idx)
            if close_idx is None:
                continue
            candidate = s[open_idx : close_idx + 1].strip()
            # Quick reject extremely small candidates.
            if len(candidate) < 2:
                continue
            try:
                obj = json.loads(candidate)
            except Exception:
                continue

            if not isinstance(obj, list):
                continue

            # Score candidates: list-of-dicts is most likely the plan.
            score = 0
            if obj and all(isinstance(x, dict) for x in obj):
                score = 100 + len(obj)
            elif len(obj) == 1 and isinstance(obj[0], list) and obj[0] and all(isinstance(x, dict) for x in obj[0]):
                score = 90 + len(obj[0])
            elif obj and any(isinstance(x, dict) for x in obj):
                score = 50 + sum(1 for x in obj if isinstance(x, dict))
            else:
                # Keep as a last resort; still better than None.
                score = 1

            # Bonus if it literally looks like an array of objects.
            if re.search(r"\[\s*\{", candidate):
                score += 20

            if score > best_score:
                best_score = score
                best = obj

        return best

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

    def _postprocess_structured_plan(self, answer_text: str):
        """
        DP models (especially small ones) often get field details slightly wrong:
        - <GENERATED>-k-output_key (missing < >)
        - dependence=[1] instead of [0]
        - dependence_content as list instead of {\"0\": [\"key\"]}
        - missing outputs
        This postprocess tries to normalize into the canonical format expected by the evaluator.
        """
        if not isinstance(answer_text, str):
            return answer_text
        try:
            plan = json.loads(answer_text)
        except Exception:
            return answer_text

        # --- Shape repairs for common DP failure modes ---
        # 1) Model outputs nested lists like [[{...}, {...}]] -> flatten
        # 2) Model outputs junk lists like [[], []] or [[ ]] -> treat as invalid
        if isinstance(plan, list):
            # Flatten single nested list: [[...]] -> [...]
            if len(plan) == 1 and isinstance(plan[0], list):
                plan = plan[0]
            # If it's a list of lists, pick the first list-of-dicts if any.
            if plan and all(isinstance(x, list) for x in plan):
                picked = None
                for sub in plan:
                    if isinstance(sub, list) and sub and all(isinstance(y, dict) for y in sub):
                        picked = sub
                        break
                if picked is not None:
                    plan = picked
        else:
            return answer_text

        if not isinstance(plan, list) or not plan:
            return answer_text

        # Drop non-object entries (models often emit leading [-1], [[0], ...], etc.).
        plan = [x for x in plan if isinstance(x, dict)]
        if not plan:
            return answer_text

        # Canonicalize common alternative field names to our evaluator schema.
        # Some models output keys like step_index/agent_name/dependencies.
        for item in plan:
            if "step" not in item and isinstance(item.get("step_index"), int):
                item["step"] = item["step_index"]
            if "agent" not in item and isinstance(item.get("agent_name"), str):
                item["agent"] = item["agent_name"]
            if "dependence" not in item and isinstance(item.get("dependencies"), list):
                item["dependence"] = item["dependencies"]
            # Normalize dependence_content key variants
            if "dependence_content" not in item and "dependency_content" in item:
                item["dependence_content"] = item.get("dependency_content")

        # Keep only steps that declare an agent name (typos are handled by the evaluator as agent_mismatch).
        plan = [x for x in plan if isinstance(x.get("agent"), str) and x.get("agent").strip()]
        if not plan:
            return answer_text

        # Coerce common scalar / string types into protocol shapes.
        for item in plan:
            st = item.get("step")
            if isinstance(st, str) and st.strip().lstrip("-").isdigit():
                item["step"] = int(st.strip())
            dep = item.get("dependence")
            if isinstance(dep, (int, float)) and int(dep) == dep:
                item["dependence"] = [int(dep)]
            elif (
                isinstance(dep, list)
                and len(dep) == 1
                and isinstance(dep[0], str)
                and dep[0].strip().lstrip("-").isdigit()
            ):
                item["dependence"] = [int(dep[0].strip())]
            inp = item.get("inputs")
            if not isinstance(inp, dict):
                item["inputs"] = {}

        plan.sort(key=lambda x: x.get("step") if isinstance(x.get("step"), int) else 10**9)

        # Fill missing outputs from tool manifest; build step->outputs mapping
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
                # only one predecessor allowed by protocol
                item["dependence"] = [predecessor]
                dedup = []
                for ck in consumed_keys:
                    if ck not in dedup:
                        dedup.append(ck)
                item["dependence_content"] = {str(predecessor): dedup}

        return json.dumps(plan, ensure_ascii=False)

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

    def generate(self, user_question: str) -> str:
        examples_str = ""
        if self.use_fewshot and self.prompt.get("examples"):
            examples_str = "Examples: "
            for example in self.prompt["examples"]:
                examples_str += "\n" + f"{example['input']}"
                examples_str += "\n" + f"{example['output']}"

        system_prompt = self.prompt["system_prompt"].format(
            agents_desc=self.tools_desc,
            examples=examples_str,
        )
        user_prompt = self.prompt["user_prompt"].format(task_desc=user_question)
        io_input = system_prompt + "\n" + user_prompt

        io_output_list = self.io.generate(io_input, num_return=1)
        if not isinstance(io_output_list, (list, tuple)):
            io_output_list = [io_output_list]
        self.last_all_completions = io_output_list
        self.last_selected_completion = io_output_list[0] if io_output_list else None

        raw = (io_output_list[0] or "").strip() if io_output_list else ""
        extracted = self.evaluator.extract_answer_from_model_completion(raw)
        # If evaluator fails, attempt a best-effort JSON extraction from raw.
        if extracted is None:
            obj = self._extract_json_array_from_text(raw)
            if isinstance(obj, list):
                extracted = json.dumps(obj, ensure_ascii=False)
            else:
                return raw

        # If evaluator returned a string that still includes prefix/fences,
        # parse and re-dump to canonical JSON before postprocess.
        obj2 = self._extract_json_array_from_text(extracted) if isinstance(extracted, str) else None
        if isinstance(obj2, list):
            extracted = json.dumps(obj2, ensure_ascii=False)

        post = self._postprocess_structured_plan(extracted)
        if self._is_canonical_plan_json(post):
            return post

        # One-shot format repair: ask the model to only reformat into the canonical plan schema.
        repair_prompt = (
            "You previously produced an invalid plan format.\n"
            "Your task is STRICTLY to FIX FORMAT/SYNTAX ONLY.\n"
            "Do NOT change the plan content: do NOT add/remove steps, do NOT change agent choices, do NOT change step order, and do NOT change any input values.\n"
            "Only convert the existing plan you already intended into the canonical schema.\n\n"
            "Reformat into EXACTLY a valid JSON array of step objects with keys:\n"
            "agent (string), step (int), dependence (list with one int), dependence_content (null or object), inputs (object), outputs (list of strings).\n"
            "Rules:\n"
            "- Use only agent names and input/output keys from the provided manifest.\n"
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

