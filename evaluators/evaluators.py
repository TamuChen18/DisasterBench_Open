import json
import random
from typing import List, Union
import re
from collections import defaultdict
try:
    from json_repair import repair_json
except ModuleNotFoundError:
    repair_json = None
import traceback

class Evaluator:
    def __init__(self):
        self.answer_marker = "The structured task plan is: "
        self.completion_count = []

    @staticmethod
    def _dependence_as_list(dep):
        """Normalize dependence to a list before sorting (models may emit -1 instead of [-1])."""
        if dep is None:
            return []
        if isinstance(dep, (list, tuple)):
            return list(dep)
        if isinstance(dep, (int, float)):
            return [dep]
        if isinstance(dep, str):
            return [dep]
        try:
            return list(dep)
        except TypeError:
            return [dep]

    def remove_newlines(self, text):
        return re.sub(r"\n+", " ", text)

    @staticmethod
    def _step_sort_key(d):
        if not isinstance(d, dict):
            return 10**9
        s = d.get("step", d.get("step_index"))
        if isinstance(s, int):
            return s
        if isinstance(s, str) and s.strip().lstrip("-").isdigit():
            return int(s.strip())
        return 10**9

    def _coerce_flat_plan_list(self, obj):
        """
        From a top-level JSON list that may mix lists/primitives with step dicts,
        extract a flat [dict, ...] suitable for evaluation. Returns None if nothing usable.
        """
        if not isinstance(obj, list) or not obj:
            return None

        def is_step_dict(d):
            if not isinstance(d, dict):
                return False
            if isinstance(d.get("agent"), str) and d.get("agent").strip():
                return True
            if isinstance(d.get("agent_name"), str) and d.get("agent_name").strip():
                return True
            return False

        # Best case: already a list of dicts. Prefer dicts that look like steps (have agent-ish fields),
        # but if none match, still return dicts so we can avoid "format_error" due to stray primitives.
        if all(isinstance(x, dict) for x in obj):
            out = [x for x in obj if is_step_dict(x)]
            return out if out else obj

        out = [x for x in obj if is_step_dict(x)]
        if out:
            return out

        flattened = []
        for x in obj:
            if isinstance(x, list):
                for y in x:
                    if is_step_dict(y):
                        flattened.append(y)
            elif is_step_dict(x):
                flattened.append(x)
        if flattened:
            return flattened

        # Last resort: if there are ANY dicts anywhere at top-level, keep them.
        any_dicts = [x for x in obj if isinstance(x, dict)]
        if any_dicts:
            return any_dicts
        return None

    @staticmethod
    def _canonicalize_step_dict(d: dict) -> dict:
        """
        For evaluation, keep only the canonical schema fields and fix common aliases.
        This prevents strict equivalence checks from failing due to extra, irrelevant keys.
        """
        if not isinstance(d, dict):
            return d
        out = {}
        # aliases
        step = d.get("step", d.get("step_index"))
        agent = d.get("agent", d.get("agent_name"))
        dependence = d.get("dependence", d.get("dependencies"))
        dep_content = d.get("dependence_content", d.get("dependency_content"))
        inputs = d.get("inputs")
        if not isinstance(inputs, dict) or inputs == {}:
            alt_in = d.get("input")
            if isinstance(alt_in, dict):
                inputs = alt_in
        outputs = d.get("outputs")

        out["step"] = step
        out["agent"] = agent
        out["inputs"] = inputs if isinstance(inputs, dict) else {}
        out["outputs"] = outputs if isinstance(outputs, list) else []
        out["dependence"] = dependence
        out["dependence_content"] = dep_content
        return out

    def _canonicalize_plan_list(self, plan_list):
        if not isinstance(plan_list, list):
            return plan_list
        return [self._canonicalize_step_dict(x) if isinstance(x, dict) else x for x in plan_list]

    def _unwrap_structured_plan_object(self, obj):
        """JSON mode may return {reasoning, structured_plan: [...]}."""
        if not isinstance(obj, dict):
            return obj
        for key in (
            "structured_plan",
            "plan",
            "structured_task_plan",
            "task_plan",
            "answer",
        ):
            val = obj.get(key)
            if isinstance(val, list):
                return val
        return obj

    def normalize_answer_for_evaluation(self, test_type: str, model_answer):
        """
        RAP may wrap rollouts as [[{step}, ...], ...]; check_* expects a flat [dict, ...].
        Does not change gold labels. If the parsed value is not a flat plan, try
        extract_answer_from_model_completion. DP may emit nested arrays or extra text;
        apply the same normalization. CoT/ToT pass through when already flat.
        """
        if test_type not in ("rap", "dp", "react", "cot", "tot", "sc") or model_answer is None:
            return model_answer
        if not isinstance(model_answer, str):
            return model_answer
        def is_flat_plan(o):
            return isinstance(o, list) and len(o) > 0 and all(isinstance(x, dict) for x in o)

        def find_embedded_plan(o):
            # Many models emit something like: [ [...], [{...},{...}], ... ] or wrap rollouts.
            if not isinstance(o, list):
                return None
            for item in o:
                if is_flat_plan(item) and all(("agent" in x and "step" in x) for x in item):
                    return item
            return None

        try:
            obj = json.loads(self.remove_newlines(model_answer))
        except json.JSONDecodeError:
            extracted = self.extract_answer_from_model_completion(model_answer)
            return extracted if extracted else model_answer

        unwrapped = self._unwrap_structured_plan_object(obj)
        if unwrapped is not obj:
            obj = unwrapped

        if is_flat_plan(obj):
            canon = self._canonicalize_plan_list(obj)
            return json.dumps(canon, ensure_ascii=False)

        embedded = find_embedded_plan(obj)
        if embedded is not None:
            canon = self._canonicalize_plan_list(embedded)
            return json.dumps(canon, ensure_ascii=False)

        coerced = self._coerce_flat_plan_list(obj)
        if coerced is not None:
            coerced = sorted(coerced, key=self._step_sort_key)
            canon = self._canonicalize_plan_list(coerced)
            return json.dumps(canon, ensure_ascii=False)
        # If it's a JSON list but contains no usable dict steps, normalize to empty list instead of
        # letting downstream emit "format_error" (this becomes early_stop/empty_output, which is a
        # better reflection of model output quality than parser shape issues).
        if isinstance(obj, list):
            return "[]"

        extracted = self.extract_answer_from_model_completion(model_answer)
        if extracted:
            try:
                ex_obj = json.loads(self.remove_newlines(extracted))
            except Exception:
                return extracted
            if is_flat_plan(ex_obj):
                canon = self._canonicalize_plan_list(ex_obj)
                return json.dumps(canon, ensure_ascii=False)
            embedded = find_embedded_plan(ex_obj)
            if embedded is not None:
                canon = self._canonicalize_plan_list(embedded)
                return json.dumps(canon, ensure_ascii=False)
            coerced_ex = self._coerce_flat_plan_list(ex_obj)
            if coerced_ex is not None:
                coerced_ex = sorted(coerced_ex, key=self._step_sort_key)
                canon = self._canonicalize_plan_list(coerced_ex)
                return json.dumps(canon, ensure_ascii=False)
            if isinstance(ex_obj, list):
                return "[]"
        return extracted if extracted else model_answer

    def normalize_outputs(self, text: Union[str, dict, list]):
        normalized_text = self.remove_newlines(text).replace("\\", "").strip()
        return normalized_text

    def check_braces_balance(self, text):
        stack = []
        opening = {'(': ')', '{': '}', '[': ']'}
        closing = {')', '}', ']'}

        for char in text:
            if char in opening:
                stack.append(opening[char])
            elif char in closing:
                if not stack or char != stack.pop():
                    return False

        return len(stack) == 0 
    
    def isolate_answer(self, text) -> str:
        if text is None:
            return None
        assert isinstance(text, str)

        text = self.normalize_outputs(text)

        if not self.check_braces_balance(text):
            return None
        else:
            # 1) Preferred: a JSON array of objects: [ {...}, {...}, ... ]
            pattern = r'\[\s*(?:\{(?:[^{}]|\{[^{}]*\})*\}(?:\s*,\s*\{(?:[^{}]|\{[^{}]*\})*\})*)\s*\]'
            match = re.search(pattern, text, flags=re.S)
            if match:
                return match.group(0)

            # 2) Fallback: a comma-separated sequence of JSON objects without outer brackets:
            #    {...}, {...}, {...}
            obj_seq_pattern = r'\{\s*(?:[^{}]|\{[^{}]*\})*\}\s*(?:,\s*\{\s*(?:[^{}]|\{[^{}]*\})*\}\s*)+'
            seq_match = re.search(obj_seq_pattern, text, flags=re.S)
            if seq_match:
                return "[" + seq_match.group(0).strip() + "]"

            # 3) Fallback: a single JSON object (wrap to list if it looks like a step item)
            obj_pattern = r'\{\s*(?:[^{}]|\{[^{}]*\})*\}\s*'
            obj_match = re.search(obj_pattern, text, flags=re.S)
            if obj_match:
                candidate = obj_match.group(0).strip()
                if '"agent"' in candidate and '"step"' in candidate:
                    return "[" + candidate + "]"
        return None

    def find_most_confident_answer(self, user_question, completions, promptbuilder, io):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list) 
        answer2ids = defaultdict(list)
        for id, c in enumerate(completions):
            try:
                model_answer = self.extract_answer_from_model_completion(c)
                if model_answer is None:
                    continue
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if self.check_answers_equivalence(model_answer, existing_answer):
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)
                if not has_existed:
                    answer2completions[model_answer].append(c)
                    answer2ids[model_answer].append(id)
            except:
                pass

        if len(answer2completions.keys()) == 0:
            random_id = random.randrange(len(completions))
            random_completion = completions[random_id]
            return (random_completion, random_completion, random_id, 1e-3)
        
        if None in answer2ids:
            del answer2ids[None]
        if None in answer2completions:
            del answer2completions[None]
        assert len(answer2completions.keys()) > 0, "There are no valid completions."
        
        most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))
        sorted_answers = sorted(answer2completions.keys(), key=lambda x: len(answer2completions[x]), reverse=True)
        if len(sorted_answers) > 1 and len(answer2completions[sorted_answers[0]])  == len(answer2completions[sorted_answers[1]]):
            
            check_io_input = promptbuilder.build_check_confidence_prompt(user_question = user_question,
                                                                         output1=answer2completions[sorted_answers[0]][0],
                                                                         output2=answer2completions[sorted_answers[1]][0])
            check_output_list = io.generate(
                model_input=check_io_input, max_tokens=10, num_return=5, stop=["\n", "\n\n"]
            )
            try:
                check_output_list = [z.strip()[0] for z in check_output_list] 
                one_count = check_output_list.count('1')
            except:
                one_count = 5
            if one_count >= 3:
                most_confident_answer = sorted_answers[0]
            else:
                most_confident_answer = sorted_answers[1]

        assert (
            len(answer2completions[most_confident_answer]) > 0
        ), "There are no completions for the most confident answer."
        
        confidence = len(answer2completions[most_confident_answer]) / len(completions)
        assert confidence > 0
        return (
            most_confident_answer,
            answer2completions[most_confident_answer][0],  
            answer2ids[most_confident_answer][0],
            confidence,
        ) 
    
    def check_answers_equivalence(self, model_answer, existing_answer):
        """
        Check if two JSON-encoded answers are equivalent in structure and content.
        Uses deep structural comparison via self.deep_equal.
        """
        if not model_answer:
            return False

        try:
            model_answer = json.loads(self.remove_newlines(model_answer))
            existing_answer = json.loads(self.remove_newlines(existing_answer))
        except json.JSONDecodeError:
            print("Invalid JSON format in model or existing answer.")
            return False

        # Both must be arrays of equal length
        if not isinstance(model_answer, list) or not isinstance(existing_answer, list):
            return False
        if len(model_answer) != len(existing_answer):
            return False

        # Compare each item deeply
        for a, b in zip(model_answer, existing_answer):
            if not self.deep_equal(a, b):
                return False

        return True
    
    # def deep_equal(self, obj1, obj2):
    #     if isinstance(obj1, dict) and isinstance(obj2, dict):
    #         if set(obj1.keys()) != set(obj2.keys()):
    #             return False
    #         for k in obj1.keys():
    #             if k == "dependence_content" and obj1[k] is not None and obj2[k] is not None:
    #                 if isinstance(obj1[k], dict) and isinstance(obj2[k], dict):
    #                     if not self.deep_equal(list(obj1[k].values()), list(obj2[k].values())):
    #                         return False
    #                     else:
    #                         continue
    #                 else:
    #                     if not self.deep_equal(obj1[k], obj2[k]):
    #                         return False
    #                     else:
    #                         continue

    #             if k not in obj2.keys() or not self.deep_equal(obj1[k], obj2[k]):
    #                 return False                                                         
    #         return True

    #     elif isinstance(obj1, list) and isinstance(obj2, list):
    #         if len(obj1) != len(obj2):
    #             return False
    #         return all(self.deep_equal(a, b) for a, b in zip(sorted(obj1), sorted(obj2)))

    #     else:
    #         return obj1 == obj2

    def deep_equal(self, obj1, obj2):
        """
        Recursively check deep equality between two JSON-like Python objects.
        Handles dicts, lists, and primitive types.
        Special case: 'dependence_content' only compares its values, not keys.
        Special case: 'dependence' may be -1 vs [-1] from model JSON.
        """
        # Type mismatch or None mismatch
        if type(obj1) != type(obj2):
            return False

        # ---- Dict comparison ----
        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False

            for k in obj1:
                v1, v2 = obj1[k], obj2[k]

                if k == "dependence":
                    if sorted(self._dependence_as_list(v1)) != sorted(self._dependence_as_list(v2)):
                        return False
                    continue

                # General recursive comparison
                if not self.deep_equal(v1, v2):
                    return False

            return True

        # ---- List comparison ----
        elif isinstance(obj1, list):
            if len(obj1) != len(obj2):
                return False
            return all(self.deep_equal(a, b) for a, b in zip(obj1, obj2))

        # ---- Base case ----
        else:
            return obj1 == obj2

    def stochastic_select_answer(self, completion2score, answer2completions, completions):
        answer2score = {}
        answer_counts = {}
        for completion, score in completion2score.items():
            answer = self.extract_answer_from_model_completion(completion)
            if answer in answer2score:
                answer2score[answer] += score
                answer_counts[answer] += 1
            else:
                answer2score[answer] = score
                answer_counts[answer] = 1

        for answer in answer2score:
            answer2score[answer] /= answer_counts[answer]

        top_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)[:1]
        answers, scores = zip(*top_answers)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            selected_answer = random.choices(answers, weights=probabilities, k=1)[0]
        except:
            selected_answer = random.choices(answers, k=1)[0]

        most_confident_completion = answer2completions[selected_answer][0]
        completion_index = completions.index(most_confident_completion)
        confidence = answer2score[selected_answer]

        return selected_answer, most_confident_completion, completion_index, confidence

    def stochastic_calculate_completion_scores(self, prior_weights, answer2completions):
        completion2count = {}
        for answer, comps in answer2completions.items():
            count = len(comps)
            for comp in comps:
                completion2count[comp] = count
        completion2score = {}
        for idx, comp in enumerate(completion2count.keys()):
            weight = prior_weights[idx] if prior_weights is not None else 1
            score = weight * completion2count[comp]
            completion2score[comp] = score
        return completion2score

    def stochastic_select_response(self, completion2score, completions):
        sorted_completions = sorted(completion2score.items(), key=lambda x: x[1], reverse=True)[:1]
        completions, scores = zip(*sorted_completions)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            sampled_completion = random.choices(completions, weights=probabilities, k=1)[0]
        except:
            sampled_completion = random.choices(completions, k=1)[0]
        confidence = completion2score[sampled_completion]
        most_confident_answer = self.extract_answer_from_model_completion(sampled_completion)
        id_of_most_confident = completions.index(sampled_completion)
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def stochastic_find_most_confident_answer(self, completions: List[str], prior_weights: List[float] = None):

        if not completions or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        answer2counts = defaultdict(list)

        for idx, comp in enumerate(completions):
            try:
                answer = self.extract_answer_from_model_completion(comp)
                answer2completions[answer].append(comp)
            except:
                continue

        if not answer2completions:
            return None, None, None, None
        
        for answer, completions in answer2completions.items():
            answer2counts[answer] = len(completions)
        
        self.completion_count.append(answer2counts)

        completion2score = self.stochastic_calculate_completion_scores(prior_weights, answer2completions)

        most_confident_answer, sampled_completion, id_of_most_confident, confidence = self.stochastic_select_response(
            completion2score, completions
        )
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def extract_answer_from_model_completion(self, completion) -> str:

        assert isinstance(completion, str)

        answer_split = self.isolate_answer(completion)

        # Fallback: model text may contain valid JSON array without passing brace-balance checks on full text
        if answer_split is None or len(answer_split) == 0:
            if isinstance(completion, str) and "[" in completion and "]" in completion:
                i = completion.find("[")
                j = completion.rfind("]")
                if i != -1 and j > i:
                    answer_split = completion[i : j + 1]
        
        if answer_split is None or len(answer_split) == 0:
            return None
        
        json_str = None
        candidates = [answer_split]
        brace_normalized = answer_split.replace("{{", "{").replace("}}", "}")
        if brace_normalized != answer_split:
            candidates.append(brace_normalized)

        for candidate in candidates:
            try:
                loaded_json = json.loads(candidate)
                plan = self._unwrap_structured_plan_object(loaded_json)
                if plan is not loaded_json and isinstance(plan, list):
                    json_str = json.dumps(plan)
                else:
                    json_str = json.dumps(loaded_json)
                break
            except Exception:
                try:
                    if repair_json is None:
                        raise ModuleNotFoundError("json_repair is not installed")
                    repaired_json = json.loads(repair_json(candidate))
                    plan = self._unwrap_structured_plan_object(repaired_json)
                    if plan is not repaired_json and isinstance(plan, list):
                        json_str = json.dumps(plan)
                    else:
                        json_str = json.dumps(repaired_json)
                    break
                except Exception:
                    continue

        if json_str is None:
            print(traceback.format_exc())
        
        return json_str

    def extract_answer_from_gold_solution(self, solution) -> str:
        return json.dumps(solution)
    
    def check_tools_correctness(self, model_answer, gt_answer):
        """
        Check if the (step, agent) pairs in model and ground truth answers match exactly.
        """
        if not model_answer:
            return False

        if not isinstance(model_answer, str) or not isinstance(gt_answer, str):
            return False

        # Parse JSON safely
        try:
            model_answer = json.loads(model_answer)
            gt_answer = json.loads(gt_answer)
        except json.JSONDecodeError:
            print("Invalid JSON format in model or ground truth answer.")
            return False

        # Must be lists of equal length
        if not isinstance(model_answer, list) or not isinstance(gt_answer, list):
            return False
        if len(model_answer) != len(gt_answer):
            return False

        # Extract (step, agent) pairs
        model_tools = [
            (item.get("step"), item.get("agent"))
            for item in model_answer
            if isinstance(item, dict) and "step" in item and "agent" in item
        ]
        gt_tools = [
            (item.get("step"), item.get("agent"))
            for item in gt_answer
            if isinstance(item, dict) and "step" in item and "agent" in item
        ]

        # If the structure differs (missing fields), fail early
        if len(model_tools) != len(gt_tools):
            return False

        # Direct list comparison (order matters)
        return model_tools == gt_tools

    def check_parameters_correctness(self, model_answer, gt_answer):
        """
        Check if the 'input' parameters match for each step between model and ground truth.
        """
        if not model_answer:
            return False

        if not isinstance(model_answer, str) or not isinstance(gt_answer, str):
            return False

        # Try parsing JSON
        try:
            model_answer = json.loads(model_answer)
            gt_answer = json.loads(gt_answer)
        except json.JSONDecodeError:
            print("Invalid JSON format in model or ground truth answer.")
            return False

        # Both must be arrays of equal length
        if not isinstance(model_answer, list) or not isinstance(gt_answer, list):
            return False
        if len(model_answer) != len(gt_answer):
            return False

        # Compare each step one by one
        for model_item, gt_item in zip(model_answer, gt_answer):
            if not isinstance(model_item, dict) or not isinstance(gt_item, dict):
                return False
            if model_item.get("step") != gt_item.get("step"):
                return False
            if model_item.get("agent") != gt_item.get("agent"):
                return False

            model_inputs = model_item.get("inputs", {})
            gt_inputs = gt_item.get("inputs", {})

            # Deep equality check for the inputs dictionary
            if not self.deep_equal(model_inputs, gt_inputs):
                return False
            
            model_outputs = model_item.get("outputs", [])
            gt_outputs = gt_item.get("outputs", [])

            # Deep equality check for the outputs dictionary
            if not self.deep_equal(model_outputs, gt_outputs):
                return False

        # All steps passed
        return True

    def check_dependencies_correctness(self, model_answer, gt_answer):
        """
        Check whether 'dependence' lists and the *values* of 'dependence_content'
        match between model and ground truth answers for each step.
        """
        if not model_answer:
            return False

        if not isinstance(model_answer, str) or not isinstance(gt_answer, str):
            return False

        # Parse JSON safely
        try:
            model_answer = json.loads(model_answer)
            gt_answer = json.loads(gt_answer)
        except json.JSONDecodeError:
            print("Invalid JSON format in model or ground truth answer.")
            return False

        # Must be lists of equal length
        if not isinstance(model_answer, list) or not isinstance(gt_answer, list):
            return False
        if len(model_answer) != len(gt_answer):
            return False

        # Step-by-step comparison
        for model_item, gt_item in zip(model_answer, gt_answer):
            if not isinstance(model_item, dict) or not isinstance(gt_item, dict):
                return False
            if model_item.get("step") != gt_item.get("step"):
                return False
            if model_item.get("agent") != gt_item.get("agent"):
                return False

            # Check dependence lists
            if not self.deep_equal(
                sorted(self._dependence_as_list(model_item.get("dependence"))),
                sorted(self._dependence_as_list(gt_item.get("dependence"))),
            ):
                return False

            # Extract and sort dependence_content values (ignore keys)
            model_dependence_content = model_item.get("dependence_content")
            gt_dependence_content = gt_item.get("dependence_content")
            
            if model_dependence_content is None and gt_dependence_content is None:
                continue
            elif (model_dependence_content is None) != (gt_dependence_content is None):
                return False

            if not self.deep_equal(model_dependence_content, gt_dependence_content):
                return False

        # If all matched
        return True

    def analyze_error_propagation(self, model_answer, gt_answer):
        """
        First-Point-of-Failure (FPoF) analysis: locate the first incorrect step and error type.
        Returns:
        {
            "is_correct": bool,
            "error_step": int or None,
            "error_type": str or None,
            "details": str
        }
        """
        if not model_answer:
            return {
                "is_correct": False,
                "error_step": 0,
                "error_type": "empty_output",
                "details": "Model produced no answer.",
            }

        try:
            model_list = json.loads(model_answer) if isinstance(model_answer, str) else model_answer
            gt_list = json.loads(gt_answer) if isinstance(gt_answer, str) else gt_answer
        except json.JSONDecodeError:
            return {
                "is_correct": False,
                "error_step": 0,
                "error_type": "format_error",
                "details": "Invalid JSON format.",
            }

        if not isinstance(model_list, list) or not isinstance(gt_list, list):
            return {
                "is_correct": False,
                "error_step": 0,
                "error_type": "format_error",
                "details": "Answer is not a list.",
            }

        min_len = min(len(model_list), len(gt_list))

        # Walk steps in array order to find the First-Point-of-Failure
        #
        # IMPORTANT: align by step index `i`, but still verify that each object's `"step"`
        # matches the ground-truth `"step"` at that position. Otherwise a model can emit
        # wrong `step` ids (e.g., -1 vs 1) while keeping agents/inputs aligned by index,
        # which would incorrectly look like a perfect match if we only compare fields ignoring `step`.
        for i in range(min_len):
            m_item = model_list[i]
            g_item = gt_list[i]

            if not isinstance(m_item, dict) or not isinstance(g_item, dict):
                return {
                    "is_correct": False,
                    "error_step": g_item.get("step", i + 1) if isinstance(g_item, dict) else i + 1,
                    "error_type": "format_error",
                    "details": f"Step item is not an object at index {i}.",
                }

            current_step = g_item.get("step", i + 1)

            # 0. Step id must match GT at the same index
            # NOTE: benchmark JSON uses 0-based step ids.
            if m_item.get("step") != g_item.get("step"):
                return {
                    "is_correct": False,
                    "error_step": current_step,
                    "error_type": "parameter_error",
                    "details": f"Step id mismatch at index {i}: expected step={g_item.get('step')}, got {m_item.get('step')}",
                }

            # 1. Agent mismatch
            if m_item.get("agent") != g_item.get("agent"):
                return {
                    "is_correct": False,
                    "error_step": current_step,
                    "error_type": "agent_mismatch",
                    "details": f"Expected {g_item.get('agent')}, got {m_item.get('agent')}",
                }

            # 2. Input parameter errors
            if not self.deep_equal(m_item.get("inputs", {}), g_item.get("inputs", {})):
                return {
                    "is_correct": False,
                    "error_step": current_step,
                    "error_type": "parameter_error",
                    "details": f"Inputs mismatch at step {current_step}",
                }

            # 2b. Outputs must match GT (check_answers_equivalence compares full steps;
            # FPoF also flags missing/wrong outputs explicitly).
            if not self.deep_equal(m_item.get("outputs", []), g_item.get("outputs", [])):
                return {
                    "is_correct": False,
                    "error_step": current_step,
                    "error_type": "parameter_error",
                    "details": f"Outputs mismatch at step {current_step}",
                }

            # 3. Dependence topology errors
            if not self.deep_equal(
                sorted(self._dependence_as_list(m_item.get("dependence"))),
                sorted(self._dependence_as_list(g_item.get("dependence"))),
            ):
                return {
                    "is_correct": False,
                    "error_step": current_step,
                    "error_type": "dependency_error",
                    "details": f"Dependence logic mismatch at step {current_step}",
                }

            # 4. Dependence content errors
            m_dep_content = m_item.get("dependence_content")
            g_dep_content = g_item.get("dependence_content")
            if (m_dep_content is None) != (g_dep_content is None) or not self.deep_equal(m_dep_content, g_dep_content):
                return {
                    "is_correct": False,
                    "error_step": current_step,
                    "error_type": "dependency_content_error",
                    "details": f"Dependence content mismatch at step {current_step}",
                }

        # Prefix matched but length differs → early stop or extra hallucinated steps
        if len(model_list) < len(gt_list):
            return {
                "is_correct": False,
                "error_step": min_len + 1,
                "error_type": "early_stop",
                "details": f"Model stopped at step {min_len}, expected {len(gt_list)}",
            }
        if len(model_list) > len(gt_list):
            return {
                "is_correct": False,
                "error_step": len(gt_list) + 1,
                "error_type": "hallucinated_extra_steps",
                "details": f"Model generated {len(model_list)} steps, expected {len(gt_list)}",
            }

        return {
            "is_correct": True,
            "error_step": None,
            "error_type": None,
            "details": "Perfect match.",
        }