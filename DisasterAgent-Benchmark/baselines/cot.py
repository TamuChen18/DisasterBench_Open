from collections import defaultdict
import os
import random
import json
import re
from typing import List, Tuple
from evaluators.evaluators import Evaluator
from utils.common_utils import read_json, read_yaml
import traceback

_JSON_MODE_SUFFIX = (
    "\n\nIMPORTANT (JSON mode): Your entire reply must be one JSON object with no markdown. "
    'Use keys "reasoning" (string, brief chain-of-thought) and "structured_plan" '
    "(array of step objects using the same schema as in the examples)."
)

class ChainOfThoughts:
    def __init__(self, io, evaluator, use_fewshot, num_chain_of_thought, json_mode=False):
        self.io = io
        self.evaluator = evaluator
        self.use_fewshot = use_fewshot
        self.num_chain_of_thought = num_chain_of_thought
        self.json_mode = json_mode
        self.prompt = read_yaml("baselines/baseline_prompts/cot_prompt.yaml")["CHAIN_OF_THOUGHT"]
        self.tools_desc = read_json("interfaces/tools/tools_manifest.json")
        self.last_all_completions = None
        self.last_selected_completion = None
        self._generated_ref_pattern = re.compile(r"^<GENERATED>-(\d+)-<?([^<>]+)>?$")
        self._agent_outputs = {
            agent_name: list(agent_spec.get("output", {}).keys())
            for agent_name, agent_spec in self.tools_desc.items()
        }

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

    def _postprocess_structured_plan(self, answer_text):
        if not isinstance(answer_text, str):
            return answer_text
        try:
            plan = json.loads(answer_text)
        except json.JSONDecodeError:
            return answer_text
        if not isinstance(plan, list):
            return answer_text

        # Build step -> outputs map, filling missing outputs from tool manifest when possible.
        step_to_outputs = {}
        for item in plan:
            if not isinstance(item, dict):
                continue
            step_idx = item.get("step")
            if not isinstance(step_idx, int):
                continue
            outputs = item.get("outputs", [])
            if not isinstance(outputs, list) or len(outputs) == 0:
                agent_name = item.get("agent")
                outputs = self._agent_outputs.get(agent_name, []) if isinstance(agent_name, str) else []
                if outputs:
                    item["outputs"] = outputs
            step_to_outputs[step_idx] = item.get("outputs", []) if isinstance(item.get("outputs", []), list) else []

        for item in plan:
            if not isinstance(item, dict):
                continue
            inputs = item.get("inputs", {})
            if not isinstance(inputs, dict):
                inputs = {}
                item["inputs"] = inputs

            predecessor = None
            consumed_keys = []
            for input_name, input_value in list(inputs.items()):
                normalized_value, parsed_ref = self._normalize_generated_ref(input_value)
                if parsed_ref is None:
                    continue
                ref_step, ref_key = parsed_ref
                valid_outputs = step_to_outputs.get(ref_step, [])
                if isinstance(valid_outputs, list) and valid_outputs and ref_key not in valid_outputs and len(valid_outputs) == 1:
                    ref_key = valid_outputs[0]
                    normalized_value = f"<GENERATED>-{ref_step}-<{ref_key}>"
                inputs[input_name] = normalized_value

                if predecessor is None:
                    predecessor = ref_step
                consumed_keys.append(ref_key)

            if predecessor is None:
                item["dependence"] = [-1]
                item["dependence_content"] = None
            else:
                dedup_keys = []
                for k in consumed_keys:
                    if k not in dedup_keys:
                        dedup_keys.append(k)
                item["dependence"] = [predecessor]
                item["dependence_content"] = {str(predecessor): dedup_keys}

        return json.dumps(plan, ensure_ascii=False)

    def most_likely_answer(self, user_question, io_output_list):
        self.last_all_completions = io_output_list
        self.last_selected_completion = None
        if len(io_output_list) == 1:
            self.last_selected_completion = io_output_list[0]
            most_confident_answer = self.evaluator.extract_answer_from_model_completion(io_output_list[0])
        else:  
            answer2completions = defaultdict(list) 
            for id, c in enumerate(io_output_list):
                try:
                    model_answer = self.evaluator.extract_answer_from_model_completion(c)
                    if model_answer is None:
                        continue
                    has_existed = False
                    for existing_answer in answer2completions.keys():
                        if self.evaluator.check_answers_equivalence(model_answer, existing_answer):
                            assert not has_existed
                            has_existed = True
                            answer2completions[existing_answer].append(c)
                            break
                    if not has_existed:
                        answer2completions[model_answer].append(c)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    continue

            if len(answer2completions.keys()) == 0:
                random_id = random.randrange(len(io_output_list))
                random_completion = io_output_list[random_id]
                self.last_selected_completion = random_completion
                return self._postprocess_structured_plan(random_completion)
            
            if None in answer2completions:
                del answer2completions[None]
            assert len(answer2completions.keys()) > 0, "There are no valid completions."

            most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))
            self.last_selected_completion = answer2completions[most_confident_answer][0]

        return self._postprocess_structured_plan(most_confident_answer)

    def generate(self, user_question):
        if self.use_fewshot:
            examples_str = "Examples: "
            for example in self.prompt["examples"]:
                examples_str += "\n" + f"{example['input']}"
                examples_str += "\n" + f"{example['output']}"
        else:
            examples_str = ""

        system_prompt = self.prompt["system_prompt"].format(agents_desc = self.tools_desc, examples = examples_str) #.replace("{{", "{").replace("}}", "}")
        user_prompt = self.prompt["user_prompt"].format(task_desc = user_question) #.replace("{{", "{").replace("}}", "}")
        if self.json_mode:
            user_prompt = user_prompt + _JSON_MODE_SUFFIX

        io_input = system_prompt + "\n" + user_prompt

        io_output_list = self.io.generate(io_input, num_return=self.num_chain_of_thought)
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list] 

        return self.most_likely_answer(user_question, cleaned_io_output_list)