import itertools
import json
import re
from evaluators.evaluators import Evaluator
from interfaces.IO_Interface import IO_Interface
from utils.common_utils import read_yaml, read_json

class TreeOfThoughts:
    def __init__(self, io, evaluator, num_generate_sample, num_evaluate_sample, n_select_sample):
        self.io = io
        self.evaluator = evaluator
        self.num_steps = 2
        self.num_generate_sample = num_generate_sample
        self.num_evaluate_sample = num_evaluate_sample
        self.n_select_sample = n_select_sample
        self.prompt = read_yaml("baselines/baseline_prompts/tot_prompts.yaml")["TREE_OF_THOUGHTS"] 
        self.agents_desc = read_json("interfaces/tools/tools_manifest.json")

    @staticmethod
    def _parse_vote_index(output: str, num_choices: int):
        """Parse critic vote index (0-based). Many models ignore the exact template."""
        if not output or num_choices < 1:
            return None
        patterns = [
            r"The best choice is:\s*(\d+)",
            r"best choice is[:\s]+(\d+)",
            r"Best choice[:\s]+(\d+)",
            r"choose\s+(?:choice\s*)?(\d+)",
            r"Choice\s*(\d+)\s+(?:is|wins|best)",
            r"option\s*(\d+)(?:\s|$|,|\.)",
        ]
        for p in patterns:
            m = re.search(p, output, re.IGNORECASE | re.DOTALL)
            if m:
                v = int(m.group(1))
                if 1 <= v <= num_choices:
                    return v - 1
        tail = output[-500:]
        m = re.search(r"The best choice is:\s*(\d+)", tail, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 1 <= v <= num_choices:
                return v - 1
        for line in reversed(output.strip().splitlines()):
            line = line.strip()
            m = re.match(r"^(\d+)\s*\.?$", line)
            if m:
                v = int(m.group(1))
                if 1 <= v <= num_choices:
                    return v - 1
        return None

    def generate_thoughts(self, x, y, step):
        """Generate multiple potential next steps (thoughts)."""
        prompt = self.prompt["generate_prompt"].format(agents_desc = self.agents_desc, user_question = x).rstrip() + y.strip()
        if step == 1:
            prompt += "\n\nThe structured task plan is: "
            # prompt += "\n\n"
        # Step 0: stop before the model emits the final JSON so we can branch in step 1.
        # Step 1: do not pass stop — OpenRouter + strict stops often truncate the JSON or yield junk
        # like "The structured task plan is: the examples."
        stop_tokens = [["The structured task plan is:", "Response:", "Instruction:"], ["Let's think step by step", "Response:", "Instruction:"]]
        stop = stop_tokens[step] if step == 0 else None
        thoughts = self.io.generate(prompt, self.num_generate_sample, stop=stop)
        if not isinstance(thoughts, (list, tuple)):
            thoughts = [thoughts] if thoughts is not None else []

        if step == 1:
            out = []
            for thought in thoughts:
                if thought is None:
                    continue
                t = str(thought).strip()
                if not t:
                    continue
                tail = t.split("The structured task plan is: ")[-1].strip()
                out.append(y.strip() + "\n\nThe structured task plan is: " + tail)
            return out
        return [y + _ for _ in thoughts if _ is not None]
        
    def evaluate_thoughts(self, x, ys, step):
        """Evaluate the promise of a thought."""
        prompt = self.prompt["vote_prompt"]
        if step == 1:
            prompt += " You should only choose an option with a valid structured plan. Make sure that the structured plan of the best choice complies with the provided structure."
        prompt += f"\nAgents Description: {self.agents_desc}"
        prompt += f"\nUser Instruction: {x}"
        prompt += f"\nPlans:"
        for i, y in enumerate(ys, 1):
            prompt += f"Choice {i}: \n{y}\n"
        
        io_outputs = self.io.generate(prompt, self.num_evaluate_sample, stop=None)
        if not isinstance(io_outputs, (list, tuple)):
            io_outputs = [io_outputs] if io_outputs is not None else []

        results = [0] * len(ys)
        for output in io_outputs:
            if not output:
                continue
            idx = self._parse_vote_index(output, len(ys))
            if idx is not None:
                results[idx] += 1
            else:
                snippet = (output[:200] + "…") if len(output) > 200 else output
                print(f"vote no match (showing head): {snippet!r}")
        return results
    
    def generate(self, user_question):
        """Main search loop using a breadth-first search with pruning."""
        x = user_question
        ys = ['']
        for step in range(self.num_steps):
            new_ys = [self.generate_thoughts(x, y, step) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))
            values = self.evaluate_thoughts(x, new_ys, step)
            select_ids = sorted(ids, key=lambda i: values[i], reverse=True)[: self.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]
            if all(v == 0 for v in values) and new_ys:
                for ny in new_ys:
                    if self.evaluator.extract_answer_from_model_completion(ny) is not None:
                        select_new_ys = [ny]
                        break
            ys = select_new_ys
        for y in ys:
            ans = self.evaluator.extract_answer_from_model_completion(y)
            if ans is not None:
                return ans
        return self.evaluator.extract_answer_from_model_completion(ys[0] if ys else "")
