from collections import defaultdict
from copy import deepcopy
import json
import os
from interfaces.IO_Interface import IO_Interface
from utils.common_utils import read_json, read_yaml
from mcts import MCTS, MCTSNode
import warnings
import random
from tqdm import tqdm
import re

# If every batch of completions fails extract_answer_from_model_completion, _r1_fn would
# loop forever. Cap outer rounds so a pathological model cannot hang the benchmark.
MAX_R1_CONFIDENCE_ROUNDS = 32


def is_terminal_question(prompt, prompt_index):
    question = prompt.split(f"Question {prompt_index}:")[-1].strip()
    if 'Now we can answer' in question:
        return True
    return False


class ReasoningMCTSNode(MCTSNode):
    @property
    def visited(self):
        return self._visited

    def __init__(self, prompt, question_prompt, gen_fn, reward_fn, depth, max_depth, r1_default, r_alpha, prompt_index, 
                 parent: 'ReasoningMCTSNode' = None, r0=0.):
        self._conf = None
        self.children = []
        self.prompt = prompt
        self.question_prompt = question_prompt
        self.gen_fn = gen_fn
        self.reward_fn = reward_fn
        self.depth = depth
        self.max_depth_allowed = max_depth
        self._r0 = r0
        self._r1 = self._r1_default = r1_default
        self._r_alpha = r_alpha
        self._ans_list = None
        self._visited = False
        self.parent = parent
        self._prompt_index = prompt_index

    def _child_node(self, prompt, question_prompt, r0):
        return ReasoningMCTSNode(prompt, question_prompt, self.gen_fn, self.reward_fn, self.depth + 1, self.max_depth_allowed,
                                 self._r1_default, self._r_alpha, self._prompt_index, parent=self, r0=r0)

    def _get_children(self):
        self._visited = True
        self._calculate_reward() # calculates r1 score for the nodes using self.prompt and self.depth, adds the answer to the self.prompt
        print(f"==========\n{'Question 4:' + self.prompt.split('Question 4:', 1)[-1]}\n==========")
        if self.is_terminal: # if terminal, do not generate new children, rather return the children as []
            return self.children
        # self.prompt has the context and the previous depth answer
        questions, question_prompts, r0 = self.gen_fn(self.prompt, self.question_prompt, self.depth)
        for question, qp, r in zip(questions, question_prompts, r0):
            self.children.append(self._child_node(question, qp, r))
        return self.children

    def find_children(self):
        self.children = self.children or self._get_children()
        return self.children

    def find_one_child(self) -> MCTSNode:
        return random.choice(self.find_children())

    def _calculate_reward(self):
        self.prompt, self._r1, self._ans_list = self.reward_fn(self.prompt, self.depth)

    def _static_terminal(self):
        return is_terminal_question(self.prompt, self._prompt_index)

    @property
    def is_terminal(self):
        return self._static_terminal() or self.reward < -1 or self.depth >= self.max_depth_allowed

    @property
    def reward(self):
        if self._r0 < 0 or self._r1 < 0:
            return min(self._r0, self._r1)
        return self._r0 ** self._r_alpha * self._r1 ** (1 - self._r_alpha)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.gen_fn is None or self.reward_fn is None:
            warnings.warn('MCTSNode loaded from pickle is read-only; Do not further roll out the tree!')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['gen_fn'] = None
        state['reward_fn'] = None
        return state



class RAP:
    """
    Reasoning via Planning (RAP) algorithm using MCTS for structured reasoning.
    """

    def __init__(self, io, evaluator, n_sample_subquestion, max_depth, n_sample_confidence, w_exp, r_alpha, r1_default, mcts_rollouts):
        self.io = io
        self.evaluator = evaluator

        # Default hyperparameters
        self.n_sample_subquestion = n_sample_subquestion
        self.max_depth = max_depth
        self.n_sample_confidence = n_sample_confidence
        self.w_exp = w_exp
        self.r_alpha = r_alpha
        self.r1_default = r1_default
        self.mcts_rollouts = mcts_rollouts

        self.prompts = read_yaml('baselines/baseline_prompts/rap_prompts.yaml')["REASONING_VIA_PLANNING"]
        self.tools_list = read_json('interfaces/tools/tools_manifest.json')
        self.prompt_index = self.prompts["prompt_index"]

    def _r0_fn(self, q_inp, questions, depth):
        """Evaluates usefulness of subquestions."""
        new_subq_prefix = f"New Question {self.prompt_index}.{depth}: "
        suffix = "Is the question useful? "

        inputs = [
            q_inp + new_subq_prefix +
            q.replace('Now we can answer the question: ', '') + "\n" + suffix
            for q in questions
        ]

        outputs = self.io.generate(inputs, num_return=1)
        r0 = []
        for out in outputs:
            match = re.search(r'\b(Yes|No)\b', out, re.IGNORECASE)
            if match is None:
                r0.append(0.5)
                continue
            start_index = match.start()
            normalized = out[start_index:].strip()
            
            if normalized.startswith("Yes"):
                r0.append(1.0)
            elif normalized.startswith("No"):
                r0.append(0.0)
            else:
                r0.append(0.5)
        return r0

    def _r1_fn(self, inp, depth):
        """Evaluates confidence of generated answers."""
        if f'Question {self.prompt_index}.' not in inp:
            return 0, inp, []

        answer_prefix = f"Answer {self.prompt_index}.{depth - 1}: "
        world_input = inp + "\n" + answer_prefix

        answer2completions = defaultdict(list)
        all_answers = []

        rounds = 0
        while not answer2completions:
            rounds += 1
            if rounds > MAX_R1_CONFIDENCE_ROUNDS:
                print(
                    f"[RAP] _r1_fn: no parseable answer after {MAX_R1_CONFIDENCE_ROUNDS} rounds "
                    f"(depth={depth}); aborting r1 sampling for this node."
                )
                return 0.0, world_input, []

            completions = self.io.generate(
                model_input=world_input,
                num_return=self.n_sample_confidence,
                stop=[f"Question {self.prompt_index}.{depth}:"]
            )

            for c in completions:
                if not isinstance(c, str) or len(c)==0:
                    continue
                c_str = c.split(f"Answer {self.prompt_index}.{depth-1}:", 1)[-1].split(f"Question {self.prompt_index}.{depth-1}:", 1)[-1].strip()
                sub_answer = self.evaluator.extract_answer_from_model_completion(c_str)
                if not isinstance(sub_answer, str):
                    continue
                all_answers.append(sub_answer)

                matched = False
                for existing in list(answer2completions.keys()):
                    if self.evaluator.check_answers_equivalence(sub_answer, existing):
                        answer2completions[existing].append(c_str)
                        matched = True
                        break
                if not matched:
                    answer2completions[sub_answer].append(c_str)

        most_confident, completions_for_answer = max(
            answer2completions.items(), key=lambda p: len(p[1])
        )

        r1 = len(completions_for_answer) / max(len(all_answers), 1)
        representative_output = completions_for_answer[0]
        return r1, world_input + representative_output, all_answers

    def _reward_fn(self, inp, depth):
        r1, answer, ans_list = self._r1_fn(inp, depth)
        return answer, r1, ans_list

    def _gen_fn(self, inp, q_inp, depth):
        subprefix = f"Question {self.prompt_index}.{depth}: "
        io_input = inp.rstrip() + "\n" + subprefix
        next_depth = depth + 1
        will_be_terminal = (next_depth >= self.max_depth)

        if will_be_terminal:
            io_input += " Now we can answer the question: "

        io_outputs = self.io.generate(
            model_input=io_input,
            num_return=self.n_sample_subquestion,
            stop=["Answer ", f"Answer {self.prompt_index}.{depth}:"]
        )

        questions = [o.split(subprefix)[-1].strip() for o in io_outputs]
        questions = list(dict.fromkeys(questions))  # remove duplicates
        r0 = self._r0_fn(q_inp, questions, depth)

        child_prompts = [io_input + q for q in questions]
        question_outputs = [q_inp.rstrip() + "\n" + subprefix + q for q in questions]

        return child_prompts, question_outputs, r0

    def generate(self, user_question: str) -> str:
        """
        Run Reasoning via Planning (RAP) to answer the user question.
        Returns the final model-generated answer string.
        """
        print(f"Starting RAP reasoning for question: {user_question}")

        input_prompts = (
            self.prompts["subquestion_subanswer"].format(agents_list=self.tools_list)
            + f"\nQuestion {self.prompt_index}: " + user_question.strip() + "\n"
        )
        input_question_prompts = (
            self.prompts["subquestion_usefulness"].format(agents_list=self.tools_list)
            + f"\nQuestion {self.prompt_index}: " + user_question.strip() + "\n"
        )

        # Initialize MCTS
        mcts = MCTS(w_exp=self.w_exp, prior=True, aggr_reward='mean', aggr_child='max')
        root = ReasoningMCTSNode(
            input_prompts, input_question_prompts,
            self._gen_fn, self._reward_fn,
            depth=1, max_depth=self.max_depth,
            r1_default=self.r1_default, r_alpha=self.r_alpha,
            prompt_index=self.prompt_index
        )

        trajs, best_traj, best_r = [], None, float("-inf")

        print("Running MCTS rollouts...")
        for i in range(self.mcts_rollouts):
            mcts.rollout(root)
            max_n, max_r = mcts.max_mean_terminal(root)
            traj = f"Question {self.prompt_index}: " + max_n.prompt.split(f"Question {self.prompt_index}: ")[-1]
            trajs.append(traj)

            if max_r is not None and max_r > best_r:
                best_r = max_r
                best_traj = traj

            print(f"\n===== Rollout {i+1} =====\n{traj}\n=====================")

        print("Rollouts completed.")

        rap_completion = best_traj or (trajs[-1] if trajs else input_prompts)

        matches = re.findall(r"(Answer\s+\d+(?:\.\d+)*:.*?)(?=Question|\Z)", rap_completion, flags=re.S)
        rap_completion = matches[-1].strip() if matches else rap_completion

        rap_answer = self.evaluator.extract_answer_from_model_completion(rap_completion)
        print(f"\nFinal RAP Answer:\n{rap_answer}")
        return rap_answer