#python test/test_generate.py --train_test_json_data train.jsonl --max_threads 1 --num_chain_of_thought 2 --use_fewshot True

import os
import sys
import gc
import time
import traceback
import json

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from cot import ChainOfThoughts
from tot import TreeOfThoughts
from dp import DirectPlanning
from react import ReActPlanning

from utils.common_utils import read_jsonl
from config.args import parse_args, post_process_args
from interfaces.IO_Interface import IO_Interface
from evaluators.evaluators import Evaluator

import multiprocessing
import concurrent.futures
try:
    import torch
except ModuleNotFoundError:
    torch = None
import copy
try:
    import wandb
except ModuleNotFoundError:
    wandb = None


def worker_process(args, data_item, model_name, tokenizer_name):
    """
    Function executed in each worker process.
    Reconstructs Evaluator and IO locally.
    """

    local_args = copy.deepcopy(args)

    generation_kwargs = {
            "max_tokens": local_args.max_tokens,
            "temperature": local_args.temperature,
            "top_p": local_args.top_p
        }
    if getattr(local_args, "json_mode", False):
        generation_kwargs["response_format"] = {"type": "json_object"}
        
    io = IO_Interface(local_args.api, model_name, tokenizer_name, generation_kwargs)
    evaluator = Evaluator()

    json_mode = getattr(local_args, "json_mode", False)
    if local_args.test_type in ["cot", "sc"]:
        baseline_model = ChainOfThoughts(
            io, evaluator, local_args.use_fewshot, local_args.num_chain_of_thought, json_mode=json_mode
        )
    elif local_args.test_type == "tot":
        baseline_model = TreeOfThoughts(io, evaluator, local_args.num_generate_sample, local_args.num_evaluate_sample, local_args.n_select_sample)
    elif local_args.test_type == "dp":
        baseline_model = DirectPlanning(io, evaluator, local_args.use_fewshot)
    elif local_args.test_type == "react":
        baseline_model = ReActPlanning(io, evaluator, local_args.use_fewshot, num_samples=1)
    elif local_args.test_type == "rap":
        from rap import RAP
        baseline_model = RAP(io, evaluator, local_args.n_sample_subquestion, local_args.max_depth_allowed, local_args.n_sample_confidence, local_args.w_exp, local_args.r_alpha, local_args.r1_default, local_args.num_rollouts)
    else:
        print(f"Test type {local_args.test_type} is not implemented.")

            
    task_id = data_item.get("task_id")
    task_desc = data_item.get("task_desc")
    gt_solution = data_item.get("structured_plan")
    gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)

    try:
        model_answer = baseline_model.generate(task_desc)
        raw_selected_completion = getattr(baseline_model, "last_selected_completion", None)
        eval_answer = evaluator.normalize_answer_for_evaluation(
            local_args.test_type, model_answer
        )
        correct = evaluator.check_answers_equivalence(eval_answer, gt_answer)
        tools_correct = evaluator.check_tools_correctness(eval_answer, gt_answer)
        parameters_correct = evaluator.check_parameters_correctness(eval_answer, gt_answer)
        dependencies_correct = evaluator.check_dependencies_correctness(eval_answer, gt_answer)

        error_analysis = evaluator.analyze_error_propagation(eval_answer, gt_answer)
        return (eval_answer, raw_selected_completion, correct, tools_correct, parameters_correct, dependencies_correct, error_analysis)
    except Exception as e:
        traceback.format_exc()
        fail_analysis = {
            "is_correct": False,
            "error_step": 0,
            "error_type": "exception",
            "details": str(e),
        }
        return None, None, False, False, False, False, fail_analysis

def worker_thread(args, data_item, model_name, tokenizer_name):
    """
    Thread worker for OpenAI.
    Reuses evaluator and generator passed from the main thread.
    """
    local_args = copy.deepcopy(args)
    generation_kwargs = {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
    if getattr(local_args, "json_mode", False):
        generation_kwargs["response_format"] = {"type": "json_object"}
    io = IO_Interface(local_args.api, model_name, tokenizer_name, generation_kwargs)
    evaluator = Evaluator()

    json_mode = getattr(local_args, "json_mode", False)
    if local_args.test_type in ["cot", "sc"]:
        baseline_model = ChainOfThoughts(
            io, evaluator, local_args.use_fewshot, local_args.num_chain_of_thought, json_mode=json_mode
        )
    elif local_args.test_type == "tot":
        baseline_model = TreeOfThoughts(io, evaluator, local_args.num_generate_sample, local_args.num_evaluate_sample, local_args.n_select_sample)
    elif local_args.test_type == "dp":
        baseline_model = DirectPlanning(io, evaluator, local_args.use_fewshot)
    elif local_args.test_type == "react":
        baseline_model = ReActPlanning(io, evaluator, local_args.use_fewshot, num_samples=1)
    elif local_args.test_type == "rap":
        from rap import RAP
        baseline_model = RAP(io, evaluator, local_args.n_sample_subquestion, local_args.max_depth_allowed, local_args.n_sample_confidence, local_args.w_exp, local_args.r_alpha, local_args.r1_default, local_args.num_rollouts)
    else:
        print(f"Test type {local_args.test_type} is not implemented.")

    task_id = data_item.get("task_id")
    task_desc = data_item.get("task_desc")
    gt_solution = data_item.get("structured_plan")
    gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)

    try:
        model_answer = baseline_model.generate(task_desc)
        raw_selected_completion = getattr(baseline_model, "last_selected_completion", None)
        eval_answer = evaluator.normalize_answer_for_evaluation(
            local_args.test_type, model_answer
        )

        correct = evaluator.check_answers_equivalence(eval_answer, gt_answer)
        tools_correct = evaluator.check_tools_correctness(eval_answer, gt_answer)
        parameters_correct = evaluator.check_parameters_correctness(eval_answer, gt_answer)
        dependencies_correct = evaluator.check_dependencies_correctness(eval_answer, gt_answer)

        error_analysis = evaluator.analyze_error_propagation(eval_answer, gt_answer)
        return (eval_answer, raw_selected_completion, correct, tools_correct, parameters_correct, dependencies_correct, error_analysis)

    except Exception as e:
        traceback.format_exc()
        print(f"Error in worker_thread task {task_id}: {e}")
        fail_analysis = {
            "is_correct": False,
            "error_step": 0,
            "error_type": "exception",
            "details": str(e),
        }
        return None, None, False, False, False, False, fail_analysis

def main(args):
    """
    Main function to handle the execution of the script based on parsed arguments.
    Args:
        args (Namespace): Parsed command line arguments.
    Returns:
        None
    """

    if load_dotenv is not None:
        load_dotenv()

    if wandb is None:
        class _NoWandbRun:
            def log(self, *_args, **_kwargs):
                return None
            def finish(self):
                return None
        run = _NoWandbRun()
    else:
        run = wandb.init(entity="agents-research", project="disaster-management-agent", config=args)

    assert args.api in ["openai", "openrouter", "huggingface", "vllm"], "Only OpenAI/OpenRouter, vLLM and HuggingFace models are supported."
    
    num_gpus = (
        torch.cuda.device_count()
        if (torch is not None and args.api in ["huggingface"] and torch.cuda.is_available())
        else 0
    )

    if args.api in ["openai", "openrouter"]:
        model_name = args.model_ckpt
        tokenizer_name = None
        ExecutorClass = concurrent.futures.ThreadPoolExecutor
        num_workers = args.max_threads if args.max_threads > 0 else 1

    elif args.api in ["huggingface", "vllm"]:
        multiprocessing.set_start_method("spawn", force=True)
        model_name = args.model_ckpt
        tokenizer_name = args.tokenizer_ckpt or args.model_ckpt
        num_workers = 20
        ExecutorClass = concurrent.futures.ProcessPoolExecutor

    test_file = os.path.join(args.data_root, args.train_test_json_data)
    if args.task_num == 0:
        data_item_list = read_jsonl(test_file)
    else:
        data_item_list = read_jsonl(test_file)[args.task_num - 1 : args.task_num]

    if getattr(args, "task_id_file", None):
        if args.task_num != 0:
            raise ValueError("--task_id_file requires --task_num 0 (full file then subset).")
        with open(args.task_id_file, "r", encoding="utf-8") as tf:
            raw = json.load(tf)
        if isinstance(raw, dict) and "task_ids" in raw:
            id_list = raw["task_ids"]
        else:
            id_list = raw
        id_set = {int(x) for x in id_list}
        data_item_list = [x for x in data_item_list if x.get("task_id") in id_set]
        data_item_list.sort(key=lambda x: x.get("task_id", 0))
    
    total_correct = 0
    total_correct_tools = 0
    total_correct_parameters = 0
    total_correct_dependencies = 0
    num_tested = 0
    detailed_predictions_log = []

    start_time = time.time()
    
    with ExecutorClass(max_workers=num_workers) as executor:
        futures = {}
        for i, data_item in enumerate(data_item_list):
            try:
                if args.api in ["huggingface", "vllm"]:
                    futures[executor.submit(worker_process, args, data_item, model_name, tokenizer_name)] = data_item
                else:
                    futures[executor.submit(worker_thread, args, data_item, model_name, tokenizer_name)] = data_item
            except Exception as e:
                # keep original behaviour of skipping failures
                print(f"Error on item {i}: {e}")
                continue
        
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        completed = concurrent.futures.as_completed(futures)
        n_futures = len(futures)
        if tqdm is not None and not getattr(args, "no_progress", False) and n_futures > 0:
            completed = tqdm(
                completed,
                total=n_futures,
                desc=f"{args.test_type}",
                unit="task",
                mininterval=0.2,
                file=sys.stderr,
            )

        for i, future in enumerate(completed):
            try:
                model_answer, raw_selected_completion, correct, tools_correct, parameters_correct, dependencies_correct, error_analysis = future.result()
            except Exception as e:
                print(f"Future {i} failed: {e}")
                traceback.format_exc()
                continue

            print("="*50)
            
            print(f"get result for {futures[future].get('task_id')}!\n")
            print(f"Task {futures[future].get('task_id')} Ground Truth: {futures[future].get('structured_plan')}\n")
            print(f"Task {futures[future].get('task_id')} Model Answer: {model_answer}\n")

            num_tested += 1
            total_correct += int(correct)
            total_correct_tools += int(tools_correct)
            total_correct_parameters += int(parameters_correct)
            total_correct_dependencies += int(dependencies_correct)

            detailed_predictions_log.append({
                "task_id": futures[future].get("task_id"),
                "task_desc": futures[future].get("task_desc"),
                "gt_solution": futures[future].get("structured_plan"),
                "model_answer": model_answer,
                "raw_selected_completion": raw_selected_completion,
                "metrics": {
                    "is_perfect_match": correct,
                    "tools_correct": tools_correct,
                    "parameters_correct": parameters_correct,
                    "dependencies_correct": dependencies_correct,
                },
                "error_analysis": error_analysis,
            })

            print(f"Overall Accuracy: {(total_correct/(num_tested))*100:.2f}\n")
            print(f"Tools Accuracy: {(total_correct_tools/(num_tested))*100:.2f}\n")
            print(f"Parameter Accuracy: {(total_correct_parameters/(num_tested))*100:.2f}\n")
            print(f"Dependency Accuracy: {(total_correct_dependencies/(num_tested))*100:.2f}\n") 

            print("="*50)
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    average_time = elapsed_time / num_tested if num_tested > 0 else 0
    
    minutes, seconds = divmod(elapsed_time, 60)
    average_mins, average_secs = divmod(average_time, 60)

    if num_tested > 0:
        overall_acc = (total_correct / num_tested) * 100
        tools_acc = (total_correct_tools / num_tested) * 100
        params_acc = (total_correct_parameters / num_tested) * 100
        deps_acc = (total_correct_dependencies / num_tested) * 100
    else:
        overall_acc = tools_acc = params_acc = deps_acc = 0

    run.log({
        "tools_accuracy": f"{tools_acc:.2f}",
        "parameters_accuracy": f"{params_acc:.2f}",
        "dependencies_accuracy": f"{deps_acc:.2f}",
        "overall_accuracy": f"{overall_acc:.2f}",
        "total_execution_time": f"{minutes} min {seconds: .2f} sec",
        "average_execution_time": f"{average_mins} min {average_secs: .2f} sec",
    })

    print("Execution completed successfully.")
    print(f"Time taken to complete execution: {minutes} min {seconds: .2f} sec")
    print(f"Average time taken: {average_mins} min {average_secs: .2f} sec")

    result_path = os.path.join(args.run_outputs_dir, f"{args.test_type}_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Num tested: {num_tested}\n")
        f.write(f"Num correct: {total_correct}\n")
        f.write(f"Overall Accuracy: {overall_acc:.2f}\n")
        f.write(f"Tools Accuracy: {tools_acc:.2f}\n")
        f.write(f"Parameter Accuracy: {params_acc:.2f}\n")
        f.write(f"Dependency Accuracy: {deps_acc:.2f}\n")
        f.write(f"Time taken to complete execution: {minutes} min {seconds: .2f} sec\n")
        f.write(f"Average time taken: {average_mins} min {average_secs: .2f} sec\n")

    output_json_path = os.path.join(args.run_outputs_dir, f"{args.test_type}_detailed_predictions.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_predictions_log, f, indent=4, ensure_ascii=False)
    print(f"\n[Success] Detailed predictions and error analysis saved to: {output_json_path}")

    run.finish()

if __name__ == "__main__":
    args = parse_args()
    args = post_process_args(args)
    print(args)
    main(args)