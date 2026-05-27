import os
from datetime import datetime
from argparse import ArgumentParser
from utils.common_utils import save_json

def parse_args():
    """Function to parse command line arguments.
    Returns:
        Namespace: Parsed command line arguments.
    """
    
    parser = ArgumentParser()
    
    #general arguments
    parser.add_argument("--mode", type=str, default="mcts", choices=["mcts", "distill", "distilled_mcts", "eval", "baseline"])
    parser.add_argument("--max_threads", type=int, default=32, help="Maximum number of threads per process")
    parser.add_argument("--if_use_cards", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--task_num", type=int, default=0)
    parser.add_argument(
        "--task_id_file",
        type=str,
        default=None,
        help="Optional JSON file: list of task_id or {\"task_ids\": [...]} to run a subset of benchmark (e.g. dev split). "
        "When set, use with --task_num 0.",
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument(
        "--train_test_json_data",
        type=str,
        default="benchmark.jsonl",
        choices=["train.jsonl", "test.jsonl", "benchmark.jsonl"],
    )
    parser.add_argument("--verbose", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar in test_baseline.py",
    )
    parser.add_argument(
        "--json_mode",
        action="store_true",
        help="Request API JSON object response (response_format type=json_object). "
        "CoT prompts ask for {reasoning, structured_plan}.",
    )
    parser.add_argument(
        "--results_method_name",
        type=str,
        default=None,
        help="Override results/<model>/<name>/ folder (default: cot_json_mode when --json_mode).",
    )

    #model arguments
    allowed_apis = ["openai", "openrouter", "huggingface", "vllm"]
    parser.add_argument("--api", type=str, choices=allowed_apis, default="openai", help=f"API to use: Choose from {allowed_apis}.")
    parser.add_argument("--model_ckpt", type=str, default="gpt-4o-mini", help="path of the model to use")
    parser.add_argument("--tokenizer_ckpt", type=str, default=None, help="path of the tokenizer to use")
    parser.add_argument("--sim_model", default="sentence-transformers/all-mpnet-base-v2") 
    parser.add_argument("--max_tokens", type=int, default=2048, help="max_tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--top_k", type=int, default=20, help="top_k")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p")

    #directory arguments
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to save model artifacts")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Baseline outputs root: results/<model_ckpt slashes to hyphens>/<test_type>/ (cot|tot|rap)",
    )
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Directory to save model outputs")
    parser.add_argument("--prompts_dir", type=str, default="src/prompts", help="Directory to load prompts")
    parser.add_argument("--tools_dir", type=str, default="interfaces/tools", help="Directory to load tools")

    #mcts generator arguments
    parser.add_argument("--num_rollouts", type=int, default=5, help="Number of rollouts for MCTS")
    parser.add_argument("--max_depth_allowed", type=int, default=5)
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)  
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const") 

    #reasoning arguments
    parser.add_argument("--disable_rephrase", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 1: rephrase")
    parser.add_argument("--disable_direct_answer", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 2: direct answer")
    parser.add_argument("--disable_chain_of_thought", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 3: chain-of-thought")
    parser.add_argument("--disable_divide_and_conquer", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 4: divide and conquer")
    parser.add_argument("--disable_self_refine", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 5: self-reflection and refinement")

    parser.add_argument("--num_direct_answer", type=int, default=10)
    parser.add_argument("--num_chain_of_thought", type=int, default=16)
    parser.add_argument("--num_divide_and_conquer", type=int, default=10, help="Number of trials for 'divide and conquer'.")
    parser.add_argument("--num_divide_and_conquer_votes", type=int, default=10, help="Number of trials for subquestions of each question.")
    parser.add_argument("--use_fewshot", type=lambda x: (str(x).lower() == 'true'), default=True)

    parser.add_argument("--enable_potential_score", default=False)
    parser.add_argument("--disable_answer_selection", default=False)

    # eval mode args
    parser.add_argument("--test_type", choices=["cot", "tot", "rap", "dp", "react", "sc"], default="cot") 

    # TOT args
    parser.add_argument("--num_generate_sample", type=int, default=7)
    parser.add_argument("--num_evaluate_sample", type=int, default=5)
    parser.add_argument("--n_select_sample", type=int, default=1)

    # RAP args
    parser.add_argument("--n_sample_subquestion", type=int, default=16)
    parser.add_argument("--n_sample_confidence", type=int, default=32)
    parser.add_argument("--w_exp", type=int, default=1)
    parser.add_argument("--r_alpha", type=int, default=0.5)
    parser.add_argument("--r1_default", type=int, default=1)

    # Distillation args
    parser.add_argument("--k", type=float, default=0.8)
    parser.add_argument("--attribute_type", type=str, default='semantic', choices=['condition', 'semantic', 'rerank'])
    parser.add_argument("--similarity_type", type=str, default='average', choices=['average', 'max'])
    parser.add_argument("--structure_dir", type=str, default="structure")
    parser.add_argument("--file_dir", type=str, default="artifacts/mcts/")

    #Distilled MCTS args
    parser.add_argument("--reuse_dir", type=str, default="artifacts/structure/")
    parser.add_argument("--reuse_rollouts", type=int, default=1)
    parser.add_argument("--num_cards", type=int, default=10)

    return parser.parse_args()

def post_process_args(args):
    model_name = args.model_ckpt.split("/")[-1]
    sim_model = args.sim_model.split("/")[-1]
    args.file = args.train_test_json_data.split(".")[0]
    suffix = f"{args.file}_|_rolls_{args.num_rollouts}_|_reuse_train_{args.if_use_cards}_|_re_{not args.disable_rephrase}_|_da_{not args.disable_direct_answer}_n_{args.num_direct_answer}_|_cot_{not args.disable_chain_of_thought}_|_dc_{not args.disable_divide_and_conquer}_|_sr_{not args.disable_self_refine}_|_{sim_model}"
    
    # Ensure directories exist
    args.artifacts_dir = os.path.join(args.artifacts_dir)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    # Baselines temperature policy for reproducible benchmarking:
    # - cot/react/dp/sc: deterministic (temperature=0)
    # - tot/rap: mildly stochastic search (temperature=0.5)
    test_type = getattr(args, "test_type", None)
    if test_type in {"cot", "react", "dp", "sc"}:
        args.temperature = 0.0
    elif test_type in {"tot", "rap"}:
        args.temperature = 0.5

    if args.mode == "mcts":
        # Flat layout: results/<sanitized_model>/<test_type>/ (no timestamp; cot | tot | rap)
        sanitized_model = args.model_ckpt.replace("/", "-").replace(":", "_")
        method_name = args.test_type
        if getattr(args, "json_mode", False):
            method_name = args.results_method_name or "cot_json_mode"
        args.run_outputs_dir = os.path.join(
            args.results_dir,
            sanitized_model,
            method_name,
        )
        os.makedirs(args.run_outputs_dir, exist_ok=True)
        args.logdir = os.path.join(args.run_outputs_dir, "mcts_logs.log")
        args.answer_sheets_dir = os.path.join(args.run_outputs_dir, "answer_sheets")
        os.makedirs(args.answer_sheets_dir, exist_ok=True)
        save_json(vars(args), os.path.join(args.run_outputs_dir, "config.json"))

    elif args.mode == "distill":
        args.distill_outputs_dir = os.path.join(
            args.artifacts_dir,
            args.structure_dir,
            model_name,
            suffix,
            f"{datetime.now().strftime('%m-%d_%H-%M')}"
        )
        args.root_dir = os.path.dirname(args.file_dir)
        os.makedirs(args.distill_outputs_dir, exist_ok=True)
        args.logdir = os.path.join(args.distill_outputs_dir, "distill_logs.log")

    elif args.mode == "distilled_mcts":
        if args.if_use_cards:
            if args.attribute_type == "condition":
                suffix = f"{args.file}_|_rolls_{args.num_rollouts}_|_reuse_train_{args.if_use_cards}_|_reuse_rolls_{args.reuse_rollouts}_|_{args.attribute_type}_|_reuse_paths_{args.num_cards}_|_re_{not args.disable_rephrase}_|_da_{not args.disable_direct_answer}_n_{args.num_direct_answer}_|_cot_{not args.disable_chain_of_thought}_|_dc_{not args.disable_divide_and_conquer}_|_sr_{not args.disable_self_refine}"
                args.reuse_dir = os.path.join(args.reuse_dir, f"data_|_{args.attribute_type}_|_test_question_path.json")  
            
            elif args.attribute_type in ["semantic", "rerank"]:
                suffix = f"{args.file}_|_rolls_{args.num_rollouts}_|_reuse_train_{args.if_use_cards}_|_reuse_rolls_{args.reuse_rollouts}_|_{args.attribute_type}_{args.similarity_type}_|_reuse_paths_{args.num_cards}_|_re_{not args.disable_rephrase}_|_da_{not args.disable_direct_answer}_n_{args.num_direct_answer}_|_cot_{not args.disable_chain_of_thought}_|_dc_{not args.disable_divide_and_conquer}_|_sr_{not args.disable_self_refine}"
                args.reuse_dir = os.path.join(args.reuse_dir, f"data_|_{args.attribute_type}_{args.similarity_type}_|_test_question_path.json")

        args.run_outputs_dir = os.path.join(
            args.artifacts_dir,
            args.mode,
            model_name,
            suffix,
            f"{datetime.now().strftime('%m-%d_%H-%M')}"
        )
        os.makedirs(args.run_outputs_dir, exist_ok=True)
        args.logdir = os.path.join(args.run_outputs_dir, "distilled_mcts_logs.log")
        args.answer_sheets_dir = os.path.join(args.run_outputs_dir, "answer_sheets")
        os.makedirs(args.answer_sheets_dir, exist_ok=True)
        save_json(vars(args), os.path.join(args.run_outputs_dir, "config.json"))

    elif args.mode == "eval":
        args.run_outputs_dir = os.path.join(
            args.artifacts_dir,
            args.mode,
            model_name,
            suffix,
            f"{datetime.now().strftime('%m-%d_%H-%M')}"
        )
        os.makedirs(args.run_outputs_dir, exist_ok=True)
        args.distill_outputs_dir = args.run_outputs_dir
        args.root_dir = os.path.dirname(args.file_dir)
        args.if_use_cards = True   
        args.logdir = os.path.join(args.run_outputs_dir, "eval_logs.log")
        if args.attribute_type == "condition":
            args.reuse_dir = os.path.join(args.run_outputs_dir, f"data_|_{args.attribute_type}_|_test_question_path.json")
        else:
            args.reuse_dir = os.path.join(args.run_outputs_dir, f"data_|_{args.attribute_type}_{args.similarity_type}_|_test_question_path.json")
    
    elif args.mode == "baseline":
        args.run_outputs_dir = os.path.join(
            args.artifacts_dir,
            args.mode,
            model_name,
            suffix,
            f"{datetime.now().strftime('%m-%d_%H-%M')}"
        )
        os.makedirs(args.run_outputs_dir, exist_ok=True)
        if args.test_type == "cot":
            args.num_chain_of_thought = 1 
        args.logdir = os.path.join(args.run_outputs_dir, f"{args.mode}_logs.log")
        save_json(vars(args), os.path.join(args.run_outputs_dir, "config.json"))

    else:
        raise ValueError(f"Invalid mode: {args.mode}")       

    return args
