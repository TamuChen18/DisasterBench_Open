#!/usr/bin/env python3
"""
Run non-LLM structural baselines on DisasterBench and write the same
`*_detailed_predictions.json` format as LLM runs (for Evaluator / FPoF scripts).

Baselines (pick one per run via --baseline):
  - oracle_random: GT agents + deps/outputs; randomize literal inputs (keep <GENERATED>-... refs).
  - dag_greedy_tfidf: union DAG from graph_desc (collapsed agent graph) + word-overlap
    similarity to task_desc; plan length = len(GT).
  - dag_beam_tfidf: beam search over DAG-valid agent chains using the same word-overlap
    similarity; plan length = len(GT). (Default beam_width=3, topk_per_step=8, global_topk=15)
  - shortest_path: shortest collapsed-agent path from GT[0].agent to GT[-1].agent,
    pad/truncate to len(GT).

Data: always `data/benchmark.jsonl` (full DisasterBench; train/test splits are not used).

Usage (repo root):
  PYTHONPATH=. python3 scripts/run_structural_baselines.py --baseline oracle_random
  PYTHONPATH=. python3 scripts/run_structural_baselines.py --baseline dag_greedy_tfidf --seed 0
  PYTHONPATH=. python3 scripts/run_structural_baselines.py --baseline shortest_path

Outputs (under a dedicated tree, separate from LLM `results/<model>/...`):
  results/structural_baselines/<baseline>/<baseline>_detailed_predictions.json
  results/structural_baselines/<baseline>/<baseline>_result.txt
  results/structural_baselines/<baseline>/config.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_JSONL = REPO_ROOT / "data" / "benchmark.jsonl"

# Output root: results_root/structural_baselines/<baseline>/
STRUCTURAL_BASELINES_SUBDIR = "structural_baselines"

BASELINE_CHOICES = ("oracle_random", "dag_greedy_tfidf", "dag_beam_tfidf", "shortest_path")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_plan(gt: Any) -> List[Dict[str, Any]]:
    if isinstance(gt, list):
        return [x for x in gt if isinstance(x, dict)]
    if isinstance(gt, str):
        try:
            obj = json.loads(gt)
            return [x for x in obj if isinstance(x, dict)]
        except json.JSONDecodeError:
            return []
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        choices=BASELINE_CHOICES,
        required=True,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beam_width", type=int, default=3, help="Beam width for dag_beam_tfidf.")
    ap.add_argument("--topk_per_step", type=int, default=8, help="Top-k candidates per step for dag_beam_tfidf.")
    ap.add_argument(
        "--global_topk",
        type=int,
        default=15,
        help="Restrict dag_beam_tfidf to top-K globally similar agents (set <=0 to disable).",
    )
    ap.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Directory under repo root (default: results).",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from evaluators.evaluators import Evaluator  # noqa: E402
    from baselines.structural_baselines import (  # noqa: E402
        build_agent_desc_vectors,
        dag_beam_tfidf_plan,
        dag_greedy_tfidf_plan,
        load_collapsed_agent_graph,
        oracle_random_params_plan,
        shortest_path_plan,
    )

    rng = random.Random(args.seed)
    method = args.baseline  # matches folder name and output file prefix

    manifest_path = REPO_ROOT / "interfaces" / "tools" / "tools_manifest.json"
    graph_path = REPO_ROOT / "interfaces" / "tools" / "graph_desc.json"
    data_path = BENCHMARK_JSONL
    if not data_path.is_file():
        print(f"ERROR: benchmark data not found: {data_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    agents, succ, pred = load_collapsed_agent_graph(graph_path)
    agent_vecs = build_agent_desc_vectors(manifest)

    data = _load_jsonl(data_path)
    ev = Evaluator()

    detailed: List[Dict[str, Any]] = []
    total_correct = total_tools = total_params = total_deps = 0

    for item in data:
        task_id = item.get("task_id")
        task_desc = item.get("task_desc") or ""
        gt_solution = item.get("structured_plan")
        gt_plan = _parse_plan(gt_solution)
        gt_answer = ev.extract_answer_from_gold_solution(gt_solution)

        if args.baseline == "oracle_random":
            plan = oracle_random_params_plan(gt_plan, rng)
        elif args.baseline == "dag_greedy_tfidf":
            T = len(gt_plan)
            plan = dag_greedy_tfidf_plan(task_desc, T, manifest, agents, succ, pred, rng, agent_vecs)
        elif args.baseline == "dag_beam_tfidf":
            T = len(gt_plan)
            plan = dag_beam_tfidf_plan(
                task_desc,
                T,
                manifest,
                agents,
                succ,
                pred,
                rng,
                agent_vecs,
                beam_width=args.beam_width,
                topk_per_step=args.topk_per_step,
                global_topk=(None if args.global_topk is None or args.global_topk <= 0 else args.global_topk),
            )
        else:
            plan = shortest_path_plan(gt_plan, manifest, agents, succ, rng)

        model_answer = json.dumps(plan, ensure_ascii=False)
        eval_answer = ev.normalize_answer_for_evaluation("dp", model_answer)

        correct = ev.check_answers_equivalence(eval_answer, gt_answer)
        tools_correct = ev.check_tools_correctness(eval_answer, gt_answer)
        parameters_correct = ev.check_parameters_correctness(eval_answer, gt_answer)
        dependencies_correct = ev.check_dependencies_correctness(eval_answer, gt_answer)
        error_analysis = ev.analyze_error_propagation(eval_answer, gt_answer)

        total_correct += int(correct)
        total_tools += int(tools_correct)
        total_params += int(parameters_correct)
        total_deps += int(dependencies_correct)

        fpof_type = None if error_analysis.get("is_correct") else error_analysis.get("error_type")
        row = {
            "task_id": task_id,
            "task_desc": task_desc,
            "gt_solution": gt_solution,
            "model_answer": eval_answer,
            "raw_selected_completion": None,
            "metrics": {
                "is_perfect_match": bool(correct),
                "tools_correct": bool(tools_correct),
                "parameters_correct": bool(parameters_correct),
                "dependencies_correct": bool(dependencies_correct),
            },
            "error_analysis": error_analysis,
            "fpof_error_step": error_analysis.get("error_step"),
            "fpof_error_type": fpof_type,
            "structural_baseline": {
                "name": args.baseline,
                "seed": args.seed,
                "data_file": "data/benchmark.jsonl",
            },
        }
        detailed.append(row)

    n = len(detailed)
    out_dir = (
        REPO_ROOT / args.results_root / STRUCTURAL_BASELINES_SUBDIR / method
    ).resolve()
    os.makedirs(out_dir, exist_ok=True)

    overall_acc = (total_correct / n * 100) if n else 0.0
    tools_acc = (total_tools / n * 100) if n else 0.0
    params_acc = (total_params / n * 100) if n else 0.0
    deps_acc = (total_deps / n * 100) if n else 0.0

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "baseline": args.baseline,
                "data_file": "data/benchmark.jsonl",
                "seed": args.seed,
                "n_tasks": n,
                "graph_desc": str(graph_path.relative_to(REPO_ROOT)),
                "output_directory": str(
                    out_dir.relative_to(REPO_ROOT)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (out_dir / f"{method}_result.txt").write_text(
        "\n".join(
            [
                f"Baseline: {args.baseline}",
                f"Output: {out_dir.relative_to(REPO_ROOT)}",
                "Data: data/benchmark.jsonl",
                f"Seed: {args.seed}",
                f"Num tested: {n}",
                f"Num correct: {total_correct}",
                f"Overall Accuracy: {overall_acc:.2f}",
                f"Tools Accuracy: {tools_acc:.2f}",
                f"Parameter Accuracy: {params_acc:.2f}",
                f"Dependency Accuracy: {deps_acc:.2f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_json = out_dir / f"{method}_detailed_predictions.json"
    out_json.write_text(json.dumps(detailed, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    print(f"Wrote {out_json.relative_to(REPO_ROOT)} ({n} tasks)")
    print(
        f"EM {overall_acc:.2f}% | tools {tools_acc:.2f}% | params {params_acc:.2f}% | deps {deps_acc:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
