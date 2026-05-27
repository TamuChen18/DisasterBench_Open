#!/usr/bin/env python3
"""
Compute structural complexity statistics of benchmark tasks from ground-truth DAGs.

Metrics (per task):
- linear: no branching and no merging (all nodes have indegree<=1 and outdegree<=1)
- branching: exists a node with outdegree>1
- merging: exists a node with indegree>1

Also reports:
- avg steps per task
- avg branching factor (overall mean outdegree across all nodes)
- avg branching factor (mean outdegree over non-leaf nodes only)
- max outdegree
- avg longest-path length (in nodes) using step-id order as topo order

Usage:
  python3 scripts/compute_structure_complexity.py --data data/benchmark.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class Stats:
    tasks: int = 0
    steps_total: int = 0

    linear_tasks: int = 0
    branching_tasks: int = 0
    merging_tasks: int = 0

    nodes_total: int = 0
    total_outdeg: int = 0
    nonleaf_nodes: int = 0
    nonleaf_outdeg_sum: int = 0
    max_outdeg: int = 0

    longest_path_sum: int = 0

    outdeg_hist: Counter = None  # type: ignore[assignment]
    indeg_hist: Counter = None  # type: ignore[assignment]

    def __post_init__(self):
        self.outdeg_hist = Counter()
        self.indeg_hist = Counter()


def _parse_plan(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan = obj.get("structured_plan")
    if isinstance(plan, list):
        return [x for x in plan if isinstance(x, dict)]
    plan = obj.get("gt_solution")
    if isinstance(plan, list):
        return [x for x in plan if isinstance(x, dict)]
    return []


def _build_graph(plan: List[Dict[str, Any]]) -> Tuple[List[int], Dict[int, List[int]]]:
    steps: List[int] = []
    deps_map: Dict[int, List[int]] = {}
    for s in plan:
        st = s.get("step")
        if not isinstance(st, int):
            continue
        steps.append(st)
        deps = s.get("dependence")
        if not isinstance(deps, list):
            deps = []
        deps_norm = [d for d in deps if isinstance(d, int) and d != -1]
        deps_map[st] = deps_norm
    return steps, deps_map


def _longest_path_len(step_set: set[int], deps_map: Dict[int, List[int]]) -> int:
    # Use sorted step ids as topo order; benchmark deps are backward-pointing.
    order = sorted(step_set)
    dist = {st: 1 for st in order}
    for st in order:
        best = 1
        for d in deps_map.get(st, []):
            if d in dist:
                best = max(best, dist[d] + 1)
        dist[st] = best
    return max(dist.values()) if dist else 0


def compute(data_path: Path) -> Stats:
    st = Stats()
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        plan = _parse_plan(obj)
        if not plan:
            continue

        st.tasks += 1
        steps, deps_map = _build_graph(plan)
        step_set = set(steps)
        nodes = len(step_set)
        st.steps_total += nodes
        st.nodes_total += nodes

        indeg = {s: 0 for s in step_set}
        outdeg = {s: 0 for s in step_set}

        for node, deps in deps_map.items():
            if node not in step_set:
                continue
            indeg[node] = len([d for d in deps if d in step_set])
            for d in deps:
                if d in step_set:
                    outdeg[d] += 1

        for s in step_set:
            st.outdeg_hist[outdeg.get(s, 0)] += 1
            st.indeg_hist[indeg.get(s, 0)] += 1

        st.total_outdeg += sum(outdeg.values())
        st.max_outdeg = max(st.max_outdeg, max(outdeg.values()) if outdeg else 0)

        nonleaf = [v for v in outdeg.values() if v > 0]
        st.nonleaf_nodes += len(nonleaf)
        st.nonleaf_outdeg_sum += sum(nonleaf)

        has_branch = any(v > 1 for v in outdeg.values())
        has_merge = any(v > 1 for v in indeg.values())
        is_linear = (not has_branch) and (not has_merge)

        if is_linear:
            st.linear_tasks += 1
        if has_branch:
            st.branching_tasks += 1
        if has_merge:
            st.merging_tasks += 1

        st.longest_path_sum += _longest_path_len(step_set, deps_map)

    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/benchmark.jsonl")
    args = ap.parse_args()
    data_path = Path(args.data).resolve()
    s = compute(data_path)

    def pct(x: int) -> float:
        return (x / s.tasks * 100.0) if s.tasks else 0.0

    avg_steps = (s.steps_total / s.tasks) if s.tasks else 0.0
    avg_outdeg = (s.total_outdeg / s.nodes_total) if s.nodes_total else 0.0
    avg_nonleaf_outdeg = (s.nonleaf_outdeg_sum / s.nonleaf_nodes) if s.nonleaf_nodes else 0.0
    avg_lp = (s.longest_path_sum / s.tasks) if s.tasks else 0.0

    print(f"tasks: {s.tasks}")
    print(f"avg_steps_per_task: {avg_steps:.6f}")
    print(f"linear_tasks: {s.linear_tasks} ({pct(s.linear_tasks):.2f}%)")
    print(f"branching_tasks: {s.branching_tasks} ({pct(s.branching_tasks):.2f}%)")
    print(f"merging_tasks: {s.merging_tasks} ({pct(s.merging_tasks):.2f}%)")
    print(f"avg_branching_factor_overall_outdeg: {avg_outdeg:.6f}")
    print(f"avg_branching_factor_nonleaf_outdeg: {avg_nonleaf_outdeg:.6f}")
    print(f"max_outdeg: {s.max_outdeg}")
    print(f"avg_longest_path_len: {avg_lp:.6f}")
    print(f"outdeg_hist: {dict(s.outdeg_hist)}")
    print(f"indeg_hist: {dict(s.indeg_hist)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

