"""
Non-LLM structural baselines for DisasterBench (paper / ablation).

1) oracle_random_params: GT agent order + deps/outputs; randomize literal inputs
   (keeps <GENERATED>-... refs so the plan remains structurally wired).

2) dag_greedy_tfidf: union collapsed agent DAG from graph_desc.json; step-0 picks
   among zero-in-degree agents; later picks among successors of previous choice;
   tie-break / scoring via lightweight word-overlap cosine to task_desc.
   Plan length matches len(GT) for fair EM comparison.

3) shortest_path: directed shortest agent-chain from GT[0].agent to GT[-1].agent
   on the same collapsed graph; truncate/pad to len(GT).

4) dag_beam_tfidf: DAG-aware beam search (width=k) over agent sequences. Each step
   chooses among valid successors (or fallbacks) and scores by lightweight word-overlap
   cosine similarity between task_desc and agent descriptions. Plan length matches len(GT).

All builders return a JSON array string compatible with Evaluator.normalize_answer_for_evaluation("dp", ...).
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Match GT-style refs, e.g. <GENERATED>-0-<predicted_radar_echo_frames_path>
_GEN_RE = re.compile(r"^<GENERATED>-(\d+)-<?([^<>]+)>?$")


def iter_endpoints(v: Any) -> Iterable[str]:
    if v is None:
        return
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for x in v:
            if isinstance(x, str):
                yield x


def load_collapsed_agent_graph(graph_path: Path) -> Tuple[Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Collapse bipartite typed edges: M1 -> M2 iff exists data node D with
    M1 -model_to_data-> D and D -data_to_model-> M2 (within the same links_type family).
    """
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    model_ids = {m["id"] for m in data["model_nodes"]}
    succ: Dict[str, Set[str]] = defaultdict(set)
    pred: Dict[str, Set[str]] = defaultdict(set)

    for key in ("links_type1", "links_type2", "links_type3", "links_type4"):
        md: List[Tuple[str, str]] = []  # (agent, data_id)
        dm: List[Tuple[str, str]] = []  # (data_id, agent)
        for e in data.get(key, []):
            et = e.get("type")
            tgt = e.get("target")
            if not isinstance(tgt, str):
                continue
            if et == "model_to_data":
                for s in iter_endpoints(e.get("source")):
                    if s in model_ids and tgt not in model_ids:
                        md.append((s, tgt))
            elif et == "data_to_model":
                for s in iter_endpoints(e.get("source")):
                    if s not in model_ids and tgt in model_ids:
                        dm.append((s, tgt))
        data_to_consumers: Dict[str, Set[str]] = defaultdict(set)
        for dnode, m2 in dm:
            data_to_consumers[dnode].add(m2)
        for m1, dnode in md:
            for m2 in data_to_consumers.get(dnode, set()):
                succ[m1].add(m2)
                pred[m2].add(m1)

    agents = set(model_ids)
    return agents, succ, pred


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def _vec_counter(tokens: List[str]) -> Dict[str, float]:
    c: Dict[str, float] = {}
    for t in tokens:
        c[t] = c.get(t, 0.0) + 1.0
    return c


def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k, va in a.items():
        na += va * va
    for k, vb in b.items():
        nb += vb * vb
    for k, va in a.items():
        if k in b:
            dot += va * b[k]
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def build_agent_desc_vectors(manifest: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    return {name: _vec_counter(_tokenize(json.dumps(spec, ensure_ascii=False))) for name, spec in manifest.items()}


def oracle_random_params_plan(
    gt_plan: List[Dict[str, Any]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    plan = json.loads(json.dumps(gt_plan, ensure_ascii=False))
    for st in plan:
        if not isinstance(st, dict):
            continue
        inputs = st.get("inputs")
        if not isinstance(inputs, dict):
            continue
        new_in = {}
        for k, v in inputs.items():
            if isinstance(v, str) and _GEN_RE.match(v.strip()):
                new_in[k] = v
            else:
                new_in[k] = f"/tmp/rand_{rng.randint(0, 10**9)}_{k}.dat"
        st["inputs"] = new_in
    return plan


def _pick_inputs_for_agent(
    manifest: Dict[str, Any],
    agent: str,
    step_idx: int,
    prev_agent: Optional[str],
    rng: random.Random,
) -> Tuple[Dict[str, Any], List[str], Any, List[int]]:
    spec = manifest.get(agent) or {}
    in_keys = list((spec.get("input") or {}).keys())
    out_keys = list((spec.get("output") or {}).keys())
    inputs: Dict[str, Any] = {}
    if step_idx == 0:
        dep = [-1]
        dc = None
        for k in in_keys:
            inputs[k] = f"/tmp/greedy_rand_{rng.randint(0,10**6)}_{k}.dat"
    else:
        dep = [step_idx - 1]
        prev_spec = manifest.get(prev_agent or "") or {}
        prev_out = list((prev_spec.get("output") or {}).keys())
        if not prev_out:
            prev_out = ["output_path"]
        dc = {str(step_idx - 1): prev_out}
        for k in in_keys:
            if rng.random() < 0.5 and prev_out:
                pk = rng.choice(prev_out)
                inputs[k] = f"<GENERATED>-{step_idx - 1}-<{pk}>"
            else:
                inputs[k] = f"/tmp/greedy_rand_{rng.randint(0,10**6)}_{k}.dat"
    return inputs, out_keys, dc, dep


def dag_greedy_tfidf_plan(
    task_desc: str,
    T: int,
    manifest: Dict[str, Any],
    agents: Set[str],
    succ: Dict[str, Set[str]],
    pred: Dict[str, Set[str]],
    rng: random.Random,
    agent_vecs: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    tv = _vec_counter(_tokenize(task_desc))
    starts = [a for a in agents if len(pred.get(a, set())) == 0]
    if not starts:
        starts = sorted(agents)

    def score(agent: str) -> float:
        return cosine_sparse(tv, agent_vecs.get(agent, {}))

    plan: List[Dict[str, Any]] = []
    prev_chosen: Optional[str] = None
    for i in range(T):
        if i == 0:
            cand = starts
        else:
            cand = sorted(succ.get(prev_chosen or "", set()))
            if not cand:
                cand = sorted(agents)
        best = max(cand, key=lambda a: (score(a), a))
        inputs, out_keys, dc, dep = _pick_inputs_for_agent(manifest, best, i, prev_chosen, rng)
        plan.append(
            {
                "agent": best,
                "step": i,
                "dependence": dep,
                "dependence_content": dc,
                "inputs": inputs,
                "outputs": out_keys,
            }
        )
        prev_chosen = best
    return plan


def dag_beam_tfidf_plan(
    task_desc: str,
    T: int,
    manifest: Dict[str, Any],
    agents: Set[str],
    succ: Dict[str, Set[str]],
    pred: Dict[str, Set[str]],
    rng: random.Random,
    agent_vecs: Dict[str, Dict[str, float]],
    beam_width: int = 3,
    topk_per_step: int = 8,
    global_topk: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Beam search over agent chains with DAG successor constraints.

    - step0 candidates: zero-in-degree agents (or all agents if none)
    - step i>0 candidates: successors of previous chosen agent; fallback to all agents if dead-end
    - scoring: sum of cosine_sparse(task_vec, agent_desc_vec) across steps
    - output plan: same schema as other structural baselines; dependence is linear [i-1]
    """
    if T <= 0:
        return []
    beam_width = max(1, int(beam_width))
    topk_per_step = max(1, int(topk_per_step))

    tv = _vec_counter(_tokenize(task_desc))
    starts = [a for a in agents if len(pred.get(a, set())) == 0]
    if not starts:
        starts = sorted(agents)

    def score(agent: str) -> float:
        return cosine_sparse(tv, agent_vecs.get(agent, {}))

    # Pre-score all agents to cheaply take top-k in candidate sets.
    agent_scores = {a: score(a) for a in agents}
    allowed_agents: Set[str] = set(agents)
    if global_topk is not None:
        k = max(1, int(global_topk))
        top = sorted(agents, key=lambda a: (agent_scores.get(a, 0.0), a), reverse=True)[:k]
        allowed_agents = set(top)
        # Ensure we always keep starts in the candidate set so the planner can begin.
        allowed_agents.update(starts)

    # Beam entries: (total_score, [agent_seq])
    beam: List[Tuple[float, List[str]]] = []

    # init
    init_pool = [a for a in starts if a in allowed_agents]
    if not init_pool:
        init_pool = sorted(allowed_agents)
    init_cand = sorted(init_pool, key=lambda a: (agent_scores.get(a, 0.0), a), reverse=True)[:topk_per_step]
    for a in init_cand[:beam_width]:
        beam.append((agent_scores.get(a, 0.0), [a]))

    # expand
    for _i in range(1, T):
        new_beam: List[Tuple[float, List[str]]] = []
        for s_total, seq in beam:
            prev = seq[-1]
            cand = [a for a in succ.get(prev, set()) if a in allowed_agents]
            if not cand:
                cand = sorted(allowed_agents)
            # take top-k by global similarity (tie-break stable by name)
            cand = sorted(cand, key=lambda a: (agent_scores.get(a, 0.0), a), reverse=True)[:topk_per_step]
            for a in cand:
                new_beam.append((s_total + agent_scores.get(a, 0.0), seq + [a]))

        if not new_beam:
            break
        # keep best beams; add tiny noise via rng for deterministic tie-breaking across equal scores
        new_beam.sort(key=lambda x: (x[0], rng.random()), reverse=True)
        beam = new_beam[:beam_width]

    best_seq = beam[0][1] if beam else [rng.choice(sorted(agents))]

    out: List[Dict[str, Any]] = []
    prev_agent: Optional[str] = None
    for i, ag in enumerate(best_seq):
        inputs, out_keys, dc, dep = _pick_inputs_for_agent(manifest, ag, i, prev_agent, rng)
        prev_agent = ag
        out.append(
            {
                "agent": ag,
                "step": i,
                "dependence": dep,
                "dependence_content": dc,
                "inputs": inputs,
                "outputs": out_keys,
            }
        )
    return out


def shortest_path_agent_sequence(
    first: str, last: str, agents: Set[str], succ: Dict[str, Set[str]]
) -> List[str]:
    if first not in agents or last not in agents:
        return [first, last]
    if first == last:
        return [first]
    q = deque([first])
    parent: Dict[str, Optional[str]] = {first: None}
    while q:
        u = q.popleft()
        if u == last:
            break
        for v in succ.get(u, set()):
            if v not in parent:
                parent[v] = u
                q.append(v)
    if last not in parent:
        return [first, last]
    path = []
    cur: Optional[str] = last
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def extend_path_to_length(
    path: List[str],
    T: int,
    succ: Dict[str, Set[str]],
    agents: Set[str],
    rng: random.Random,
) -> List[str]:
    path = list(path)
    if not path:
        path = [rng.choice(sorted(agents))]
    while len(path) < T:
        nxt_list = sorted(succ.get(path[-1], set()))
        if nxt_list:
            path.append(rng.choice(nxt_list))
        else:
            path.append(rng.choice(sorted(agents)))
    if len(path) > T:
        path = path[:T]
    return path


def shortest_path_plan(
    gt_plan: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    agents: Set[str],
    succ: Dict[str, Set[str]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    T = len(gt_plan)
    if T == 0:
        return []
    first = gt_plan[0].get("agent")
    last = gt_plan[-1].get("agent")
    if not isinstance(first, str) or not isinstance(last, str):
        return []
    p = shortest_path_agent_sequence(first, last, agents, succ)
    p = extend_path_to_length(p, T, succ, agents, rng)
    out: List[Dict[str, Any]] = []
    prev_agent: Optional[str] = None
    for i, ag in enumerate(p):
        inputs, out_keys, dc, dep = _pick_inputs_for_agent(manifest, ag, i, prev_agent, rng)
        prev_agent = ag
        out.append(
            {
                "agent": ag,
                "step": i,
                "dependence": dep,
                "dependence_content": dc,
                "inputs": inputs,
                "outputs": out_keys,
            }
        )
    return out
