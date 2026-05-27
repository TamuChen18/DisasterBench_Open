## Benchmark graph structural complexity

Generated: 2026-04-28 00:01

**Source**: `structured_plan` in each row of `data/benchmark.jsonl` (ground-truth DAG).

### Definitions

- **Linear task**: the graph has **no branching and no merging** — every node has out-degree ≤ 1 and in-degree ≤ 1.
- **Branching task**: some node has out-degree > 1 (one step’s output feeds multiple successors).
- **Merging task**: some node has in-degree > 1 (a step depends on multiple predecessors).
- **Average branching factor**:
  - **overall**: mean out-degree over all nodes (total edges / total nodes).
  - **non-leaf**: mean out-degree only over nodes with out-degree > 0 (closer to “active” branching strength).

> Branching and merging are not mutually exclusive; a task can have both (this benchmark has **0** merging tasks).

### Statistics (233 tasks)

- **Avg steps per task**: 2.528
- **Linear tasks**: 201 / 233 (86.27%)
- **Branching tasks**: 32 / 233 (13.73%)
- **Merging tasks**: 0 / 233 (0.00%)
- **Avg branching factor (overall mean out-degree)**: 0.331
- **Avg branching factor (non-leaf mean out-degree)**: 1.266
- **Max out-degree**: 3

### Degree histograms (node-level, pooled over all tasks)

- **Out-degree**: 0→435, 1→119, 2→29, 3→6
- **In-degree**: 0→394, 1→195

### Reproduce

From the repository root:

```bash
python3 scripts/compute_structure_complexity.py --data data/benchmark.jsonl
```
