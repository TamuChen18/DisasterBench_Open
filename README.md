**# DisasterBench

DisasterBench is a benchmark for evaluating **executable workflow grounding** in disaster-response multi-agent systems. It tests whether LLMs can generate structured, executable plans over semantically similar but operationally distinct disaster-response agents.

DisasterBench focuses on a realistic orchestration setting: given a natural-language disaster-management request, an LLM must compose typed agents into a valid multi-step workflow with correct agent selection, parameter binding, and dependency propagation.

<p align="center">
  <img src="assets/DisasterBench_pipeline.png" width="95%">
</p>

## Overview

DisasterBench contains:

- **233 expert-verified planning tasks**
- **26 task-interface agents**
- **13 functional categories**
- **81 typed compatibility edges**
- **4 semantic workflow subgraphs**
- **5 planning paradigms**: DP, CoT, ToT, RAP, and ReAct

The benchmark evaluates more than tool retrieval. Since agents are constrained by typed input/output interfaces, models must generate workflows that remain executable across the entire planning trajectory.

DisasterBench supports fine-grained evaluation through:

- Exact structured-plan match
- Tool selection accuracy
- Parameter grounding accuracy
- Dependency correctness
- First-Point-of-Failure (FPoF) diagnosis

---

# Benchmark Pipeline

DisasterBench follows a four-stage benchmark construction and evaluation pipeline.

## 1. Problem Framing

We begin with realistic disaster-management scenarios such as:

- Flood response
- Wildfire monitoring
- Hurricane impact assessment
- Urban damage analysis

Users provide high-level natural-language objectives rather than explicit workflows.

Example:

> “Estimate flood inundation depth in socially vulnerable areas under heavy rainfall conditions.”

The planner must determine:

- Which agents should be used
- In what order they should be composed
- Whether intermediate outputs remain type-compatible

---

## 2. Agent Pool & Compatibility Constraints

DisasterBench contains 26 specialized disaster-response agents spanning multiple functional domains:

- Remote sensing
- Change detection
- Image reconstruction
- Adverse-weather perception
- Hydrological forecasting
- Mobility and traffic analysis
- Damage assessment
- Multimodal event understanding

Agents are represented as typed interface schemas rather than executable checkpoints.

Each agent defines:

- Input modalities
- Output modalities
- Parameter specifications
- Dependency requirements

Valid compositions are encoded through a typed compatibility graph.

Example of a valid composition:

```text
Change Detection
    → Damage Assessment
        → Building Extraction
```

Example of an invalid composition:

```text
Rainfall Nowcasting
    → Change Detection
```

because:

```text
Rainfall Nowcasting outputs: json_array
Change Detection requires: raster image
```

All workflows in DisasterBench must satisfy interface-level compatibility constraints.

---

## 3. Task Generation & Verification

Tasks are generated through typed path sampling over the compatibility graph.

Three workflow structures are included:

- Node
- Chain
- DAG / branching workflows

Each sampled workflow is converted into a natural-language planning task and paired with a ground-truth structured plan.

All tasks are manually verified for:

- Dependency correctness
- Parameter grounding
- Task validity
- Workflow executability

The final benchmark contains:

- **233 verified planning tasks**
- Multi-step workflows with depths ranging from 1–9
- Structured DAG plans with typed dependencies

---

## 4. Planning & Evaluation

LLMs generate structured plans in JSON format under five planning paradigms:

| Method | Description |
|---|---|
| DP | Direct Planning |
| CoT | Chain-of-Thought |
| ToT | Tree-of-Thought |
| RAP | Monte-Carlo Tree Search planning |
| ReAct | Interleaved reasoning and acting |

Generated workflows are evaluated using fine-grained metrics.

| Metric | Meaning |
|---|---|
| **Overall** | Exact structured-plan match |
| **Tools** | Correct tool selection |
| **Parameters** | Correct parameter grounding |
| **Dependencies** | Correct dependency structure |
| **FPoF** | Earliest workflow failure category |

FPoF (First-Point-of-Failure) localizes the earliest structural error in a generated workflow, enabling diagnosis of:

- Tool selection failures
- Parameter grounding failures
- Dependency propagation failures
- Early planning termination

---

# Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
export PYTHONPATH=.
export OPENROUTER_API_KEY=...   # or OPENAI_API_KEY
```

Run Chain-of-Thought planning:

```bash
python3 baselines/test_baseline.py \
  --test_type cot \
  --api openrouter \
  --model_ckpt deepseek/deepseek-v3.2 \
  --data_root data \
  --train_test_json_data benchmark.jsonl \
  --task_num 0 \
  --max_threads 12 \
  --max_tokens 8192 \
  --num_chain_of_thought 16 \
  --use_fewshot True
```

- `--task_num 0` runs the full benchmark
- `--task_num N` runs a single task

Outputs are saved to:

```text
results/<model>/<method>/
```

including:

```text
*_result.txt
*_detailed_predictions.json
```

---

# Supported Planning Methods

| Method | Flag |
|---|---|
| Direct Planning | `--test_type dp` |
| Chain-of-Thought | `--test_type cot` |
| Tree-of-Thought | `--test_type tot` |
| RAP (MCTS) | `--test_type rap` |
| ReAct | `--test_type react` |

---

# Structural Baselines (No LLM)

Run graph-based structural baselines:

```bash
PYTHONPATH=. python3 scripts/run_structural_baselines.py \
  --baseline shortest_path
```

Additional supported baselines:

- `oracle_random`
- `dag_greedy_tfidf`
- `dag_beam_tfidf`

---

# Repository Structure

```text
DisasterBench/
├── assets/
│   └── DisasterBench_pipeline.png
├── data/
│   └── benchmark.jsonl
├── interfaces/tools/
│   ├── tools_manifest.json
│   └── graph_desc.json
├── evaluators/
│   └── evaluators.py
├── baselines/
├── scripts/
│   ├── run_structural_baselines.py
│   └── compute_structure_complexity.py
├── docs/
│   └── RESULTS_STRUCTURE_COMPLEXITY.md
└── config/
    └── args.py
```

---

# Dataset Format

Each JSONL row contains:

```json
{
  "task_id": "...",
  "task_desc": "...",
  "structured_plan": {...}
}
```

where:

- `task_id` — unique task identifier
- `task_desc` — natural-language planning request
- `structured_plan` — executable DAG workflow with agents, parameters, and dependencies

Reproduce structural statistics:

```bash
python3 scripts/compute_structure_complexity.py \
  --data data/benchmark.jsonl
```

---

# Agents

Agents are represented as typed interface schemas rather than executable checkpoints.

Agent definitions:

```text
interfaces/tools/tools_manifest.json
```

Compatibility constraints:

```text
interfaces/tools/graph_desc.json
```

---

# Supported APIs

## OpenAI

```bash
--api openai --model_ckpt gpt-4o-mini
```

## OpenRouter

```bash
--api openrouter --model_ckpt deepseek/deepseek-v3.2
```

Current release supports OpenAI-compatible and OpenRouter-compatible APIs.

---

# Citation

```bibtex
@misc{disasterbench2026,
  title={DisasterBench: Benchmarking LLM Planning under Typed Tool Interface Constraints},
  year={2026}
}
```

---

# License

Released under the MIT License.**
