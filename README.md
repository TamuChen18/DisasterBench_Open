# DisasterBench

A benchmark for evaluating LLM structured plan generation over typed multi-agent workflows in disaster-management scenarios.

DisasterBench contains **233 planning tasks**, **26 task-interface agents**, and a compatibility graph defining valid inter-agent dependencies.

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=.
export OPENROUTER_API_KEY=...   # or OPENAI_API_KEY

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

`--task_num 0` runs all tasks; `--task_num N` runs a single task (smoke test).

Outputs: `results/<model>/<method>/` with `*_result.txt` and `*_detailed_predictions.json`.

### Other methods

| Method | Flag |
|--------|------|
| Direct Planning | `--test_type dp` |
| Tree-of-Thoughts | `--test_type tot` |
| RAP (MCTS) | `--test_type rap` |
| ReAct | `--test_type react` |

### Structural baselines (no LLM)

```bash
PYTHONPATH=. python3 scripts/run_structural_baselines.py --baseline shortest_path
```

Also: `oracle_random`, `dag_greedy_tfidf`, `dag_beam_tfidf`.

## Repository layout

```
DisasterBench/
├── data/benchmark.jsonl              # 233 tasks + gold plans
├── interfaces/tools/
│   ├── tools_manifest.json           # 26 agent schemas
│   └── graph_desc.json               # allowed agent transitions
├── evaluators/evaluators.py          # scoring + FPoF
├── baselines/                        # CoT, ToT, RAP, DP, ReAct + prompts
├── config/args.py
├── scripts/
│   ├── run_structural_baselines.py
│   └── compute_structure_complexity.py
└── docs/RESULTS_STRUCTURE_COMPLEXITY.md
```

## Dataset

Each JSONL row:

- `task_id` — unique id  
- `task_desc` — natural-language task  
- `structured_plan` — executable DAG plan with agents, parameters, and dependencies

Ground-truth plans are DAGs. Structural breakdown: [docs/RESULTS_STRUCTURE_COMPLEXITY.md](docs/RESULTS_STRUCTURE_COMPLEXITY.md).

Reproduce stats:

```bash
python3 scripts/compute_structure_complexity.py --data data/benchmark.jsonl
```

## Metrics

| Metric | Meaning |
|--------|---------|
| **Overall** | Exact structured-plan match over agents, parameters, and dependencies |
| **Tools** | Correct (step, agent) pairs |
| **Parameters** | Correct per-step inputs |
| **Dependencies** | Correct dependence structure |
| **FPoF** | Earliest failure category in the generated plan |

## Agents

Agents are represented as typed tool/interface schemas rather than executable model checkpoints:

- `interfaces/tools/tools_manifest.json`

## Supported LLM backends

- **OpenAI** — `--api openai --model_ckpt gpt-4o-mini`  
- **OpenRouter** — `--api openrouter --model_ckpt deepseek/deepseek-v3.2`  

Current release supports OpenAI- and OpenRouter-compatible APIs.

## Citation

```bibtex
@misc{disasterbench2026,
  title={DisasterBench: Benchmarking LLM Planning under Typed Tool Interface Constraints},
  year={2026}
}
```

## License

Released under the MIT License.
