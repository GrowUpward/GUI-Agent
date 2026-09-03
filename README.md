# GUI-Agent-Qwen3.5

A resource-efficient mobile GUI agent built on Qwen3.5-0.8B. The project studies whether staged SFT and GRPO can transfer a small multimodal model from single-step grounding to structured CAGUI action prediction under two 24 GB GPUs.

## Status

This is an independent, cleaned research repository extracted from an earlier UI-R1 experiment workspace. It contains source code and reproducible entry points only; datasets, model weights, caches, and historical checkpoints stay outside Git.

Archived SFT result on a fixed 128-sample CAGUI generation test (before the coordinate-order audit):

| Metric | Value |
| --- | ---: |
| Format/parse success | 88.28% |
| Action-type accuracy | 73.44% |
| Argument accuracy | 9.38% |
| Click hit rate | 16.13% |
| Mean click distance (normalized 0–1000 coordinates) | 275.10 |
| Input-text exact match | 43.75% |
| Mean reward | 0.3317 |

The audit found that the legacy training path preserved CAGUI's `[y, x]` values while labeling them as model-facing `[x, y]`. Action-type and text metrics remain informative, but coordinate-dependent metrics must be re-established with the corrected pipeline before they are used as headline results. The current server can swap outputs from the archived adapter with `LEGACY_YX_OUTPUT=1`.

The GRPO implementation and short-run checkpoints exist, but a controlled SFT-versus-GRPO improvement has not yet been established. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## What is original here

The upstream UI-R1 project focuses on rule-based RL for GUI action prediction with Qwen2.5-VL. This repository adds and maintains the following work:

- Qwen3.5-0.8B LoRA/QLoRA adaptation;
- CAGUI episode parsing and history-aware step prediction;
- an audited CAGUI `[y, x]` to action `[x, y]` coordinate boundary;
- episode-level train/validation/test splitting;
- CAGUI bridge SFT with generation-based per-action metrics;
- structured format/action/argument reward and a compact custom GRPO loop;
- environment diagnostics and configuration-driven launchers;
- optional FastAPI, ADB, and Gradio execution prototypes.

See [`NOTICE`](NOTICE) for upstream attribution.

## Architecture

```text
CAGUI episodes
  -> history-aware step dataset
  -> Qwen3.5-0.8B + QLoRA SFT
  -> deterministic smoke evaluation
  -> grouped rollout + structured reward + KL-regularized GRPO
  -> metrics/checkpoints

Screenshot -> model server -> structured action -> human confirmation -> ADB
```

## Repository layout

```text
.
├── configs/                  # Environment templates; no secrets or machine paths
├── docs/                     # Architecture, experiment status, and attribution
├── scripts/                  # Stable shell entry points
├── src/gui_agent/
│   ├── data/                 # Dataset inspection
│   ├── deployment/           # FastAPI, ADB, and Gradio prototypes
│   ├── training/             # Active SFT and GRPO implementations
│   └── doctor.py             # Environment and path diagnostics
└── tests/                    # Fast parser/reward/dataset tests
```

## Quick start

Python 3.10 and a CUDA-enabled PyTorch environment are recommended.

```bash
cd /data/data1/zhaozhiqing/project/GUI-Agent
python -m pip install -e '.[dev,demo]'
cp configs/local.env.example configs/local.env
# Edit configs/local.env, then:
set -a
. configs/local.env
set +a
gui-agent-doctor --strict
```

On the current server, use the existing interpreter explicitly:

```bash
export PYTHON=/data/data0/zhaozhiqing/anaconda3/envs/python3.10/bin/python
$PYTHON -m pip install -e . --no-deps
```

`--no-deps` preserves the current CUDA/PyTorch stack. Install missing packages deliberately rather than downgrading the environment with the original UI-R1 `setup.sh`.

The current server environment already satisfies the training dependencies, but does not contain `pytest`, `fastapi`, `uvicorn`, or `gradio`. Training and dependency-light smoke tests work as-is. Install the `dev` and/or `demo` extras before using those optional paths, then check the demo stack with `scripts/doctor.sh --strict --demo`.

## Data check and dry-run

```bash
scripts/doctor.sh --strict
scripts/test.sh
scripts/inspect_data.sh --json
scripts/dry_run.sh
```

`scripts/test.sh` always runs dependency-light assertions and also runs the pytest suite when pytest is installed. The dry-run reads a small number of real episodes and validates target conversion and reward computation without loading model weights.

## SFT

```bash
RUN_NAME=sft-cagui-$(date +%Y%m%d-%H%M) \
MAX_STEPS=300 \
LEARNING_RATE=1e-4 \
BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=2 \
MAX_IMAGE_SIDE=448 \
scripts/train_sft.sh
```

To continue an adapter, set `INIT_ADAPTER` and always use a new `RUN_NAME`.

## GRPO

First run deterministic smoke generation:

```bash
SFT_ADAPTER=/path/to/sft-adapter \
RUN_NAME=grpo-smoke-$(date +%Y%m%d-%H%M) \
scripts/train_grpo.sh --smoke_generate 30 --no_do_sample
```

Then run a logged 10-step probe:

```bash
SFT_ADAPTER=/path/to/sft-adapter \
RUN_NAME=grpo-probe-$(date +%Y%m%d-%H%M) \
MAX_STEPS=10 \
SAVE_STEPS=10 \
LOGGING_STEPS=1 \
NUM_GENERATIONS=4 \
scripts/train_grpo.sh
```

Compare the resulting model against the SFT adapter on the same split and decoding configuration before increasing the training budget.

## Mobile prototype

The server accepts a base model and an optional PEFT adapter:

```bash
MODEL_PATH="$BASE_MODEL" \
ADAPTER_PATH="$SFT_ADAPTER" \
scripts/serve_model.sh
```

Set `LEGACY_YX_OUTPUT=1` only for adapters trained with the pre-audit coordinate convention. Leave it disabled for all new training runs.

Run the ADB controller on the machine that can reach the Android device or emulator. Execution remains human-confirmed by default.

## Known limitations

- The current reward is based on labeled step actions, not verified post-execution environment state.
- CAGUI is click-heavy, so overall action accuracy can hide minority-action failures.
- The current best model still has weak coordinate/argument accuracy.
- The archived adapter was trained with a legacy coordinate-order mismatch; the server compatibility flag does not replace corrected retraining and evaluation.
- GRPO benefit has not yet been demonstrated in a controlled experiment.
- The mobile path is a prototype and must be validated with the Qwen3.5 adapter before claiming end-to-end task success.

## Citation and license

This repository is Apache-2.0 licensed and retains attribution to UI-R1. If you use the upstream method, cite the official AAAI 2026 paper listed in [`NOTICE`](NOTICE).
