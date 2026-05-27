# orena-focus-vqa

Personal participation repo for the [ORena SAVE FOCUS Challenge](https://orena-focus-challenge.org) — FRAME, SEGMENT, and PROCEDURE tracks.

**Goal:** Exposure, learning, and honest benchmarking across all 3 tracks.

## Tracks

| Track | Input | Time Budget | VRAM |
|---|---|---|---|
| FRAME | Single image | 5s/question | 48GB |
| SEGMENT | ≤5min video clip | 15s/question | 48GB |
| PROCEDURE | Full procedure video | 30s/question | 80GB |

## Repo Structure

```
orena-focus-vqa/
├── configs/          # Per-track YAML configs (model, prompts, inference params)
├── notebooks/        # Exploration, prototyping, error analysis
├── src/focus_vqa/    # Core reusable package
│   ├── data/         # Dataset loaders wrapping orena-focus
│   ├── models/       # VLM wrappers (Qwen, LLaVA, etc.)
│   ├── prompts/      # Prompt templates per capability group
│   ├── evaluation/   # Wraps orena Evaluator, adds custom logging
│   └── utils/        # Shared helpers
├── scripts/          # CLI: inference, evaluation, submission prep
├── submission/       # Docker container for challenge submission
├── tests/            # Unit tests
└── results/          # Local evaluation outputs (gitignored except .gitkeep)
```

## Setup

```bash
pip install orena-focus
pip install -r requirements.txt
```

## Data (Kaggle)

On Kaggle: Add Data → HuggingFace → `orena-dkfz/heico-focus-vqa`
No download or re-upload needed. The `orena-focus` package resolves it automatically.

## Quickstart

```python
from focus import FocusDataset, DatasetSplit, Track
ds = FocusDataset("heico", DatasetSplit.TRAIN, Track.FRAME)
request, reference = ds[0]
print(request.question)
```

## Progress Log

- [ ] Webinar attended (2026-05-28)
- [ ] First inference running (FRAME)
- [ ] Beat FRAME baseline
- [ ] SEGMENT submission
- [ ] PROCEDURE attempt
