# Conflict Abstein

Multilingual VLM conflict datasets and translation pipeline.

## Overview

Two components:

1. **Conflict dataset design** (`prompt.md`) — methodology for constructing three new VLM conflict-evaluation datasets (SEA-VL, COCO-Counterfactuals, WorldCuisines), each with 100 hand-crafted/curated rows following the schema of the existing `rpg-conflict` dataset on HuggingFace. Each row pairs an image with an original caption, a minimally altered conflicting caption, and a question targeting the swapped attribute.

2. **Translation pipeline** (`translation/`) — batch translation of all 10 datasets under the [`multilingual-vlm-conflict`](https://huggingface.co/multilingual-vlm-conflict) HuggingFace organisation into 22 non-English languages using [CohereLabs/aya-expanse-32b](https://huggingface.co/CohereLabs/aya-expanse-32b) served via [vLLM](https://github.com/vllm-project/vllm).

## Translation pipeline

### Datasets translated

| Dataset | Rows | Source |
|---|---|---|
| `code-conflict` | 100 | HuggingFace |
| `3D-Object-Conflict` | 100 | HuggingFace |
| `physics-conflict` | 100 | HuggingFace |
| `rpg-conflict` | 100 | HuggingFace |
| `sea-vl-conflict` | 100 | HuggingFace |
| `coco-counterfactual-conflict` | 100 | HuggingFace |
| `worldcuisines-conflict` | 100 | HuggingFace |
| `pendulum-conflict` | 100 | HuggingFace |
| `radiology-conflict` | 100 | HuggingFace |
| `DrivingVQA-conflict` | 100 | HuggingFace |

### Target languages

Arabic, Chinese (Simplified), Czech, Dutch, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian, Vietnamese.

### Usage

```bash
# Translate everything (10 datasets × 22 languages)
python main.py

# Smoke test: 1 dataset, 1 language, 3 rows
python main.py --smoke

# Translate specific datasets and languages with a row limit
python main.py --datasets code-conflict rpg-conflict --languages id vi --limit 10
```

The pipeline is resumable: already-finished dataset/language pairs are skipped on re-run. Output is saved to `translation/output/<dataset>/<lang_code>/` as HuggingFace `datasets`-compatible disk format.

### Requirements

- Python 3.10+
- CUDA-capable GPU with ~60 GB VRAM (for Aya Expanse 32B in FP16)
- See `translation/requirements.txt`

## Output structure

```
translation/output/
├── <dataset>/
│   ├── <lang_code>/           # saved with datasets.save_to_disk()
│   │   ├── dataset_info.json
│   │   └── data-*.arrow
│   └── ...
└── ...
```

## Files

| File | Purpose |
|---|---|
| `prompt.md` | Conflict dataset design specification |
| `translation_prompt.md` | Translation task prompt used to generate the pipeline |
| `translation/config.py` | Model ID, dataset list, target languages, column policy |
| `translation/pipeline.py` | Load → normalise → translate → persist per dataset |
| `translation/translator.py` | Aya Expanse client with batched prompting and fallback retry |
| `translation/main.py` | CLI entry point with smoke-test mode |
