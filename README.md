# Conflict Abstein

Multilingual image-text-conflict evaluation for vision-language models: dataset construction, translation into 22 languages, and activation steering to reduce text bias.

Everything ships to the [`multilingual-vlm-conflict`](https://huggingface.co/multilingual-vlm-conflict) HuggingFace organisation; the full evaluation corpus lives in [`multilingual-vlm-conflict/combined`](https://huggingface.co/datasets/multilingual-vlm-conflict/combined) (10 datasets × 23 languages × 100 rows = 23,000 examples).

## The task

Every row pairs an image with a truthful `original_caption` and a `conflicting_caption` that alters exactly one object or attribute, creating an image-text conflict. A `question` targets precisely the altered element, with three candidate answers: `image_bias` (true, grounded in the image), `text_bias` (asserted by the conflicting caption), and `distractor` (plausible third option). A model that follows the caption over its own eyes exhibits text bias; the end goal is measuring — and steering away — that failure mode, including the option to abstain.

## Components

### 1. Dataset design (`prompt.md`)

Methodology for constructing three new conflict datasets (SEA-VL, COCO-Counterfactuals, WorldCuisines), each with 100 curated rows following the schema of the existing `rpg-conflict` dataset. Build scripts live in `build/` (gitignored: large downloads and intermediates).

### 2. Translation pipeline (`translation/`)

Batch-translates all 10 datasets of the org into 22 non-English languages with [CohereLabs/aya-expanse-32b](https://huggingface.co/CohereLabs/aya-expanse-32b) served by [vLLM](https://github.com/vllm-project/vllm), then combines everything into the `combined` repo.

**Datasets** (100 rows each): `code-conflict`, `3D-Object-Conflict`, `physics-conflict`, `rpg-conflict`, `sea-vl-conflict`, `coco-counterfactual-conflict`, `worldcuisines-conflict`, `pendulum-conflict`, `radiology-conflict`, `DrivingVQA-conflict`.

**Languages**: Arabic, Czech, German, Greek, Spanish, Persian, French, Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Romanian, Russian, Turkish, Ukrainian, Vietnamese, Simplified Chinese (+ the English source).

```bash
cd translation
python main.py                    # translate everything (resumable; skips finished pairs)
python main.py --smoke            # 1 dataset, 1 language, 3 rows
python main.py --datasets code-conflict --languages id vi --limit 10

python combine.py --dry-run       # build the union-schema combined dataset, print stats
python combine.py                 # build + push to multilingual-vlm-conflict/combined
```

`combine.py` unions the per-dataset schemas (dataset-exclusive columns are NULL for other datasets), adds `language` and `dataset_link` columns, verifies 23,000 rows, and pushes data + dataset card.

| file | purpose |
|---|---|
| `config.py` | model id, dataset list, target languages, column policy |
| `pipeline.py` | load → normalise → translate → persist per dataset |
| `translator.py` | Aya Expanse client: batched JSON-array prompting, per-text fallback retry |
| `main.py` | translation CLI with smoke-test mode |
| `combine.py` | union-schema combine + push of all datasets × languages |

Requires a CUDA GPU with ~60 GB VRAM for Aya Expanse 32B in BF16 (`translation/requirements.txt`). Outputs land in `translation/output/<dataset>/<lang_code>/` (gitignored) as `datasets.save_to_disk` format.

### 3. Inference & activation steering (`infer_and_steering/`)

Evaluates 12 multilingual VLM families (Qwen2.5-VL / Qwen3-VL, Pangea, mBLIP, Granite Vision, Jina VLM, SEA-LION VL, PALO, Maya, Aya Vision, InternVL3, MiniCPM-V) on the combined dataset and applies mean-difference activation steering: contrast faithful-caption vs conflicting-caption activations per decoder layer, then inject the "trust the image" direction into the residual stream during generation.

```bash
cd infer_and_steering
python run_inference.py --model qwen2.5-vl-7b --languages en id --limit 200   # baseline
python extract_vectors.py --model qwen2.5-vl-7b                              # steering vectors
python run_inference.py --model qwen2.5-vl-7b --vectors vectors/qwen2.5-vl-7b.pt --alpha 8
python evaluate.py results/*.jsonl --by language                              # bias-rate tables
```

See [`infer_and_steering/README.md`](infer_and_steering/README.md) for the model registry, method details, and per-model caveats.

## Repository layout

```
├── prompt.md                 # conflict dataset design specification
├── translation_prompt.md     # prompt used to generate the translation pipeline
├── translation/              # translate 10 datasets × 22 languages + combine/push
├── infer_and_steering/       # VLM inference, steering vectors, evaluation
├── build/                    # (gitignored) dataset build downloads & scripts
└── .venv/                    # (gitignored)
```

Large artifacts never enter git: dataset outputs, logs, steering vectors, and results are all gitignored — the datasets live on the HF Hub instead.
