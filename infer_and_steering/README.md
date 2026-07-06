# Inference & Activation Steering

Evaluate multilingual VLMs on [`multilingual-vlm-conflict/combined`](https://huggingface.co/datasets/multilingual-vlm-conflict/combined) (10 datasets × 23 languages × 100 image-text-conflict rows) and steer them away from text bias with mean-difference activation steering.

## Method

Each row pairs an image with a caption that contradicts it in exactly one attribute, plus a question about that attribute with three candidate answers: `image_bias` (true, grounded in the image), `text_bias` (asserted by the conflicting caption), and `distractor`.

1. **Baseline inference** (`run_inference.py`) — MCQ prompting with the conflicting caption in context; `--abstain` adds a fourth "cannot be answered" option, `--no-caption` runs the caption-free control.
2. **Vector extraction** (`extract_vectors.py`) — the same row is prompted twice, with the *faithful* caption and with the *conflicting* caption; each decoder layer's hidden state at the last prompt token is recorded, and `vector[layer] = mean(faithful) − mean(conflicting)`. Positive alpha therefore steers a conflicted input toward the image-consistent activation regime.
3. **Steered inference** (`run_inference.py --vectors ...`) — forward hooks add `alpha · unit(vector)` to the residual stream at the selected layers (default: 40/50/60% of the stack) during generation.
4. **Scoring** (`evaluate.py`) — parses the chosen letter and reports image-bias / text-bias / distractor / abstain / unparsed rates per dataset, language, or model.

## Models

| registry key | HF repo | status |
|---|---|---|
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | native |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | native |
| `pangea-7b` | `neulab/Pangea-7B-hf` | native (LLaVA-NeXT) |
| `mblip-mt0-xl` | `Gregor/mblip-mt0-xl` | native (BLIP-2 seq2seq) |
| `granite-vision-3.3-2b` | `ibm-granite/granite-vision-3.3-2b` | native |
| `jina-vlm` | `jinaai/jina-vlm` | remote code |
| `sea-lion-v4-8b-vl` | `aisingapore/Qwen-SEA-LION-v4-8B-VL` | native |
| `palo-7b` | `MBZUAI/PALO-7B` | needs [PALO repo](https://github.com/mbzuai-oryx/PALO) |
| `maya-8b` | `maya-multimodal/maya` | needs [Maya repo](https://github.com/nahidalam/maya) |
| `aya-vision-8b` | `CohereLabs/aya-vision-8b` | native |
| `internvl3-8b` | `OpenGVLab/InternVL3-8B-hf` | native |
| `minicpm-v-4.5` | `openbmb/MiniCPM-V-4_5` | remote code (`.chat()` API) |

PALO and Maya are LLaVA forks without transformers-native weights: the loader raises `NotImplementedError` with a pointer until their upstream code is vendored in.

## Usage

```bash
pip install -r requirements.txt

# 1. baseline
python run_inference.py --model qwen2.5-vl-7b --languages en id --limit 200

# 2. extract steering vectors (100 faithful/conflicting pairs, english)
python extract_vectors.py --model qwen2.5-vl-7b

# 3. steered run, sweep alpha
python run_inference.py --model qwen2.5-vl-7b --vectors vectors/qwen2.5-vl-7b.pt --alpha 8 --languages en id --limit 200

# 4. compare
python evaluate.py results/qwen2.5-vl-7b.jsonl results/qwen2.5-vl-7b.steered.jsonl --by model
```

`vectors/` and `results/` are gitignored.

## Files

| file | purpose |
|---|---|
| `config.py` | combined-repo id, language map, model registry, steering defaults |
| `data.py` | dataset loading/filtering, MCQ prompt builder, contrast pairs |
| `models.py` | unified `VLM` wrapper: load, generate, locate decoder layers |
| `steering.py` | activation capture, mean-difference vectors, steering hooks |
| `extract_vectors.py` | CLI: build steering vectors per model |
| `run_inference.py` | CLI: baseline / steered / abstain / control runs |
| `evaluate.py` | CLI: bias-rate tables from result jsonl files |
