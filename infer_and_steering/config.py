"""Single source of truth: combined dataset, model registry, steering defaults."""

from dataclasses import dataclass
from pathlib import Path

HUB_ORG = "multilingual-vlm-conflict"
COMBINED_REPO = f"{HUB_ORG}/combined"

ROOT = Path(__file__).parent
VECTORS_DIR = ROOT / "vectors"    # gitignored: steering vectors (torch .pt)
RESULTS_DIR = ROOT / "results"    # gitignored: inference outputs (jsonl)

# Language code -> value of the combined dataset's `language` column.
LANGUAGES = {
    "en": "english",
    "ar": "arabic",
    "cs": "czech",
    "de": "german",
    "el": "greek",
    "es": "spanish",
    "fa": "persian",
    "fr": "french",
    "he": "hebrew",
    "hi": "hindi",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "tr": "turkish",
    "uk": "ukrainian",
    "vi": "vietnamese",
    "zh": "simplified chinese",
}

DATASETS = [
    "code-conflict",
    "3D-Object-Conflict",
    "physics-conflict",
    "rpg-conflict",
    "sea-vl-conflict",
    "coco-counterfactual-conflict",
    "worldcuisines-conflict",
    "pendulum-conflict",
    "radiology-conflict",
    "DrivingVQA-conflict",
]


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    loader: str = "auto"  # auto | seq2seq | minicpm | custom
    trust_remote_code: bool = False
    note: str = ""


# All repo ids verified to exist on the Hub (2026-07-07).
MODELS: dict[str, ModelSpec] = {
    # Qwen VL family
    "qwen2.5-vl-7b": ModelSpec("Qwen/Qwen2.5-VL-7B-Instruct"),
    "qwen3-vl-8b": ModelSpec("Qwen/Qwen3-VL-8B-Instruct"),
    # Multilingual LLaVA-NeXT
    "pangea-7b": ModelSpec("neulab/Pangea-7B-hf"),
    # BLIP-2 + mT0 (encoder-decoder language model)
    "mblip-mt0-xl": ModelSpec("Gregor/mblip-mt0-xl", loader="seq2seq"),
    "granite-vision-3.3-2b": ModelSpec("ibm-granite/granite-vision-3.3-2b"),
    "jina-vlm": ModelSpec("jinaai/jina-vlm", trust_remote_code=True),
    # SEA-LION v4 vision models (also: Gemma-SEA-LION-v4-4B-VL / -27B-VL)
    "sea-lion-v4-8b-vl": ModelSpec("aisingapore/Qwen-SEA-LION-v4-8B-VL"),
    "palo-7b": ModelSpec(
        "MBZUAI/PALO-7B",
        loader="custom",
        note="LLaVA fork, not transformers-native; needs https://github.com/mbzuai-oryx/PALO",
    ),
    "maya-8b": ModelSpec(
        "maya-multimodal/maya",
        loader="custom",
        note="LLaVA + Aya-23, not transformers-native; needs https://github.com/nahidalam/maya",
    ),
    "aya-vision-8b": ModelSpec("CohereLabs/aya-vision-8b"),
    "internvl3-8b": ModelSpec("OpenGVLab/InternVL3-8B-hf"),
    # MiniCPM-V exposes a custom .chat() API via remote code
    "minicpm-v-4.5": ModelSpec("openbmb/MiniCPM-V-4_5", loader="minicpm", trust_remote_code=True),
}

# Steer the middle of the decoder stack by default; mid layers carry the most
# linearly-decodable behavioural directions in practice.
DEFAULT_LAYER_FRACTIONS = (0.4, 0.5, 0.6)
DEFAULT_ALPHA = 6.0
