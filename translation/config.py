"""Single source of truth: model, datasets, languages, column policy."""

MODEL_ID = "CohereLabs/aya-expanse-32b"
HUB_ORG = "multilingual-vlm-conflict"

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

# All languages Aya Expanse supports, minus English (the source).
TARGET_LANGUAGES = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Simplified Chinese",
}

# Some datasets shipped with a space in the column name; normalise everywhere.
COLUMN_RENAMES = {"conflict type": "conflict_type"}

# String columns that are metadata, not natural language: never translated.
SKIP_COLUMNS = {
    "language",
    "serial_no",
    "conflict_type",
    "file_name",
    "original_row_id",
    "is_patched",
}
