"""Combine all 10 conflict datasets x (English + 22 translations) into one dataset.

Union schema: dataset-exclusive columns are NULL for the other datasets;
`language` holds the row's language; `dataset_link` (rightmost column) points
to the per-dataset source repo on the Hub.

Usage:
    python combine.py                 # build + push to <HUB_ORG>/combined
    python combine.py --dry-run       # build only, print stats
"""

import argparse
import logging
from pathlib import Path

from datasets import Dataset, Value, concatenate_datasets, load_from_disk
from huggingface_hub import HfApi

import config
from pipeline import apply_renames, load_source

log = logging.getLogger(__name__)

COMBINED_REPO = f"{config.HUB_ORG}/combined"

CARD = """---
license: cc-by-sa-4.0
task_categories:
- visual-question-answering
- image-text-to-text
language:
- en
- ar
- cs
- de
- el
- es
- fa
- fr
- he
- hi
- id
- it
- ja
- ko
- nl
- pl
- pt
- ro
- ru
- tr
- uk
- vi
- zh
tags:
- vlm
- image-text-conflict
- multilingual
- multimodal
size_categories:
- 10K<n<100K
---

# Multilingual VLM Conflict — Combined

All 10 datasets of the **multilingual-vlm-conflict** suite in one place, in English plus 22
languages translated with [CohereLabs/aya-expanse-32b](https://huggingface.co/CohereLabs/aya-expanse-32b)
(Arabic, Czech, German, Greek, Spanish, Persian, French, Hebrew, Hindi, Indonesian, Italian,
Japanese, Korean, Dutch, Polish, Portuguese, Romanian, Russian, Turkish, Ukrainian, Vietnamese,
Simplified Chinese). 10 datasets x 23 languages x 100 examples = 23,000 rows.

Each example pairs an image with a truthful `original_caption` and a `conflicting_caption`
that alters exactly one object or attribute, creating an image-text conflict. The `question`
asks about precisely the element that changes; `image_bias` is the true value grounded in the
image, `text_bias` the altered value asserted by the conflicting caption, and `distractor` a
plausible third option.

## Columns

The schema is the union across the 10 datasets: columns exclusive to some datasets are
`NULL` for rows coming from the others.

| column | description | present in |
|---|---|---|
| `image` | the image (truthful) | all |
| `serial_no` | source identifier (string) | all |
| `original_caption` | caption faithful to the image | all |
| `conflicting_caption` | caption with one object/attribute changed | all |
| `question` | asks about the changed object/attribute | all |
| `image_bias` | true value, grounded in the image | all |
| `text_bias` | altered value asserted by `conflicting_caption` | all |
| `distractor` | plausible third option, different from both biases | all |
| `open-ended_answer` | free-form answer grounded in the image | physics, radiology, DrivingVQA |
| `conflicting_open-ended_answer` | free-form answer following the conflicting caption | physics, radiology, DrivingVQA |
| `conflict_type` | category of the conflict | all |
| `conflict_description` | free-form description of the conflict | 3D-Object |
| `caption_verification` | verification notes for the caption | radiology |
| `file_name` | source file name | pendulum |
| `original_row_id` | row id in the source dataset | pendulum |
| `original_text_bias` | text bias before patching | pendulum |
| `is_patched` | whether the row was patched | pendulum |
| `language` | language of the row's text fields (lowercase English name) | all |
| `dataset_link` | URL of the per-dataset repo this row comes from | all |

## Source datasets

| dataset | link |
|---|---|
{dataset_rows}

## Licenses

The combination is released under **CC-BY-SA-4.0** (required by the share-alike source
datasets `sea-vl-conflict` and `worldcuisines-conflict`, both CC-BY-SA-4.0;
`coco-counterfactual-conflict` is CC-BY-4.0). See each per-dataset repo for source details.
"""


def dataset_card() -> str:
    rows = "\n".join(
        f"| `{name}` | https://huggingface.co/datasets/{config.HUB_ORG}/{name} |"
        for name in config.DATASETS
    )
    return CARD.replace("{dataset_rows}", rows)

# Union of all columns across the 10 datasets; dataset_link must stay rightmost.
COLUMN_ORDER = [
    "image",
    "serial_no",
    "original_caption",
    "conflicting_caption",
    "question",
    "image_bias",
    "text_bias",
    "distractor",
    "open-ended_answer",
    "conflicting_open-ended_answer",
    "conflict_type",
    "conflict_description",
    "caption_verification",
    "file_name",
    "original_row_id",
    "original_text_bias",
    "is_patched",
    "language",
    "dataset_link",
]


def normalise(ds: Dataset, dataset_name: str) -> Dataset:
    ds = apply_renames(ds)
    # serial_no is int64 in most datasets but string in pendulum-conflict.
    if ds.features["serial_no"].dtype != "string":
        ds = ds.cast_column("serial_no", Value("string"))
    link = f"https://huggingface.co/datasets/{config.HUB_ORG}/{dataset_name}"
    ds = ds.add_column("dataset_link", [link] * len(ds))
    for column in COLUMN_ORDER:
        if column not in ds.column_names:
            ds = ds.add_column(column, [None] * len(ds), feature=Value("string"))
    return ds.select_columns(COLUMN_ORDER)


def build(out_root: Path) -> Dataset:
    parts = []
    for name in config.DATASETS:
        parts.append(normalise(load_source(name), name))
        for code in config.TARGET_LANGUAGES:
            translated = load_from_disk(str(out_root / name / code))
            parts.append(normalise(translated, name))
        log.info("%s: english + %d translations collected", name, len(config.TARGET_LANGUAGES))
    # concatenate_datasets follows the underlying arrow table order, not the
    # features order, so re-apply the canonical order (dataset_link stays last).
    combined = concatenate_datasets(parts).select_columns(COLUMN_ORDER)
    log.info("combined: %d rows, %d columns", len(combined), len(combined.column_names))
    return combined


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--repo", default=COMBINED_REPO)
    parser.add_argument("--dry-run", action="store_true", help="build only, do not push")
    args = parser.parse_args()

    combined = build(args.out)
    languages = sorted(set(combined["language"]))
    log.info("languages (%d): %s", len(languages), languages)
    assert len(combined) == len(config.DATASETS) * (len(config.TARGET_LANGUAGES) + 1) * 100

    if args.dry_run:
        return
    combined.push_to_hub(
        args.repo,
        private=False,
        max_shard_size="500MB",
        commit_message="Add combined dataset: 10 conflict datasets x 23 languages",
    )
    HfApi().upload_file(
        path_or_fileobj=dataset_card().encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    log.info("pushed -> https://huggingface.co/datasets/%s", args.repo)


if __name__ == "__main__":
    main()
