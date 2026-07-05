"""Load, normalise, translate, and persist one dataset at a time.

Output layout (resumable: finished dataset/language pairs are skipped):
    <out_root>/<dataset>/<lang_code>/   # datasets.load_from_disk-compatible
"""

import logging
from pathlib import Path

from datasets import Dataset, load_dataset

import config
from translator import AyaTranslator

log = logging.getLogger(__name__)


def load_source(name: str) -> Dataset:
    ds = load_dataset(f"{config.HUB_ORG}/{name}", split="train")
    for old, new in config.COLUMN_RENAMES.items():
        if old in ds.column_names:
            ds = ds.rename_column(old, new)
    return ds


def text_columns(ds: Dataset) -> list[str]:
    return [
        name
        for name, feature in ds.features.items()
        if getattr(feature, "dtype", None) == "string" and name not in config.SKIP_COLUMNS
    ]


def translate_dataset(
    name: str,
    translator: AyaTranslator,
    targets: dict[str, str],
    out_root: Path,
    limit: int | None = None,
) -> None:
    ds = load_source(name)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    columns = text_columns(ds)
    # Column-wise access skips image decoding; one batch = one row's text fields.
    batches = [list(row) for row in zip(*(ds[column] or [""] for column in columns))]
    batches = [["" if text is None else text for text in row] for row in batches]
    log.info("%s: %d rows, translating columns %s", name, len(ds), columns)

    for code, language in targets.items():
        out_dir = out_root / name / code
        if (out_dir / "dataset_info.json").exists():
            log.info("%s/%s already done, skipping", name, code)
            continue
        translated = translator.translate_batches(batches, language)
        result = ds.map(
            lambda _, i: dict(zip(columns, translated[i]), language=language.lower()),
            with_indices=True,
            desc=f"{name}/{code}",
        )
        result.save_to_disk(str(out_dir))
        log.info("%s/%s saved to %s", name, code, out_dir)
