"""Combined-dataset access and prompt construction for conflict evaluation.

Every row of multilingual-vlm-conflict/combined pairs an image with a caption
that contradicts it in exactly one attribute. `mcq` renders a row as a
multiple-choice prompt; `contrast_prompts` renders the same row twice (faithful
caption vs conflicting caption) for steering-vector extraction.
"""

import random

from datasets import Dataset, load_dataset

import config

OPTION_KEYS = ("image_bias", "text_bias", "distractor")
ABSTAIN_KEY = "abstain"
ABSTAIN_TEXT = "The caption contradicts the image, so this cannot be answered."


def load_combined(datasets: list[str] | None = None, languages: list[str] | None = None) -> Dataset:
    ds = load_dataset(config.COMBINED_REPO, split="train")
    if datasets:
        links = {f"https://huggingface.co/datasets/{config.HUB_ORG}/{name}" for name in datasets}
        ds = ds.filter(lambda link: link in links, input_columns="dataset_link")
    if languages:
        names = {config.LANGUAGES[code] for code in languages}
        ds = ds.filter(lambda lang: lang in names, input_columns="language")
    return ds


def mcq(row: dict, caption: str | None, abstain: bool = False, seed: int = 0) -> tuple[str, dict[str, str]]:
    """Render one row as an MCQ prompt; returns (prompt, letter -> option key).

    Option order is shuffled deterministically per row so the answer letter
    carries no signal; `caption` is usually row["conflicting_caption"] (the
    conflict condition) or row["original_caption"] / None (controls).
    """
    options = [(key, row[key]) for key in OPTION_KEYS]
    random.Random(f"{row['serial_no']}/{row['language']}/{seed}").shuffle(options)
    if abstain:
        options.append((ABSTAIN_KEY, ABSTAIN_TEXT))
    letters = "ABCD"[: len(options)]
    lines = [f"{letter}. {text}" for letter, (_, text) in zip(letters, options)]
    caption_line = f"Caption: {caption}\n" if caption else ""
    prompt = (
        f"{caption_line}Question: {row['question']}\n"
        + "\n".join(lines)
        + "\nAnswer with the letter of exactly one option."
    )
    return prompt, {letter: key for letter, (key, _) in zip(letters, options)}


def contrast_prompts(row: dict) -> tuple[str, str]:
    """(faithful-caption prompt, conflicting-caption prompt), identical otherwise.

    The steering vector is mean(faithful) - mean(conflicting): adding it during
    conflicted inference pushes activations toward the image-consistent regime.
    """
    faithful, _ = mcq(row, caption=row["original_caption"])
    conflicting, _ = mcq(row, caption=row["conflicting_caption"])
    return faithful, conflicting
