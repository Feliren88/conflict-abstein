"""Push the 22 translations of every dataset to its per-dataset Hub repo.

Each language becomes a named config (its ISO code) with a `train` split, e.g.
load_dataset("multilingual-vlm-conflict/code-conflict", "id"); the default
config (the English originals) is left untouched.

Usage:
    python push_translations.py                                  # everything missing
    python push_translations.py --datasets code-conflict --languages ar
"""

import argparse
import logging
import time
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import DatasetCard

import config

log = logging.getLogger(__name__)


def existing_configs(repo: str) -> set[str]:
    data = DatasetCard.load(repo).data.to_dict()
    return {entry["config_name"] for entry in data.get("configs", [])}


def push(ds, repo: str, code: str, language: str, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        try:
            ds.push_to_hub(repo, config_name=code, commit_message=f"Add {language} translation ({code} config)")
            return
        except Exception as error:
            log.warning("%s config %s failed (attempt %d/%d): %s", repo, code, attempt, attempts, error)
            if attempt == attempts:
                raise
            time.sleep(30 * attempt)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS, choices=config.DATASETS)
    parser.add_argument("--languages", nargs="+", default=list(config.TARGET_LANGUAGES), choices=list(config.TARGET_LANGUAGES))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()

    for name in args.datasets:
        repo = f"{config.HUB_ORG}/{name}"
        done = existing_configs(repo)
        for code in args.languages:
            if code in done:
                log.info("%s/%s already on the Hub, skipping", name, code)
                continue
            ds = load_from_disk(str(args.out / name / code))
            push(ds, repo, code, config.TARGET_LANGUAGES[code])
            log.info("%s/%s pushed", name, code)
    log.info("done")


if __name__ == "__main__":
    main()
