"""Translate every multilingual-vlm-conflict dataset with Aya Expanse 32B.

Usage:
    python main.py                          # everything: 10 datasets x 22 languages
    python main.py --smoke                  # 1 dataset, 1 language, 3 rows
    python main.py --datasets code-conflict --languages id vi --limit 10

Re-running skips dataset/language pairs already saved under --out.
"""

import argparse
import logging
from pathlib import Path

import config
from pipeline import translate_dataset
from translator import AyaTranslator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS, choices=config.DATASETS)
    parser.add_argument("--languages", nargs="+", default=list(config.TARGET_LANGUAGES), choices=list(config.TARGET_LANGUAGES))
    parser.add_argument("--limit", type=int, default=None, help="translate only the first N rows")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--model", default=config.MODEL_ID)
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end sanity run")
    args = parser.parse_args()
    if args.smoke:
        args.datasets, args.languages, args.limit = args.datasets[:1], args.languages[:1], 3
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    targets = {code: config.TARGET_LANGUAGES[code] for code in args.languages}
    translator = AyaTranslator(args.model)
    for name in args.datasets:
        translate_dataset(name, translator, targets, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
