"""Extract "trust the image" steering vectors from faithful/conflicting pairs.

Usage:
    python extract_vectors.py --model qwen2.5-vl-7b                    # english, 100 pairs
    python extract_vectors.py --model qwen2.5-vl-7b --languages en id --limit 200
"""

import argparse
import logging

import config
import data
import steering
from models import VLM

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(config.MODELS))
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS, choices=config.DATASETS)
    parser.add_argument("--languages", nargs="+", default=["en"], choices=list(config.LANGUAGES))
    parser.add_argument("--limit", type=int, default=100, help="number of contrast pairs")
    parser.add_argument("--out", default=None, help="default: vectors/<model>.pt")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    out = config.VECTORS_DIR / f"{args.model}.pt" if args.out is None else config.ROOT / args.out

    vlm = VLM(args.model)
    rows = data.load_combined(args.datasets, args.languages).shuffle(seed=42)
    rows = rows.select(range(min(args.limit, len(rows))))

    faithful_states, conflicting_states = [], []
    for i, row in enumerate(rows):
        faithful, conflicting = data.contrast_prompts(row)
        faithful_states.append(steering.capture(vlm, row["image"], faithful))
        conflicting_states.append(steering.capture(vlm, row["image"], conflicting))
        if (i + 1) % 10 == 0:
            log.info("captured %d/%d pairs", i + 1, len(rows))

    vectors = steering.mean_difference(faithful_states, conflicting_states)
    steering.save_vectors(out, vectors, {
        "model": args.model,
        "datasets": args.datasets,
        "languages": args.languages,
        "n_pairs": len(rows),
        "direction": "faithful - conflicting (positive alpha steers toward the image)",
    })


if __name__ == "__main__":
    main()
