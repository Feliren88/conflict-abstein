"""Run (optionally steered) MCQ inference on the combined conflict dataset.

Usage:
    python run_inference.py --model qwen2.5-vl-7b --languages en id --limit 50   # baseline
    python run_inference.py --model qwen2.5-vl-7b --vectors vectors/qwen2.5-vl-7b.pt --alpha 8
    python run_inference.py --model qwen2.5-vl-7b --abstain                      # 4th abstain option

Each output line records the option map, so evaluate.py can score which bias
(image / text / distractor / abstain) the model followed.
"""

import argparse
import contextlib
import json
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
    parser.add_argument("--languages", nargs="+", default=list(config.LANGUAGES), choices=list(config.LANGUAGES))
    parser.add_argument("--limit", type=int, default=None, help="cap on total rows")
    parser.add_argument("--abstain", action="store_true", help="add an explicit abstain option")
    parser.add_argument("--no-caption", action="store_true", help="control run without the conflicting caption")
    parser.add_argument("--vectors", default=None, help="steering vectors .pt from extract_vectors.py")
    parser.add_argument("--alpha", type=float, default=config.DEFAULT_ALPHA)
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="layer ids to steer (default: mid stack)")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--out", default=None, help="default: results/<model>[.steered].jsonl")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    suffix = ".steered" if args.vectors else ""
    out = config.RESULTS_DIR / f"{args.model}{suffix}.jsonl" if args.out is None else config.ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    vlm = VLM(args.model)
    rows = data.load_combined(args.datasets, args.languages)
    if args.limit:
        rows = rows.shuffle(seed=42).select(range(min(args.limit, len(rows))))

    steer = contextlib.nullcontext()
    if args.vectors:
        vectors, meta = steering.load_vectors(config.ROOT / args.vectors)
        layer_ids = args.layers or steering.default_layer_ids(len(vlm.decoder_layers()))
        log.info("vectors from %s (%s)", args.vectors, meta)
        steer = steering.Steer(vlm, vectors, layer_ids, args.alpha)

    with steer, out.open("w") as sink:
        for i, row in enumerate(rows):
            caption = None if args.no_caption else row["conflicting_caption"]
            prompt, option_map = data.mcq(row, caption=caption, abstain=args.abstain)
            response = vlm.generate(row["image"], prompt, max_new_tokens=args.max_new_tokens)
            sink.write(json.dumps({
                "dataset": row["dataset_link"].rsplit("/", 1)[-1],
                "language": row["language"],
                "serial_no": row["serial_no"],
                "options": option_map,
                "response": response,
                "model": args.model,
                "alpha": args.alpha if args.vectors else 0.0,
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                log.info("%d/%d rows", i + 1, len(rows))
    log.info("wrote %d rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
