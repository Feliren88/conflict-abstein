"""Score run_inference.py output: which bias did the model follow?

Usage:
    python evaluate.py results/qwen2.5-vl-7b.jsonl [more.jsonl ...] [--by language]
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

OUTCOMES = ("image_bias", "text_bias", "distractor", "abstain", "unparsed")


def choice(response: str, options: dict[str, str]) -> str:
    """First standalone option letter in the response, mapped to its bias key."""
    match = re.search(rf"\b([{''.join(options)}])\b", response.upper())
    return options[match.group(1)] if match else "unparsed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--by", choices=("dataset", "language", "model"), default="dataset")
    args = parser.parse_args()

    tallies: dict[str, Counter] = defaultdict(Counter)
    for path in args.results:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            tallies[row[args.by]][choice(row["response"], row["options"])] += 1

    width = max(map(len, tallies), default=8)
    print(f"{args.by:<{width}}  " + "".join(f"{o:>12}" for o in OUTCOMES) + f"{'n':>8}")
    for group in sorted(tallies):
        counts, total = tallies[group], sum(tallies[group].values())
        rates = "".join(f"{counts[o] / total:>11.1%} " for o in OUTCOMES)
        print(f"{group:<{width}}  {rates}{total:>8}")


if __name__ == "__main__":
    main()
