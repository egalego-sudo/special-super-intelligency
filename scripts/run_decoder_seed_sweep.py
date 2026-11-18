"""Run and persist a reproducible multi-seed decoder research experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from special_super_intelligency.experiment import run_decoder_seed_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[131, 137, 139])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_decoder_seed_sweep(
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        output_path=args.output,
    )
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
