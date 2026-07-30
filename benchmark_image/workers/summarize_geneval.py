#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    buckets = defaultdict(list)
    rows = 0
    with Path(args.results).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            buckets[str(row["tag"])].append(float(row["correct"]))
            rows += 1
    if not buckets:
        raise RuntimeError("GenEval produced no scored rows.")
    if not args.allow_partial and rows != 2212:
        raise RuntimeError(f"GenEval official output must contain 2212 image rows, got {rows}.")
    per_task = {name: sum(values) / len(values) for name, values in sorted(buckets.items())}
    summary = {
        "overall": sum(per_task.values()) / len(per_task),
        "per_task": per_task,
        "num_images": rows,
        "complete": rows == 2212,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
