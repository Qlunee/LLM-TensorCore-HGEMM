#!/usr/bin/env python3
"""Select runtime shapes, retaining phase/module provenance."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import bench_square

def select_shapes(path, limit):
    queues = {"prefill": [], "decode": []}
    seen = {phase: set() for phase in queues}
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            phase = row["phase"]
            if phase not in queues:
                continue
            key = tuple(int(row[k]) for k in ("M", "N", "K"))
            if min(key) <= 0:
                raise ValueError("nonpositive runtime shape")
            if key not in seen[phase]:
                seen[phase].add(key)
                queues[phase].append(dict(m=key[0], n=key[1], k=key[2],
                                          phase=phase, module=row["module"]))
    chosen, unique = [], set()
    while len(chosen) < limit and any(queues.values()):
        for phase in queues:
            if not queues[phase]:
                continue
            item = queues[phase].pop(0)
            key = (item["m"], item["n"], item["k"])
            if key not in unique:
                unique.add(key)
                chosen.append(item)
            if len(chosen) == limit:
                break
    if not chosen:
        raise ValueError("no runtime shapes")
    return chosen

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    args, remaining = p.parse_known_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    if any(x == "--shapes" or x.startswith("--shapes=") for x in remaining):
        raise ValueError("shapes are generated from runtime CSV")
    shapes = select_shapes(args.csv, args.limit)
    target = args.output.with_suffix(".shapes.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"shapes": shapes}, indent=2))
    for item in shapes:
        print(item)
    previous = sys.argv
    try:
        sys.argv = ["bench_square.py", "--shapes", str(target),
                    "--output", str(args.output), *remaining]
        bench_square.main()
    finally:
        sys.argv = previous

if __name__ == "__main__":
    main()
