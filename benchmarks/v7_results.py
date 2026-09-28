"""Pure CSV validation and aggregation shared by tuning/report tools."""
import csv
import json
import math
import statistics
from pathlib import Path

def variation(times):
    return (max(times) - min(times)) / statistics.median(times)

def read_runs(paths):
    if len(paths) < 3:
        raise ValueError("at least three runs are required")
    runs, manifests, signatures = [], [], set()
    for path in map(Path, paths):
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        env = manifest["environment"]
        mapping = {}
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row["status"] != "pass":
                    raise ValueError(f"correctness failure: {path}")
                key = (*[int(row[k]) for k in ("M", "N", "K")], row["provider"])
                value = float(row["median_us"])
                if key in mapping or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"duplicate/invalid result: {key}")
                signatures.add(tuple(row[k] for k in (
                    "gpu", "sm", "cuda", "torch", "layout", "input_dtype",
                    "accum_dtype", "output_dtype", "warmup", "samples",
                    "launches_per_sample", "driver", "git_sha")) +
                    (env["code_fingerprint"], env["seed"], env["shapes_sha256"]))
                mapping[key] = row
        if not mapping:
            raise ValueError(f"empty CSV: {path}")
        runs.append(mapping)
        manifests.append(manifest)
    if len(signatures) != 1 or any(set(run) != set(runs[0]) for run in runs[1:]):
        raise ValueError("mixed environment/protocol/source/shape coverage")
    return runs, manifests

def times(runs, key):
    return [float(run[key]["median_us"]) for run in runs]
