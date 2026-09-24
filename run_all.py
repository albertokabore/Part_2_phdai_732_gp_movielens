"""Run the whole Part 2 pipeline, in order, and record provenance.

    python run_all.py           # full run; tuning takes several minutes
    python run_all.py --quick   # shrunken grids to prove the wiring; never cite

Outputs: results/ (JSON and CSV, including KEY_NUMBERS.md), figures/ (PNG),
and results/run_manifest.json recording library versions, the seed, the data
fingerprint, and how long each step took.
"""

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config as C
from src.data import load_data
from src.eda import run_eda
from src.evaluation import run_final_evaluation
from src.fairness import run_fairness
from src.key_numbers import build as build_key_numbers
from src.mitigation import run_mitigation
from src.models import run_roster, surprise
from src.splits import describe_split, make_split, train_test
from src.temporal import run_temporal
from src.tuning import run_tuning


def main(quick=False):
    import matplotlib

    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "quick_smoke_test": quick,
        "seed": C.SEED,
        "python": platform.python_version(),
        "surprise": getattr(surprise, "__version__", "unknown"),
        "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "steps": {},
    }
    path = C.RESULTS_DIR / "run_manifest.json"

    def step(name, fn, *a, **k):
        print(f"[{name}]", flush=True)
        t0 = time.perf_counter()
        out = fn(*a, **k)
        manifest["steps"][name] = round(time.perf_counter() - t0, 1)
        path.write_text(json.dumps(manifest, indent=2, default=str))
        return out

    ratings, users, items, log = step("load_data", load_data)
    manifest["ratings_sha256"] = log["ratings_sha256"]

    splits = {s: make_split(ratings, s) for s in C.SPLIT_STRATEGIES}
    manifest["split_sha256"] = {s: hashlib.sha256(
        (C.RESULTS_DIR / f"split_{s}.csv").read_bytes()).hexdigest() for s in splits}
    train, test = train_test(ratings, splits[C.HEADLINE_SPLIT])
    (C.RESULTS_DIR / "split_summary.json").write_text(json.dumps(
        [describe_split(ratings, a, s) for s, a in splits.items()], indent=2, default=float))

    step("eda", run_eda, ratings, users, items)
    roster_table, roster_frames, roster_fitted = step("roster", run_roster, train, test)
    tuning = step("tuning", run_tuning, train, quick=quick)
    tuned_params = tuning["selected"]["params"]
    _, tuned, frame, top_n = step("final_evaluation", run_final_evaluation, train, test, items,
                                  tuned_params, roster_table, roster_frames)
    step("fairness", run_fairness, frame, users, train, top_n, items)
    step("mitigation", run_mitigation, tuned, frame, roster_fitted["svd"], roster_frames["svd"], train)
    step("temporal", run_temporal, ratings, tuned_params)
    step("key_numbers", build_key_numbers, quick=quick)

    manifest["status"] = "complete"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Done. Read results/KEY_NUMBERS.md. Total {sum(manifest['steps'].values()):.0f} s.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="shrunken grids; never cite the output")
    main(quick=p.parse_args().quick)
