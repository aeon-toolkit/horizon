"""Simulated correctness of aeon's Theta, DOTM and AutoCES against other toolkits.

Forecasts the twelve horizon processes (six ARIMA, six ETS) at lengths 100 and
500, 100 series each, with the seeds of the horizon accuracy benchmark, and
writes ``accuracy.csv`` in horizon's format. ``family`` is the forecaster
(theta, dotm, autoces) and ``toolkit`` the implementation, so horizon's
analysis compares aeon with statsforecast (and smooth for AutoCES) per
forecaster: paired Wilcoxon signed-rank tests (``significance.csv``) and TOST
equivalence (``equivalence.csv``). Every forecast goes through the M4 script's
``fit_forecaster``, so aeon's Theta is deseasonalised M4-style, as on M4.

    PYTHONPATH=<aeon worktree>:src python scripts/sim_other_stats.py \\
        --out results/aeon-8585c7bf/sim-other --n-jobs 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"
os.environ["M_COMP_DESEASONALISE"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

ENTRIES = {  # family -> [(toolkit, backend, forecaster)]
    "theta": [("aeon", "aeon", "Theta"), ("statsforecast", "statsforecast", "Theta")],
    "dotm": [("aeon", "aeon", "DOTM"), ("statsforecast", "statsforecast", "DOTM")],
    "autoces": [("aeon", "aeon", "AutoCES"), ("statsforecast", "statsforecast", "AutoCES"),
                ("smooth", "smooth", "AutoCES")],
}
LENGTHS = [100, 500]


def _one(task):
    """Fit every entry on one simulated series; return accuracy rows."""
    import m_comp_experiments as mc
    from horizon.experiments import score

    pname, sp, n, h, r, seed, y = task
    train, test = y[:n], y[n:]
    rows = []
    for family, entries in ENTRIES.items():
        for toolkit, backend, name in entries:
            start = time.perf_counter()
            res = mc.fit_forecaster(backend, name, train, h, sp)
            elapsed = time.perf_counter() - start
            ok = res.status == "ok" and res.forecast is not None
            sc = (score(train, test, np.asarray(res.forecast, float), sp) if ok
                  else dict.fromkeys(["mae", "rmse", "mase", "smape"]))
            rows.append(dict(mode="auto", family=family, process=pname, sp=sp, n=n,
                             h=h, replicate=r, seed=seed, toolkit=toolkit,
                             time_s=elapsed, **sc, selected_model=res.model,
                             error=None if ok else str(res.status)))
    return rows


def main() -> None:
    from horizon.experiments import _seed, horizon_for
    from horizon.simulate import get_processes

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    procs = get_processes()
    names = [p.name for p in procs]
    tasks = []
    for proc in procs:
        sp, pidx = proc.seasonal_period, names.index(proc.name)
        h = horizon_for(sp)
        for n in LENGTHS:
            for r in range(args.replicates):
                seed = _seed(pidx, n, r, 0)
                y = np.asarray(proc.sample(n + h, random_state=seed), dtype=np.float64)
                tasks.append((proc.name, sp, n, h, r, seed, y))
    rows = []
    with ProcessPoolExecutor(args.n_jobs) as pool:
        for i, out in enumerate(pool.map(_one, tasks, chunksize=4)):
            rows.extend(out)
            if (i + 1) % 400 == 0:
                print(f"{i + 1}/{len(tasks)} series", flush=True)
    acc = pd.DataFrame(rows)
    acc.to_csv(args.out / "accuracy.csv", index=False)

    import m_comp_experiments as mc
    from horizon.analyse import accuracy_summary, significance
    from horizon.sweep import equivalence

    significance(acc).to_csv(args.out / "significance.csv", index=False)
    equivalence(acc).to_csv(args.out / "equivalence.csv", index=False)
    accuracy_summary(acc).to_csv(args.out / "accuracy_summary.csv", index=False)
    (args.out / "run_info.json").write_text(json.dumps(
        {"command": " ".join(sys.argv), "aeon": mc._aeon_info(),
         "lengths": LENGTHS, "replicates": args.replicates,
         "entries": ENTRIES}, indent=2, default=str))
    print(acc.groupby(["family", "toolkit"]).agg(
        mase=("mase", "median"), errors=("error", lambda e: e.notna().sum())))


if __name__ == "__main__":
    main()
