"""Run time against series length: aeon's Theta, DOTM and AutoCES vs others.

Times one fit-and-forecast per series, single threaded, through the same code
path as the M4 runs (``m_comp_experiments.fit_forecaster``; aeon's Theta is
deseasonalised there as on M4). Series come from two horizon processes,
ETS(A,A,N) (non-seasonal) and ETS(A,A,A) m=12 (seasonal), with the seeds of
the horizon timing experiments. Each (backend, forecaster) pays its JIT
compilation in a warm-up before timing.

    PYTHONPATH=<aeon worktree> python scripts/timing_other_stats.py \\
        --out results/aeon-8585c7bf/timing-other --repeats 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"
os.environ["M_COMP_DESEASONALISE"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import m_comp_experiments as mc  # noqa: E402

from horizon.experiments import _seed, horizon_for  # noqa: E402
from horizon.simulate import get_processes  # noqa: E402

ENTRIES = [  # (backend, forecaster)
    ("aeon", "Theta"), ("statsforecast", "Theta"),
    ("aeon", "DOTM"), ("statsforecast", "DOTM"),
    ("aeon", "AutoCES"), ("statsforecast", "AutoCES"), ("smooth", "AutoCES"),
]
PROCESSES = ["ets_aan", "ets_aaa_m12"]
LENGTHS = [100, 200, 500, 1000, 2000, 5000, 10000]
OFFSET = 10_000_000
TIME_BUDGET = 30.0  # skip longer lengths once an entry's median exceeds this


def _version(package: str) -> dict:
    """Version and install path of a package, to tell statsforecast builds apart."""
    import importlib

    module = importlib.import_module(package)
    return {"version": getattr(module, "__version__", None), "path": module.__file__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--all-processes", action="store_true",
                        help="time on all 12 horizon processes, not just two ETS ones")
    parser.add_argument("--entries", nargs="+", metavar="BACKEND:FORECASTER",
                        help="time only these (e.g. statsforecast:Theta); default all")
    args = parser.parse_args()
    entries = ([tuple(e.split(":", 1)) for e in args.entries] if args.entries
               else ENTRIES)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "timing.csv"
    done = set()
    if path.exists():
        prev = pd.read_csv(path)
        done = set(zip(prev.process, prev.n, prev.repeat, prev.backend, prev.forecaster))

    rng = np.random.default_rng(0)
    for backend, name in entries:  # warm-up: compile on short series
        for period in (1, 12):
            y = 100 + np.cumsum(rng.normal(size=120)) + (
                5 * np.sin(np.arange(120) * 2 * np.pi / period) if period > 1 else 0)
            mc.fit_forecaster(backend, name, np.abs(y) + 1, 12, period)

    all_names = [p.name for p in get_processes()]
    for proc in (get_processes() if args.all_processes
                 else get_processes("ets", PROCESSES)):
        sp = proc.seasonal_period
        h = horizon_for(sp)
        pidx = all_names.index(proc.name)
        over = set()
        for n in sorted(args.lengths):
            rows = []
            for r in range(args.repeats):
                y = np.asarray(proc.sample(n, random_state=_seed(pidx, n, r, OFFSET)),
                               dtype=np.float64)
                for backend, name in entries:
                    if (backend, name) in over or (proc.name, n, r, backend, name) in done:
                        continue
                    start = time.perf_counter()
                    res = mc.fit_forecaster(backend, name, y, h, sp)
                    elapsed = time.perf_counter() - start
                    rows.append(dict(process=proc.name, sp=sp, n=n, h=h, repeat=r,
                                     backend=backend, forecaster=name,
                                     time_s=elapsed, status=res.status,
                                     model=res.model))
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(path, mode="a", header=not path.exists(), index=False)
                med = df.groupby(["backend", "forecaster"]).time_s.median()
                over |= {k for k, v in med.items() if v > TIME_BUDGET}
                print(f"[timing] {proc.name} n={n} " + json.dumps(
                    {f"{b}:{f}": round(v, 4) for (b, f), v in med.items()}), flush=True)
    t = pd.read_csv(path)
    summary = (t[t.status == "ok"].groupby(["process", "n", "backend", "forecaster"])
               .time_s.median().rename("median_s").reset_index())
    summary.to_csv(args.out / "timing_summary.csv", index=False)
    (args.out / "run_info.json").write_text(json.dumps(
        {"command": " ".join(sys.argv),
         "statsforecast": _version("statsforecast"), "aeon": mc._aeon_info() if hasattr(mc, "_aeon_info") else None,
         "repeats": args.repeats, "lengths": args.lengths}, indent=2, default=str))


if __name__ == "__main__":
    main()
