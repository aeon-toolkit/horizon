"""Rolling-origin one-step forecasts of aeon's forecasters on the aaltd long series.

Each aaltd problem is one series stored as ``<name>_TRAIN.csv`` and
``<name>_TEST.csv`` (a column headed ``0``); the two are joined into the full
series ``y`` of length ``N`` (up to 10,000). The forecaster is refitted at each
of the last ``--origins`` (30) time points: fit to ``y[:N - 30 + i]`` and
forecast ``y[N - 30 + i]`` one step ahead, ``i = 0..29``. Every fit goes
through the M4 script's ``fit_forecaster`` (``--deseasonalise`` as on M4), so
the forecasters are configured exactly as in the M4 runs.

Seasonal periods follow the Monash archive's conventions from the sampling
frequency (half-hourly 48, hourly 24, daily 7, 10-minute 144, minutely 1440,
weekly 52, monthly 12). ``Naive`` is the last value; ``SeasonalNaive`` the value
one period back.

Output: one CSV per forecaster, ``<out>/<forecaster>.csv``, a row per
(problem, origin); reruns skip rows already written, so a stopped run resumes.

    PYTHONPATH=<aeon worktree> python scripts/aaltd_experiments.py \\
        --forecasters Naive SeasonalNaive --n-jobs 32 \\
        --out /local/ajb2u23/Results/Forecasting/aaltd-8585c7bf
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

DATA = Path("/local/ajb2u23/Data/Forecasting/aaltd")
ORIGINS = 30
# problem-name prefix -> seasonal period (Monash conventions by sampling frequency)
PERIODS = {
    "australian_electricity_demand": 48, "elecdemand": 48,
    "london_smart_meters": 48, "electricity_hourly": 24, "kdd_cup_2018": 24,
    "oikolab_weather": 24, "pedestrian_counts": 24, "traffic_hourly": 24,
    "m4_hourly": 24, "saugeenday": 7, "sunspot": 7, "us_births": 7,
    "weather": 7, "m4_daily": 7, "m4_weekly": 52, "m4_monthly": 12,
    "solar_10_minutes": 144, "wind_farms_minutely": 1440,
}
# name used here -> (backend, M4-script forecaster, use the seasonal period)
ALIASES = {"Naive": ("aeon", "NaiveForecaster", False)}
TASK_TIMEOUT = 3 * 3600  # seconds per fit; SIGALRM, so pure-Python waits only


def period_of(problem: str) -> int:
    """Seasonal period of an aaltd problem from its name."""
    for prefix, m in PERIODS.items():
        if problem.startswith(prefix):
            return m
    raise ValueError(f"no seasonal period for {problem!r}")


def load(problem: str) -> np.ndarray:
    """The full series: TRAIN followed by TEST."""
    parts = [pd.read_csv(DATA / problem / f"{problem}_{s}.csv").iloc[:, 0]
             for s in ("TRAIN", "TEST")]
    return np.concatenate(parts).astype(np.float64)


def problems() -> list[str]:
    return sorted(p.name for p in DATA.iterdir() if p.is_dir())


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout


def _one(task):
    """One fit and one-step forecast; returns a result row."""
    import m_comp_experiments as mc

    forecaster, problem, origin, n_origins = task
    y = load(problem)
    m = period_of(problem)
    cut = y.size - n_origins + origin
    backend, name, seasonal = ALIASES.get(forecaster, ("aeon", forecaster, True))
    period = m if seasonal else 1
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TASK_TIMEOUT)
    start = time.perf_counter()
    try:
        res = mc.fit_forecaster(backend, name, y[:cut], 1, period)
        status, pred = res.status, (
            float(np.ravel(res.forecast)[0]) if res.forecast is not None else np.nan)
        model = res.model
        deseason = (res.params or {}).get("deseasonalised")
        error = getattr(res, "error", None)
    except _Timeout:
        status, pred, model, deseason = f"timeout {TASK_TIMEOUT}s", np.nan, None, None
        error = None
    finally:
        signal.alarm(0)
    return dict(forecaster=forecaster, problem=problem, origin=origin, n_train=cut,
                period=period, actual=float(y[cut]), pred=pred, status=status,
                fit_seconds=time.perf_counter() - start, model=model,
                deseasonalised=deseason, error=str(error)[:300] if error else None)


def _append(path: Path, rows: list[dict]) -> None:
    """Append rows to a results CSV, keeping the columns of an existing file."""
    df = pd.DataFrame(rows)
    if path.exists():
        cols = pd.read_csv(path, nrows=0).columns
        df = df.reindex(columns=cols)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--forecasters", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--origins", type=int, default=ORIGINS)
    parser.add_argument("--problems", nargs="*", help="default: all aaltd problems")
    parser.add_argument("--shard", default="0/1",
                        help="i/k: run every k-th problem from the i-th (for GPUs)")
    parser.add_argument("--deseasonalise", action="store_true",
                        help="M4-style deseasonalisation for the non-seasonal and "
                             "ML/DL forecasters, as in the M4 runs")
    args = parser.parse_args()
    if args.deseasonalise:
        os.environ["M_COMP_DESEASONALISE"] = "1"
    args.out.mkdir(parents=True, exist_ok=True)
    i, k = map(int, args.shard.split("/"))
    probs = (args.problems or problems())[i::k]
    # longest series first, so the slowest tasks do not straggle at the end
    probs.sort(key=lambda p: -load(p).size)

    import m_comp_experiments as mc

    aeon = mc._aeon_info() if hasattr(mc, "_aeon_info") else None
    print("aeon", json.dumps(aeon, default=str), flush=True)
    for forecaster in args.forecasters:
        path = args.out / f"{forecaster}.csv"
        done = set()
        if path.exists():
            prev = pd.read_csv(path)
            prev = prev[prev.status == "ok"]  # failed fits are retried on a rerun
            done = set(zip(prev.problem, prev.origin))
        tasks = [(forecaster, p, o, args.origins) for p in probs
                 for o in range(args.origins) if (p, o) not in done]
        print(f"=== {forecaster}: {len(tasks)} fits to run ({len(done)} done) "
              f"{time.strftime('%c')}", flush=True)
        if not tasks:
            continue
        # spawn: TensorFlow and CUDA are not fork-safe. TensorFlow workers leak
        # memory per fit, so they run in rounds of 10 fits per worker, each round
        # in a fresh pool (max_tasks_per_child hung on Python 3.11: exited
        # workers were not replaced).
        tf_model = forecaster in mc.TENSORFLOW_FORECASTERS
        size = args.n_jobs * 10 if tf_model else len(tasks)
        j = 0
        for start in range(0, len(tasks), size):
            with ProcessPoolExecutor(args.n_jobs,
                                     mp_context=mp.get_context("spawn")) as pool:
                batch = []
                futures = [pool.submit(_one, t) for t in tasks[start : start + size]]
                for fut in as_completed(futures):
                    batch.append(fut.result())
                    j += 1
                    if len(batch) >= 30:
                        _append(path, batch)
                        batch = []
                    if j % 300 == 0:
                        print(f"  {forecaster} {j}/{len(tasks)} {time.strftime('%c')}",
                              flush=True)
                if batch:
                    _append(path, batch)
        df = pd.read_csv(path)
        bad = df[df.status != "ok"]
        print(f"=== {forecaster} done: {len(df)} rows, {len(bad)} not ok "
              f"{time.strftime('%c')}", flush=True)
        info = args.out / "run_info.json"
        meta = json.loads(info.read_text()) if info.exists() else {}
        meta[forecaster] = {"command": " ".join(sys.argv), "aeon": aeon,
                            "origins": args.origins, "deseasonalise": args.deseasonalise,
                            "finished": time.strftime("%c")}
        info.write_text(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
