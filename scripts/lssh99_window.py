"""Statistical forecasters on LSSH-99 fitted to a window of recent history.

Do ARIMA and ETS forecast long series better from the last ``w`` observations
than from the whole series, and can ``w`` be chosen automatically? Same
protocol as ``aaltd_experiments.py`` (one-step forecasts, refitted at each
origin, M4-style deseasonalisation with ``--deseasonalise``), but over the last
``--origins`` (60) points: the first 30 origins are a validation period for
choosing the window, the last 30 are the test origins of the paper's LSSH-99
results. At each origin ``t`` the forecaster is fitted to ``y[t - w:t]`` for
each window ``w`` in ``--windows`` (0: the whole series ``y[:t]``). Windows at
least as long as the available history are skipped, since they equal the whole
series; ``lssh99_window_score.py`` maps them to it.

Output: ``<out>/<forecaster>.csv``, one row per (problem, origin, window);
reruns skip rows already written and retry failed fits.

    PYTHONPATH=<aeon worktree> python scripts/lssh99_window.py --deseasonalise \\
        --forecasters AutoARIMA AutoETS ARIMA ETS --n-jobs 32 \\
        --out /local/ajb2u23/Results/Forecasting/lssh99-window-8585c7bf
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
            "NUMBA_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from aaltd_experiments import load, period_of, problems  # noqa: E402

ORIGINS = 60
WINDOWS = [0, 100, 200, 500, 1000, 2000, 5000]
TASK_TIMEOUT = 3600  # seconds per fit


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout


def _one(task):
    """One fit to the window and a one-step forecast; returns a result row."""
    import m_comp_experiments as mc

    forecaster, problem, origin, n_origins, window = task
    y = load(problem)
    m = period_of(problem)
    cut = y.size - n_origins + origin
    train = y[:cut] if window == 0 else y[cut - window:cut]
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TASK_TIMEOUT)
    start = time.perf_counter()
    try:
        res = mc.fit_forecaster("aeon", forecaster, train, 1, m)
        status = res.status
        pred = float(np.ravel(res.forecast)[0]) if res.forecast is not None else np.nan
        model, error = res.model, getattr(res, "error", None)
        deseason = (res.params or {}).get("deseasonalised")
    except _Timeout:
        status, pred, model, deseason, error = f"timeout {TASK_TIMEOUT}s", np.nan, None, None, None
    finally:
        signal.alarm(0)
    return dict(forecaster=forecaster, problem=problem, origin=origin, window=window,
                n_train=train.size, period=m, actual=float(y[cut]),
                naive=float(y[cut - 1]), pred=pred, status=status,
                fit_seconds=time.perf_counter() - start, model=model,
                deseasonalised=deseason, error=str(error)[:300] if error else None)


def _warm_up(forecasters: list[str], deseasonalise: bool) -> None:
    """Compile each forecaster's numba code in this worker before any timed fit."""
    if deseasonalise:
        os.environ["M_COMP_DESEASONALISE"] = "1"
    import m_comp_experiments as mc

    rng = np.random.default_rng(0)
    for period in (1, 12):
        y = 100 + np.cumsum(rng.normal(size=120)) + (
            5 * np.sin(np.arange(120) * 2 * np.pi / period) if period > 1 else 0)
        for f in forecasters:
            mc.fit_forecaster("aeon", f, y, 1, period)


def tasks_for(forecaster: str, probs: list[str], n_origins: int, windows: list[int],
              done: set) -> list[tuple]:
    out = []
    for p in probs:
        n = load(p).size
        for o in range(n_origins):
            cut = n - n_origins + o
            for w in windows:
                if w and w >= cut:
                    continue  # the whole series
                if (p, o, w) not in done:
                    out.append((forecaster, p, o, n_origins, w))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--forecasters", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--origins", type=int, default=ORIGINS)
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOWS,
                        help="training window lengths; 0 is the whole series")
    parser.add_argument("--problems", nargs="*", help="default: all LSSH-99 problems")
    parser.add_argument("--deseasonalise", action="store_true",
                        help="M4-style deseasonalisation for the non-seasonal forecasters")
    args = parser.parse_args()
    if args.deseasonalise:
        os.environ["M_COMP_DESEASONALISE"] = "1"
    args.out.mkdir(parents=True, exist_ok=True)
    probs = sorted(args.problems or problems(), key=lambda p: -load(p).size)

    import m_comp_experiments as mc

    aeon = mc._aeon_info() if hasattr(mc, "_aeon_info") else None
    print("aeon", json.dumps(aeon, default=str), flush=True)
    for forecaster in args.forecasters:
        path = args.out / f"{forecaster}.csv"
        done = set()
        if path.exists():
            prev = pd.read_csv(path)
            prev = prev[prev.status == "ok"]
            done = set(zip(prev.problem, prev.origin, prev.window))
        tasks = tasks_for(forecaster, probs, args.origins, args.windows, done)
        # longest training sets first, so slow fits do not straggle at the end
        tasks.sort(key=lambda t: -(t[4] or 10**6))
        print(f"=== {forecaster}: {len(tasks)} fits to run ({len(done)} done) "
              f"{time.strftime('%c')}", flush=True)
        if not tasks:
            continue
        with ProcessPoolExecutor(args.n_jobs, mp_context=mp.get_context("spawn"),
                                 initializer=_warm_up,
                                 initargs=([forecaster], args.deseasonalise)) as pool:
            batch, j = [], 0
            for fut in as_completed([pool.submit(_one, t) for t in tasks]):
                batch.append(fut.result())
                j += 1
                if len(batch) >= 200:
                    pd.DataFrame(batch).to_csv(path, mode="a", header=not path.exists(),
                                               index=False)
                    batch = []
                if j % 5000 == 0:
                    print(f"  {forecaster} {j}/{len(tasks)} {time.strftime('%c')}", flush=True)
            if batch:
                pd.DataFrame(batch).to_csv(path, mode="a", header=not path.exists(), index=False)
        df = pd.read_csv(path)
        print(f"=== {forecaster} done: {len(df)} rows, {(df.status != 'ok').sum()} not ok "
              f"{time.strftime('%c')}", flush=True)
        info = args.out / "run_info.json"
        meta = json.loads(info.read_text()) if info.exists() else {}
        meta[forecaster] = {"command": " ".join(sys.argv), "aeon": aeon,
                            "origins": args.origins, "windows": args.windows,
                            "deseasonalise": args.deseasonalise,
                            "finished": time.strftime("%c")}
        info.write_text(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
