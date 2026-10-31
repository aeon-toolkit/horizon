"""Statistical forecasters on the long M4 series fitted to a window of recent history.

The M4 counterpart of ``lssh99_window.py``. Standard M4 protocol: one origin at
the end of the training series, the competition horizon h. For each series with
at least ``--min-length`` training points, each forecaster is fitted to the last
``w`` points (0: the whole training series) for each window in ``--windows``
shorter than the series, twice:

- ``split=test``: fitted to the training series, forecasting the test period;
- ``split=val``: fitted to the training series without its last h points,
  forecasting those h points (the M4 holdout), for choosing the window.

Non-seasonal forecasters are M4-deseasonalised with ``--deseasonalise``, as in the
paper's M4 runs.

Output: ``<out>/<forecaster>.csv``, one row per (series, split, window), with the
forecast as a JSON list; reruns skip rows already written and retry failed fits.

    PYTHONPATH=<aeon worktree> python scripts/m4_window.py --deseasonalise \\
        --forecasters AutoARIMA AutoETS ARIMA ETS --n-jobs 32 \\
        --out /local/ajb2u23/Results/Forecasting/m4-window-8585c7bf
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "equivalence"))
from lssh99_window import _insample, _warm_up  # noqa: E402
from m4_score import FREQS, PERIOD, read_tsf_train  # noqa: E402

HORIZON = dict(zip(FREQS, [6, 8, 18, 13, 14, 48]))
WINDOWS = [0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500]
_TRAIN: dict[str, np.ndarray] = {}


def _series(sid: str, freq: str) -> np.ndarray:
    if sid not in _TRAIN:
        _TRAIN.update(read_tsf_train(freq))
    return _TRAIN[sid]


def _one(task):
    """One fit to the window and an h-step forecast; returns a result row."""
    import m_comp_experiments as mc

    forecaster, sid, freq, split, window = task
    y = _series(sid, freq)
    h = HORIZON[freq]
    y = y[:-h] if split == "val" else y
    train = y if window == 0 else y[-window:]
    start = time.perf_counter()
    res = mc.fit_forecaster("aeon", forecaster, train, h, PERIOD[freq])
    pred = (np.ravel(res.forecast)[:h].tolist() if res.forecast is not None else None)
    return dict(forecaster=forecaster, id=sid, frequency=freq, split=split, window=window,
                n_train=train.size, status=res.status,
                fit_seconds=time.perf_counter() - start, model=res.model,
                deseasonalised=(res.params or {}).get("deseasonalised"),
                pred=json.dumps(pred), error=str(getattr(res, "error", "") or "")[:300],
                **_insample(res, train))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--forecasters", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--min-length", type=int, default=500)
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    parser.add_argument("--deseasonalise", action="store_true")
    args = parser.parse_args()
    if args.deseasonalise:
        os.environ["M_COMP_DESEASONALISE"] = "1"
    args.out.mkdir(parents=True, exist_ok=True)

    series = []  # (id, frequency, training length)
    for freq in FREQS:
        for sid, y in read_tsf_train(freq).items():
            if y.size >= args.min_length:
                series.append((sid, freq, y.size))
    print(f"{len(series)} series with >= {args.min_length} training points", flush=True)

    import m_comp_experiments as mc

    aeon = mc._aeon_info() if hasattr(mc, "_aeon_info") else None
    print("aeon", json.dumps(aeon, default=str), flush=True)
    for forecaster in args.forecasters:
        path = args.out / f"{forecaster}.csv"
        done = set()
        if path.exists():
            prev = pd.read_csv(path, usecols=["id", "split", "window", "status"])
            prev = prev[prev.status == "ok"]
            done = set(zip(prev.id, prev.split, prev.window))
        tasks = []
        for sid, freq, n in series:
            for split in ("val", "test"):
                avail = n - HORIZON[freq] if split == "val" else n
                for w in args.windows:
                    if (w == 0 or w < avail) and (sid, split, w) not in done:
                        tasks.append((forecaster, sid, freq, split, w))
        # one frequency at a time, so each worker reads few training files
        tasks.sort(key=lambda t: (t[2], -(t[4] or 10**6)))
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
                if len(batch) >= 500:
                    pd.DataFrame(batch).to_csv(path, mode="a", header=not path.exists(),
                                               index=False)
                    batch = []
                if j % 20000 == 0:
                    print(f"  {forecaster} {j}/{len(tasks)} {time.strftime('%c')}", flush=True)
            if batch:
                pd.DataFrame(batch).to_csv(path, mode="a", header=not path.exists(), index=False)
        df = pd.read_csv(path, usecols=["status"])
        print(f"=== {forecaster} done: {len(df)} rows, {(df.status != 'ok').sum()} not ok "
              f"{time.strftime('%c')}", flush=True)
        info = args.out / "run_info.json"
        meta = json.loads(info.read_text()) if info.exists() else {}
        meta[forecaster] = {"command": " ".join(sys.argv), "aeon": aeon,
                            "min_length": args.min_length, "windows": args.windows,
                            "deseasonalise": args.deseasonalise,
                            "finished": time.strftime("%c")}
        info.write_text(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
