"""Experiment runners: forecast accuracy and run time vs series length.

Both experiments write one CSV row per (series, toolkit) as soon as each block
finishes, and skip rows already present on restart, so long runs can be
interrupted and resumed.

``run_accuracy``
    For every process, length and replicate, simulate ``n + h`` points, give
    each toolkit the first ``n`` and score its ``h`` step forecast against the
    held out continuation of the known process. Every toolkit sees exactly the
    same series. Writes ``accuracy.csv`` (metrics, time, selected model) and
    ``forecasts.csv`` (every point forecast, for forecast agreement analysis).
``run_timing``
    For a fixed process, increasing lengths and a few repeats, record the wall
    clock time of one fit-and-forecast call per toolkit. A toolkit whose median
    time at some length exceeds ``time_budget`` is not run on longer series.

JIT compilation (aeon, statsforecast) is paid once by ``Adapter.warmup`` before
any timed call. Run times are wall clock from :func:`time.perf_counter`, single
threaded when launched via ``run_all.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from horizon.adapters import get_adapters
from horizon.simulate import TRUE_SPECS, get_processes

__all__ = ["run_accuracy", "run_timing", "horizon_for", "score"]

RESULTS_DIR = Path("results")

ACCURACY_KEY = ["process", "n", "replicate", "toolkit"]
TIMING_KEY = ["process", "n", "repeat", "toolkit"]


def horizon_for(sp: int) -> int:
    """Forecast horizon: 12 steps, or two full seasons if that is longer."""
    return max(12, 2 * sp)


def score(train: np.ndarray, test: np.ndarray, pred: np.ndarray, sp: int) -> dict:
    """Return MAE, RMSE, MASE and sMAPE of ``pred`` against ``test``.

    MASE is scaled by the in-sample MAE of the seasonal naive forecast with lag
    ``sp`` (lag 1 for non-seasonal processes).
    """
    err = test - pred
    lag = sp if sp > 1 and len(train) > sp else 1
    scale = np.mean(np.abs(train[lag:] - train[:-lag]))
    denom = np.abs(test) + np.abs(pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape = np.where(denom > 0, 2.0 * np.abs(err) / denom, 0.0)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mase": float(np.mean(np.abs(err)) / scale) if scale > 0 else np.nan,
        "smape": float(100.0 * np.mean(smape)),
    }


def _seed(process_index: int, n: int, replicate: int, offset: int) -> int:
    """Deterministic, collision free seed per series."""
    return int(offset + 1_000_000 * process_index + 1_000 * replicate + n % 1_000)


def _load_done(path: Path, key: list[str]) -> set[tuple]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done = pd.read_csv(path, usecols=key)
    return set(map(tuple, done[key].astype(str).itertuples(index=False)))


def _append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _call(adapter, y, h, sp):
    """Run one adapter, returning (pred, seconds, error message)."""
    adapter.selected_model = ""
    start = time.perf_counter()
    try:
        pred = adapter(y, h, sp)
        err = ""
    except Exception as e:  # record the failure, keep the benchmark going
        pred = np.full(h, np.nan)
        err = f"{type(e).__name__}: {e}"[:300]
    elapsed = time.perf_counter() - start
    if not err and not np.all(np.isfinite(pred)):
        err = "non-finite forecast"
    return pred, elapsed, err


def run_accuracy(
    families=("arima", "ets"),
    processes: list[str] | None = None,
    lengths=(100, 500),
    n_replicates: int = 100,
    toolkits: list[str] | None = None,
    results_dir: Path = RESULTS_DIR,
    seed_offset: int = 0,
    verbose: bool = True,
    mode: str = "auto",
) -> pd.DataFrame:
    """Run the accuracy experiment and return the full results table.

    Parameters
    ----------
    families : iterable of str
        Families to run, ``"arima"`` and/or ``"ets"``.
    processes : list of str or None
        Restrict to these process names.
    lengths : iterable of int
        Training series lengths.
    n_replicates : int
        Independent series per (process, length).
    toolkits : list of str or None
        Restrict to these adapter names.
    results_dir : Path
        Output directory for ``accuracy.csv`` and ``forecasts.csv``.
    seed_offset : int
        Added to every seed, to draw a different set of series.
    verbose : bool
        Print one progress line per (process, length).
    mode : {"auto", "fixed"}
        Automatic model selection, or fit the true generating model. With the
        same ``seed_offset`` both modes see identical series.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    acc_path = results_dir / "accuracy.csv"
    fc_path = results_dir / "forecasts.csv"
    done = _load_done(acc_path, ACCURACY_KEY)

    all_names = [p.name for p in get_processes()]
    for family in families:
        adapters = get_adapters(family, include=toolkits, mode=mode)
        for proc in get_processes(family, processes):
            sp = proc.seasonal_period
            h = horizon_for(sp)
            for a in adapters:
                a.spec = TRUE_SPECS[proc.name]
                a.warmup(sp)
            pidx = all_names.index(proc.name)
            for n in lengths:
                rows, fc_rows = [], []
                start = time.perf_counter()
                for r in range(n_replicates):
                    todo = [
                        a
                        for a in adapters
                        if (proc.name, str(n), str(r), a.name) not in done
                    ]
                    if not todo:
                        continue
                    seed = _seed(pidx, n, r, seed_offset)
                    y = proc.sample(n + h, random_state=seed)
                    train, test = y[:n], y[n:]
                    for a in todo:
                        pred, elapsed, err = _call(a, train, h, sp)
                        metrics = (
                            score(train, test, pred, sp)
                            if not err
                            else dict.fromkeys(["mae", "rmse", "mase", "smape"])
                        )
                        rows.append(
                            {
                                "mode": mode,
                                "family": family,
                                "process": proc.name,
                                "sp": sp,
                                "n": n,
                                "h": h,
                                "replicate": r,
                                "seed": seed,
                                "toolkit": a.name,
                                "time_s": elapsed,
                                **metrics,
                                "selected_model": a.selected_model,
                                "error": err,
                            }
                        )
                        fc_rows.extend(
                            {
                                "family": family,
                                "process": proc.name,
                                "n": n,
                                "replicate": r,
                                "toolkit": a.name,
                                "step": i + 1,
                                "y_true": test[i],
                                "y_pred": pred[i],
                            }
                            for i in range(h)
                        )
                _append(acc_path, rows)
                _append(fc_path, fc_rows)
                if verbose and rows:
                    block = pd.DataFrame(rows)
                    summary = block.groupby("toolkit")["mase"].median().round(3)
                    print(
                        f"[accuracy] {proc.name:16s} n={n:<5d} "
                        f"{len(rows):4d} runs {time.perf_counter() - start:7.1f}s "
                        f"median MASE {summary.to_dict()}",
                        flush=True,
                    )
    return pd.read_csv(acc_path)


def run_timing(
    processes: dict[str, list[str]] | None = None,
    lengths=(100, 200, 500, 1000, 2000, 5000, 10000),
    n_repeats: int = 5,
    toolkits: list[str] | None = None,
    time_budget: float = 30.0,
    results_dir: Path = RESULTS_DIR,
    seed_offset: int = 10_000_000,
    verbose: bool = True,
    mode: str = "auto",
) -> pd.DataFrame:
    """Run the run time vs series length experiment.

    Parameters
    ----------
    processes : dict of family -> list of process names, or None
        Defaults to ``{"arima": ["arma_11"], "ets": ["ets_aaa_m12"]}``.
    lengths : iterable of int
        Series lengths, run in increasing order.
    n_repeats : int
        Series per length. Each repeat is a fresh series, shared by toolkits.
    toolkits : list of str or None
        Restrict to these adapter names.
    time_budget : float
        Seconds. Once a toolkit's median time at a length exceeds this, it is
        skipped for all longer lengths of that process.
    results_dir : Path
        Output directory for ``timing.csv``.
    seed_offset : int
        Added to every seed. Defaults to keep timing series distinct from the
        accuracy series.
    verbose : bool
        Print one progress line per (process, length).
    mode : {"auto", "fixed"}
        Automatic model selection, or fit the true generating model.
    """
    if processes is None:
        processes = {"arima": ["arma_11"], "ets": ["ets_aaa_m12"]}
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "timing.csv"
    done = _load_done(path, TIMING_KEY)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()

    all_names = [p.name for p in get_processes()]
    for family, names in processes.items():
        adapters = get_adapters(family, include=toolkits, mode=mode)
        for proc in get_processes(family, names):
            sp = proc.seasonal_period
            h = horizon_for(sp)
            for a in adapters:
                a.spec = TRUE_SPECS[proc.name]
                a.warmup(sp)
            pidx = all_names.index(proc.name)
            over_budget: set[str] = set()
            # restore budget state from a previous partial run
            if not existing.empty:
                prev = existing[existing["process"] == proc.name]
                med = prev.groupby(["toolkit", "n"])["time_s"].median()
                over_budget |= {t for (t, _), v in med.items() if v > time_budget}
            for n in sorted(lengths):
                rows = []
                for r in range(n_repeats):
                    todo = [
                        a
                        for a in adapters
                        if a.name not in over_budget
                        and (proc.name, str(n), str(r), a.name) not in done
                    ]
                    if not todo:
                        continue
                    y = proc.sample(n, random_state=_seed(pidx, n, r, seed_offset))
                    for a in todo:
                        _, elapsed, err = _call(a, y, h, sp)
                        rows.append(
                            {
                                "mode": mode,
                                "family": family,
                                "process": proc.name,
                                "sp": sp,
                                "n": n,
                                "h": h,
                                "repeat": r,
                                "toolkit": a.name,
                                "time_s": elapsed,
                                "selected_model": a.selected_model,
                                "error": err,
                            }
                        )
                _append(path, rows)
                if rows:
                    med = pd.DataFrame(rows).groupby("toolkit")["time_s"].median()
                    newly = {t for t, v in med.items() if v > time_budget}
                    over_budget |= newly
                    if verbose:
                        print(
                            f"[timing] {proc.name:16s} n={n:<6d} median s "
                            f"{med.round(4).to_dict()}"
                            + (f"  over budget: {sorted(newly)}" if newly else ""),
                            flush=True,
                        )
    return pd.read_csv(path)

