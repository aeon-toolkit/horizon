"""Length sweep: is accuracy equivalent, and how does run time grow with n?

The main benchmark tests whether forecasts differ at two lengths. The sweep
asks the question needed before comparing toolkits on speed alone: across a
fine grid of lengths, are the toolkits' forecasts *equivalent* in accuracy?
A non-significant difference test does not show that, so the sweep uses an
equivalence test (TOST) on the paired MASE ratio.

``run_sweep_accuracy``
    Accuracy experiment over every (process, length), run in parallel worker
    processes, one job per (process, length), each writing its own
    ``parts/<process>_n<n>/accuracy.csv``, merged into ``accuracy.csv``.
    Run times recorded here are not used: parallel jobs contend for memory
    bandwidth.
``run_sweep_timing``
    Run time against length on a few processes, a handful of jobs in parallel
    so they do not contend, merged into ``timing.csv``.
``equivalence``
    For aeon against each other toolkit, per (family, length) pooled over
    processes and per process: the geometric mean of the per-series MASE ratio
    aeon / toolkit with a ``1 - 2 alpha`` confidence interval (paired t on
    the log ratio), and the TOST p-value for the ratio lying within
    ``[1 / (1 + margin), 1 + margin]``.
``analyse_sweep``
    Writes ``equivalence.csv``, ``accuracy_summary.csv``,
    ``timing_summary.csv`` and plots.

Every length gets its own seeds (``_length_seed_offset``), so series of
different lengths are independent rather than prefixes of each other.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from horizon.simulate import get_processes

__all__ = [
    "SWEEP_LENGTHS",
    "SWEEP_TIMING_PROCESSES",
    "run_sweep_accuracy",
    "run_sweep_timing",
    "equivalence",
    "analyse_sweep",
]

SWEEP_LENGTHS = list(range(100, 1001, 100)) + list(range(2000, 10001, 1000))
SWEEP_TIMING_PROCESSES = {
    "arima": ["arma_11", "arima_110"],
    "ets": ["ets_aan", "ets_aaa_m12"],
}
REFERENCE = "aeon"


def _length_seed_offset(n: int) -> int:
    """Seed offset per length; per-series seeds add < 2e7 (see ``_seed``)."""
    return 20_000_000 * (n // 100) + 7


def _merge(out: Path, name: str) -> pd.DataFrame:
    parts = sorted((out / "parts").glob(f"*/{name}"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(out / name, index=False)
    return df


def _accuracy_job(family, process, n, replicates, mode, out, toolkits):
    from horizon.experiments import run_accuracy

    start = time.perf_counter()
    run_accuracy(
        families=[family],
        processes=[process],
        lengths=[n],
        n_replicates=replicates,
        toolkits=toolkits,
        results_dir=out / "parts" / f"{process}_n{n}",
        seed_offset=_length_seed_offset(n),
        verbose=False,
        mode=mode,
    )
    return process, n, time.perf_counter() - start


def _timing_job(family, process, lengths, repeats, mode, out, toolkits, budget):
    from horizon.experiments import run_timing

    start = time.perf_counter()
    run_timing(
        processes={family: [process]},
        lengths=lengths,
        n_repeats=repeats,
        toolkits=toolkits,
        time_budget=budget,
        results_dir=out / "parts" / f"timing_{process}",
        verbose=False,
        mode=mode,
    )
    return process, None, time.perf_counter() - start


def _run_pool(fn, jobs, workers, label):
    # spawn: workers start clean rather than inheriting numba/BLAS state
    ctx = mp.get_context("spawn")
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(fn, *job) for job in jobs]
        for f in as_completed(futures):
            process, n, secs = f.result()
            done += 1
            where = f"{process} n={n}" if n is not None else process
            print(f"[{label}] {done}/{len(jobs)} {where} {secs:.0f}s", flush=True)


def run_sweep_accuracy(
    out: Path,
    mode: str = "auto",
    lengths=SWEEP_LENGTHS,
    replicates: int = 50,
    workers: int = 8,
    families=("arima", "ets"),
    toolkits: list[str] | None = None,
) -> pd.DataFrame:
    """Run the accuracy sweep in parallel and return the merged results."""
    out = Path(out)
    jobs = [
        (family, proc.name, n, replicates, mode, out, toolkits)
        for family in families
        for proc in get_processes(family)
        for n in lengths
    ]
    # longest first, so the pool is not left waiting on one long job
    jobs.sort(key=lambda j: -j[2])
    _run_pool(_accuracy_job, jobs, workers, f"accuracy {mode}")
    return _merge(out, "accuracy.csv")


def run_sweep_timing(
    out: Path,
    mode: str = "auto",
    lengths=SWEEP_LENGTHS,
    repeats: int = 5,
    processes: dict[str, list[str]] | None = None,
    toolkits: list[str] | None = None,
    time_budget: float = 30.0,
) -> pd.DataFrame:
    """Run the timing experiment, one parallel job per process."""
    out = Path(out)
    processes = SWEEP_TIMING_PROCESSES if processes is None else processes
    jobs = [
        (family, name, list(lengths), repeats, mode, out, toolkits, time_budget)
        for family, names in processes.items()
        for name in names
    ]
    _run_pool(_timing_job, jobs, len(jobs), f"timing {mode}")
    return _merge(out, "timing.csv")


def _tost(d: np.ndarray, margin: float, alpha: float) -> dict:
    """Equivalence of the mean of log ratios ``d`` to 0 within +-log(1+margin)."""
    k = len(d)
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(k)) if k > 1 else np.nan
    bound = np.log1p(margin)
    if not se > 0:
        p_tost = 0.0 if abs(mean) < bound else 1.0
        lo = hi = mean
        p_diff = 1.0 if mean == 0 else 0.0
    else:
        t_crit = stats.t.ppf(1 - alpha, k - 1)
        lo, hi = mean - t_crit * se, mean + t_crit * se
        p_low = stats.t.sf((mean + bound) / se, k - 1)  # H0: mean <= -bound
        p_high = stats.t.cdf((mean - bound) / se, k - 1)  # H0: mean >= bound
        p_tost = float(max(p_low, p_high))
        p_diff = float(2 * stats.t.sf(abs(mean) / se, k - 1))
    return {
        "n_series": k,
        "ratio": float(np.exp(mean)),
        "ratio_ci_low": float(np.exp(lo)),
        "ratio_ci_high": float(np.exp(hi)),
        "p_tost": p_tost,
        "p_difference": p_diff,
    }


def equivalence(
    acc: pd.DataFrame, margin: float = 0.05, alpha: float = 0.05
) -> pd.DataFrame:
    """TOST equivalence of aeon and each other toolkit on paired MASE ratios.

    ``ratio`` < 1 means aeon is more accurate. ``verdict`` is "equivalent" if
    the TOST p-value is below ``alpha`` (the ``1 - 2 alpha`` interval lies
    within the margin), "aeon better" or "aeon worse" if the interval lies
    entirely outside the margin, and "inconclusive" otherwise.
    """
    acc = acc[acc["error"].isna() | (acc["error"] == "")]
    key = ["family", "process", "n", "replicate"]
    wide = acc.pivot_table(index=key, columns="toolkit", values="mase").reset_index()
    upper, lower = 1 + margin, 1 / (1 + margin)
    rows = []
    for tk in [c for c in wide.columns if c not in key and c != REFERENCE]:
        pair = wide[key + [REFERENCE, tk]].dropna()
        pair = pair[(pair[REFERENCE] > 0) & (pair[tk] > 0)]
        pair = pair.assign(d=np.log(pair[REFERENCE] / pair[tk]))
        for (family, n), g in pair.groupby(["family", "n"]):
            for scope, gg in [("all", g), *g.groupby("process")]:
                r = _tost(gg["d"].to_numpy(), margin, alpha)
                if r["p_tost"] < alpha:
                    verdict = "equivalent"
                elif r["ratio_ci_high"] < lower:
                    verdict = "aeon better"
                elif r["ratio_ci_low"] > upper:
                    verdict = "aeon worse"
                else:
                    verdict = "inconclusive"
                rows.append(
                    {"family": family, "n": n, "scope": scope, "toolkit": tk,
                     **r, "margin": margin, "verdict": verdict}
                )
    return pd.DataFrame(rows)


def plot_equivalence(eq: pd.DataFrame, plots_dir: Path, margin: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from horizon.analyse import _colours, _save

    pooled = eq[eq["scope"] == "all"]
    for family, g in pooled.groupby("family"):
        toolkits = sorted(g["toolkit"].unique())
        colours = _colours(toolkits)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.axhspan(1 / (1 + margin), 1 + margin, color="0.9", zorder=0,
                   label=f"equivalence margin ±{margin:.0%}")
        ax.axhline(1, color="0.5", lw=0.8)
        for i, tk in enumerate(toolkits):
            c = g[g["toolkit"] == tk].sort_values("n")
            x = c["n"] * (1 + 0.02 * (i - len(toolkits) / 2))
            ax.errorbar(x, c["ratio"],
                        yerr=[c["ratio"] - c["ratio_ci_low"],
                              c["ratio_ci_high"] - c["ratio"]],
                        fmt="o-", ms=3, lw=1, capsize=2, color=colours[tk],
                        label=tk)
        ax.set_xscale("log")
        ax.set_xlabel("series length n")
        ax.set_ylabel("MASE ratio aeon / toolkit (geometric mean)")
        ax.set_title(f"{family.upper()}: accuracy relative to aeon, 90% CI")
        ax.legend(fontsize=8, ncol=2)
        _save(fig, plots_dir, f"equivalence_{family}")


def plot_speedup(summary: pd.DataFrame, plots_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from horizon.analyse import _colours, _save

    for (family, process), g in summary.groupby(["family", "process"]):
        ref = g[g["toolkit"] == REFERENCE].set_index("n")["median_s"]
        toolkits = [t for t in sorted(g["toolkit"].unique()) if t != REFERENCE]
        colours = _colours(toolkits)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.axhline(1, color="0.5", lw=0.8)
        for tk in toolkits:
            c = g[g["toolkit"] == tk].set_index("n")["median_s"]
            ratio = (c / ref).dropna()
            ax.plot(ratio.index, ratio.values, "o-", ms=3, color=colours[tk],
                    label=tk)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("series length n")
        ax.set_ylabel("toolkit time / aeon time (>1: aeon faster)")
        ax.set_title(f"{family.upper()} {process}: speed up of aeon")
        ax.legend(fontsize=8)
        _save(fig, plots_dir, f"speedup_{family}_{process}")


def analyse_sweep(out: Path, margin: float = 0.05, alpha: float = 0.05) -> None:
    """Equivalence tables, summaries and plots of a sweep directory."""
    from horizon.analyse import (
        _find,
        accuracy_summary,
        plot_timing,
        timing_summary,
    )

    out = Path(out)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    acc_path = _find(out, "accuracy.csv")
    if acc_path.exists():
        acc = pd.read_csv(acc_path)
        accuracy_summary(acc).to_csv(out / "accuracy_summary.csv", index=False)
        eq = equivalence(acc, margin, alpha)
        eq.to_csv(out / "equivalence.csv", index=False)
        plot_equivalence(eq, plots, margin)
        pooled = eq[eq["scope"] == "all"]
        print(pooled.groupby(["family", "toolkit"])["verdict"]
              .value_counts().unstack(fill_value=0).to_string())
    tim_path = _find(out, "timing.csv")
    if tim_path.exists():
        summary = timing_summary(pd.read_csv(tim_path))
        summary.to_csv(out / "timing_summary.csv", index=False)
        plot_timing(summary, plots)
        plot_speedup(summary, plots)
