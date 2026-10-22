"""Statistical tests, summary tables and plots from the experiment CSVs.

Reads ``results/accuracy.csv``, ``results/forecasts.csv`` and
``results/timing.csv`` (whichever exist) and writes:

``results/accuracy_summary.csv``
    Per family, process, length and toolkit: mean/median MASE, RMSE, mean time,
    failure count, and how often the true generating model was selected.
``results/significance.csv``
    aeon against every other toolkit, per (family, length) pooled over processes
    and per process: paired two-sided Wilcoxon signed-rank test on per-series
    MASE, Holm corrected within each (family, length, scope) group, with
    win/tie/loss counts and the median paired difference.
``results/forecast_agreement.csv``
    How close aeon's point forecasts are to each other toolkit's, relative to
    aeon's own forecast error, and how often both select the same model.
``results/timing_summary.csv``
    Median and quartile run time per process, length and toolkit, and speed up
    of aeon over each other toolkit.
``plots/``
    Run time vs length (log-log), MASE boxplots per process, and critical
    difference diagrams per (family, length). Saved as PNG and PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from horizon.simulate import TRUE_MODELS  # noqa: E402

__all__ = ["analyse_all"]

RESULTS_DIR = Path("results")
REFERENCE = "aeon"
TIES = 1e-12


def _find(results_dir: Path, name: str) -> Path:
    """``results_dir / name``, or its gzipped version if only that exists."""
    path = Path(results_dir) / name
    gz = path.with_name(name + ".gz")
    return gz if not path.exists() and gz.exists() else path


def _toolkit_order(names) -> list[str]:
    """aeon entries first, then the rest alphabetically, for stable plots."""
    names = list(dict.fromkeys(names))
    head = [n for n in names if n.startswith("aeon")]
    return sorted(head) + sorted(n for n in names if not n.startswith("aeon"))


def _colours(names) -> dict[str, str]:
    cmap = plt.get_cmap("tab10")
    return {n: cmap(i % 10) for i, n in enumerate(_toolkit_order(names))}


def _save(fig, plots_dir: Path, stem: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaNs are passed through)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv[order])
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def accuracy_summary(acc: pd.DataFrame) -> pd.DataFrame:
    """Per (family, process, n, toolkit) summary statistics."""
    acc = acc.copy()
    acc["failed"] = acc["error"].fillna("").astype(str).str.len() > 0
    acc["true_model"] = acc["process"].map(TRUE_MODELS)
    sel = acc["selected_model"].fillna("").astype(str).str.split(" ").str[0]
    acc["correct_model"] = sel == acc["true_model"]
    g = acc.groupby(["family", "process", "n", "toolkit"], sort=False)
    out = g.agg(
        runs=("mase", "size"),
        failures=("failed", "sum"),
        mean_mase=("mase", "mean"),
        median_mase=("mase", "median"),
        mean_rmse=("rmse", "mean"),
        median_rmse=("rmse", "median"),
        mean_time_s=("time_s", "mean"),
        median_time_s=("time_s", "median"),
        true_model_rate=("correct_model", "mean"),
    ).reset_index()
    out["true_model"] = out["process"].map(TRUE_MODELS)
    return out


def _paired_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rows are series (process, replicate), columns toolkits, complete only."""
    wide = df.pivot_table(
        index=["process", "replicate"], columns="toolkit", values=metric
    )
    return wide.dropna(axis=0, how="any")


def significance(acc: pd.DataFrame, metric: str = "mase") -> pd.DataFrame:
    """aeon vs each other toolkit, Wilcoxon signed-rank on paired ``metric``."""
    rows = []
    for (family, n), block in acc.groupby(["family", "n"]):
        scopes = [("all", block)] + [
            (proc, b) for proc, b in block.groupby("process", sort=False)
        ]
        for scope, df in scopes:
            wide = _paired_matrix(df, metric)
            if REFERENCE not in wide:
                continue
            group = []
            for other in _toolkit_order(wide.columns):
                if other == REFERENCE:
                    continue
                a, b = wide[REFERENCE].to_numpy(), wide[other].to_numpy()
                d = a - b
                try:
                    p = wilcoxon(a, b, zero_method="zsplit").pvalue
                except ValueError:  # all differences zero
                    p = 1.0
                group.append(
                    {
                        "family": family,
                        "n": n,
                        "scope": scope,
                        "reference": REFERENCE,
                        "toolkit": other,
                        "metric": metric,
                        "n_series": len(d),
                        "reference_median": float(np.median(a)),
                        "toolkit_median": float(np.median(b)),
                        "median_diff": float(np.median(d)),
                        "reference_wins": int(np.sum(d < -TIES)),
                        "ties": int(np.sum(np.abs(d) <= TIES)),
                        "reference_losses": int(np.sum(d > TIES)),
                        "p_value": p,
                    }
                )
            if group:
                adj = _holm([r["p_value"] for r in group])
                for r, pa in zip(group, adj):
                    r["p_holm"] = pa
                    r["significant_0.05"] = bool(pa < 0.05)
                    if not r["significant_0.05"]:
                        r["verdict"] = "no significant difference"
                    elif r["median_diff"] < 0:
                        r["verdict"] = f"{REFERENCE} better"
                    else:
                        r["verdict"] = f"{r['toolkit']} better"
                rows.extend(group)
    return pd.DataFrame(rows)


def forecast_agreement(fc: pd.DataFrame, acc: pd.DataFrame) -> pd.DataFrame:
    """How far each other toolkit's forecasts are from aeon's.

    ``rel_forecast_diff`` is mean |aeon - other| over the horizon divided by
    mean |y - aeon|: 0 means identical forecasts, values well below 1 mean the
    disagreement between toolkits is small compared with forecast error.
    """
    key = ["family", "process", "n", "replicate", "step"]
    ref = fc[fc["toolkit"] == REFERENCE].set_index(key)
    rows = []
    sel = acc.set_index(["family", "process", "n", "replicate", "toolkit"])[
        "selected_model"
    ]
    sel = sel.fillna("").astype(str).str.split(" ").str[0]
    for other, block in fc[fc["toolkit"] != REFERENCE].groupby("toolkit"):
        j = block.set_index(key).join(ref[["y_pred"]], rsuffix="_ref", how="inner")
        j = j.dropna(subset=["y_pred", "y_pred_ref"])
        j["abs_diff"] = (j["y_pred"] - j["y_pred_ref"]).abs()
        j["abs_err_ref"] = (j["y_true"] - j["y_pred_ref"]).abs()
        per_series = j.groupby(level=["family", "process", "n", "replicate"]).agg(
            diff=("abs_diff", "mean"), err=("abs_err_ref", "mean")
        )
        per_series["rel"] = per_series["diff"] / per_series["err"].replace(0, np.nan)
        for (family, process, n), g in per_series.groupby(
            level=["family", "process", "n"]
        ):
            idx_ref = [(family, process, n, r, REFERENCE) for r in g.index.get_level_values(3)]
            idx_oth = [(family, process, n, r, other) for r in g.index.get_level_values(3)]
            same = (
                sel.reindex(idx_ref).to_numpy() == sel.reindex(idx_oth).to_numpy()
            )
            rows.append(
                {
                    "family": family,
                    "process": process,
                    "n": n,
                    "toolkit": other,
                    "n_series": len(g),
                    "median_rel_forecast_diff": float(g["rel"].median()),
                    "mean_rel_forecast_diff": float(g["rel"].mean()),
                    "same_model_rate": float(np.mean(same)),
                }
            )
    return pd.DataFrame(rows)


def plot_mase_boxplots(acc: pd.DataFrame, plots_dir: Path) -> None:
    colours = _colours(acc["toolkit"].unique())
    for (family, n), block in acc.groupby(["family", "n"]):
        procs = list(dict.fromkeys(block["process"]))
        tk = _toolkit_order(block["toolkit"].unique())
        ncol = 3
        nrow = int(np.ceil(len(procs) / ncol))
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False
        )
        for ax, proc in zip(axes.flat, procs):
            data = [
                block.loc[
                    (block["process"] == proc) & (block["toolkit"] == t), "mase"
                ].dropna()
                for t in tk
            ]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
            for patch, t in zip(bp["boxes"], tk):
                patch.set_facecolor(colours[t])
                patch.set_alpha(0.75)
            for med in bp["medians"]:
                med.set_color("black")
            ax.set_title(f"{proc}  (true {TRUE_MODELS.get(proc, '?')})", fontsize=9)
            ax.set_xticks(range(1, len(tk) + 1))
            ax.set_xticklabels(tk, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("MASE")
            ax.grid(axis="y", alpha=0.3)
        for ax in list(axes.flat)[len(procs) :]:
            ax.set_visible(False)
        fig.suptitle(f"{family.upper()} forecast accuracy, n={n}")
        fig.tight_layout()
        _save(fig, plots_dir, f"mase_boxplot_{family}_n{n}")


def plot_cd(acc: pd.DataFrame, plots_dir: Path) -> None:
    from aeon.visualisation import plot_critical_difference

    for (family, n), block in acc.groupby(["family", "n"]):
        wide = _paired_matrix(block, "mase")
        if wide.shape[1] < 2 or wide.shape[0] < 5:
            continue
        labels = _toolkit_order(wide.columns)
        try:
            fig, ax = plot_critical_difference(
                wide[labels].to_numpy(),
                labels,
                lower_better=True,
                test="wilcoxon",
                correction="holm",
                alpha=0.05,
            )
        except Exception as e:
            print(f"critical difference plot failed for {family} n={n}: {e}")
            continue
        ax.set_title(
            f"{family.upper()} n={n}: mean rank of MASE over {len(wide)} series"
        )
        _save(fig, plots_dir, f"cd_{family}_n{n}")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def timing_summary(tim: pd.DataFrame) -> pd.DataFrame:
    ok = tim[tim["error"].fillna("").astype(str) == ""]
    g = ok.groupby(["family", "process", "n", "toolkit"])["time_s"]
    out = g.agg(
        repeats="size",
        median_s="median",
        q25_s=lambda x: x.quantile(0.25),
        q75_s=lambda x: x.quantile(0.75),
        mean_s="mean",
    ).reset_index()
    ref = out[out["toolkit"] == REFERENCE].set_index(["process", "n"])["median_s"]
    out["aeon_speedup"] = out["median_s"] / out.set_index(["process", "n"]).index.map(
        ref
    ).to_numpy(dtype=float)
    return out


def plot_timing(summary: pd.DataFrame, plots_dir: Path) -> None:
    colours = _colours(summary["toolkit"].unique())
    for (family, process), block in summary.groupby(["family", "process"]):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for t in _toolkit_order(block["toolkit"].unique()):
            b = block[block["toolkit"] == t].sort_values("n")
            lw = 2.5 if t.startswith("aeon") else 1.5
            ax.plot(b["n"], b["median_s"], "o-", label=t, color=colours[t], lw=lw)
            ax.fill_between(b["n"], b["q25_s"], b["q75_s"], color=colours[t], alpha=0.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("series length")
        ax.set_ylabel("fit + forecast time (s, median)")
        ax.set_title(f"Auto{family.upper()} run time, {process}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, plots_dir, f"timing_{family}_{process}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyse_all(results_dir: Path = RESULTS_DIR, plots_dir: Path | None = None) -> None:
    """Build every table and plot from whatever result files exist.

    Plots go to ``plots_dir``, by default ``results_dir / "plots"``.
    """
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots" if plots_dir is None else Path(plots_dir)
    acc_path = _find(results_dir, "accuracy.csv")
    fc_path = _find(results_dir, "forecasts.csv")
    tim_path = _find(results_dir, "timing.csv")

    if acc_path.exists():
        acc = pd.read_csv(acc_path)
        summ = accuracy_summary(acc)
        summ.to_csv(results_dir / "accuracy_summary.csv", index=False)
        sig = significance(acc)
        sig.to_csv(results_dir / "significance.csv", index=False)
        if fc_path.exists():
            agree = forecast_agreement(pd.read_csv(fc_path), acc)
            agree.to_csv(results_dir / "forecast_agreement.csv", index=False)
        plot_mase_boxplots(acc, plots_dir)
        plot_cd(acc, plots_dir)
        pooled = sig[sig["scope"] == "all"]
        if not pooled.empty:
            print("\naeon vs other toolkits, pooled over processes (MASE, Holm):")
            cols = [
                "family",
                "n",
                "toolkit",
                "n_series",
                "reference_median",
                "toolkit_median",
                "reference_wins",
                "reference_losses",
                "p_holm",
                "verdict",
            ]
            print(pooled[cols].to_string(index=False, float_format="%.4g"))

    if tim_path.exists():
        ts = timing_summary(pd.read_csv(tim_path))
        ts.to_csv(results_dir / "timing_summary.csv", index=False)
        plot_timing(ts, plots_dir)
        print("\nmedian run time (s):")
        print(
            ts.pivot_table(index=["process", "n"], columns="toolkit", values="median_s")
            .round(4)
            .to_string()
        )


