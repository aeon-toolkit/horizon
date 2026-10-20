"""Comparisons between benchmark runs.

``compare_auto_fixed``
    Within one aeon version: each toolkit's automatic forecasts against its
    forecasts from the true model. Shows how much model selection costs.
``compare_runs``
    The same experiment run twice, typically with two versions of aeon (e.g. a
    release against a development branch). The competitors are identical in
    both runs, so they act as a control: their numbers should not move, while
    aeon's show the effect of the change.

Both rely on the runs using the same seeds, so rows pair up series by series.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from horizon.analyse import REFERENCE, _colours, _find, _holm, _save  # noqa: E402

__all__ = ["compare_auto_fixed", "compare_runs"]

KEY = ["family", "process", "n", "replicate", "toolkit"]


def _read(results_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(_find(Path(results_dir), name))


def _paired_wilcoxon(d: np.ndarray) -> float:
    try:
        return float(wilcoxon(d, zero_method="zsplit").pvalue)
    except ValueError:  # all differences zero
        return 1.0


def _run_label(results_dir: Path) -> str:
    """Short label for a run: its aeon version and git commit if recorded."""
    info = Path(results_dir) / "run_info.json"
    if info.exists():
        meta = json.loads(info.read_text())
        aeon = meta.get("aeon", {})
        commit = aeon.get("git_commit", "")[:9]
        return f"aeon {aeon.get('version', '?')}" + (f" @{commit}" if commit else "")
    return Path(results_dir).name


def compare_auto_fixed(auto_dir: Path, fixed_dir: Path) -> pd.DataFrame:
    """Compare each toolkit's automatic forecasts with its true model forecasts.

    For every toolkit, a paired Wilcoxon test on MASE says whether model
    selection costs accuracy relative to fitting the true model (Holm corrected
    within each family and length). Writes ``auto_vs_fixed.csv`` to
    ``fixed_dir``.
    """
    key = ["family", "process", "n", "replicate", "toolkit"]
    auto = _read(auto_dir, "accuracy.csv")
    fixed = _read(fixed_dir, "accuracy.csv")
    auto = auto[auto["toolkit"] != "aeon (auto sp)"].copy()
    auto["toolkit"] = auto["toolkit"].replace({"statsmodels (grid)": "statsmodels"})
    j = auto[key + ["seed", "mase"]].merge(
        fixed[key + ["seed", "mase"]], on=key, suffixes=("_auto", "_fixed")
    )
    if not (j["seed_auto"] == j["seed_fixed"]).all():
        raise ValueError("auto and fixed runs used different series")
    j = j.dropna(subset=["mase_auto", "mase_fixed"])
    rows = []
    for (family, n), block in j.groupby(["family", "n"]):
        scopes = [("all", block)] + list(block.groupby("process", sort=False))
        for scope, df in scopes:
            group = []
            for t, g in df.groupby("toolkit"):
                d = (g["mase_auto"] - g["mase_fixed"]).to_numpy()
                group.append(
                    {
                        "family": family,
                        "n": n,
                        "scope": scope,
                        "toolkit": t,
                        "n_series": len(d),
                        "median_mase_auto": float(g["mase_auto"].median()),
                        "median_mase_fixed": float(g["mase_fixed"].median()),
                        "median_diff_auto_minus_fixed": float(np.median(d)),
                        "p_value": _paired_wilcoxon(d),
                    }
                )
            for r, pa in zip(group, _holm([r["p_value"] for r in group])):
                r["p_holm"] = pa
            rows.extend(group)
    out = pd.DataFrame(rows)
    out.to_csv(Path(fixed_dir) / "auto_vs_fixed.csv", index=False)
    return out


def compare_runs(run_a: Path, run_b: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Compare two runs of the same experiment, e.g. two aeon versions.

    Writes to ``out_dir``:

    ``accuracy_change.csv``
        Per family, length and process (and pooled): median MASE of aeon in
        each run, the competitors' range, and a paired Wilcoxon test of aeon B
        against aeon A on the same series (Holm corrected within family and
        length). Also checks the competitors are unchanged.
    ``verdict_change.csv``
        aeon against each competitor in run A and in run B, from each run's
        ``significance.csv``.
    ``timing_change.csv`` and ``plots/timing_<process>.png``
        Median run time per length for aeon in both runs and the competitors.

    Parameters
    ----------
    run_a, run_b : Path
        Result directories of the baseline and the changed run.
    out_dir : Path
        Where to write the comparison.
    """
    run_a, run_b, out_dir = Path(run_a), Path(run_b), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_a, label_b = _run_label(run_a), _run_label(run_b)
    outputs = {}

    acc_a = _read(run_a, "accuracy.csv")
    acc_b = _read(run_b, "accuracy.csv")
    j = acc_a[KEY + ["seed", "mase", "time_s"]].merge(
        acc_b[KEY + ["seed", "mase", "time_s"]], on=KEY, suffixes=("_a", "_b")
    )
    if not (j["seed_a"] == j["seed_b"]).all():
        raise ValueError("the two runs used different series")
    j = j.dropna(subset=["mase_a", "mase_b"])
    rows = []
    for (family, n), block in j.groupby(["family", "n"]):
        scopes = [("all", block)] + list(block.groupby("process", sort=False))
        group = []
        for scope, df in scopes:
            ref = df[df["toolkit"] == REFERENCE]
            comp = df[~df["toolkit"].str.startswith("aeon")]
            if ref.empty:
                continue
            comp_med = comp.groupby("toolkit")["mase_b"].median()
            d = (ref["mase_b"] - ref["mase_a"]).to_numpy()
            # competitors are the control: their MASE should not change
            control = (comp["mase_b"] - comp["mase_a"]).abs().max()
            group.append(
                {
                    "family": family,
                    "n": n,
                    "scope": scope,
                    "n_series": len(d),
                    "aeon_a_median_mase": float(ref["mase_a"].median()),
                    "aeon_b_median_mase": float(ref["mase_b"].median()),
                    "competitors_min_median_mase": float(comp_med.min()),
                    "competitors_max_median_mase": float(comp_med.max()),
                    "aeon_b_wins": int(np.sum(d < -1e-12)),
                    "aeon_b_losses": int(np.sum(d > 1e-12)),
                    "p_value": _paired_wilcoxon(d),
                    "competitor_max_abs_change": float(control),
                    "aeon_a_median_time_s": float(ref["time_s_a"].median()),
                    "aeon_b_median_time_s": float(ref["time_s_b"].median()),
                }
            )
        for r, pa in zip(group, _holm([r["p_value"] for r in group])):
            r["p_holm"] = pa
            if pa >= 0.05:
                r["change"] = "no significant change"
            else:
                better = r["aeon_b_median_mase"] < r["aeon_a_median_mase"]
                r["change"] = "B better" if better else "B worse"
        rows.extend(group)
    acc = pd.DataFrame(rows)
    acc.to_csv(out_dir / "accuracy_change.csv", index=False)
    outputs["accuracy_change"] = acc

    sig_a = Path(run_a) / "significance.csv"
    sig_b = Path(run_b) / "significance.csv"
    if sig_a.exists() and sig_b.exists():
        cols = ["family", "n", "scope", "competitor"]
        keep = cols + ["reference_median", "competitor_median", "p_holm", "verdict"]
        v = pd.read_csv(sig_a)[keep].merge(
            pd.read_csv(sig_b)[keep], on=cols, suffixes=("_a", "_b")
        )
        v.to_csv(out_dir / "verdict_change.csv", index=False)
        outputs["verdict_change"] = v

    tim_a, tim_b = Path(run_a), Path(run_b)
    if _find(tim_a, "timing.csv").exists() and _find(tim_b, "timing.csv").exists():
        ta = _read(tim_a, "timing.csv")
        tb = _read(tim_b, "timing.csv")
        ma = ta.groupby(["family", "process", "n", "toolkit"])["time_s"].median()
        mb = tb.groupby(["family", "process", "n", "toolkit"])["time_s"].median()
        t = pd.DataFrame({"median_s_a": ma, "median_s_b": mb}).reset_index()
        t.to_csv(out_dir / "timing_change.csv", index=False)
        outputs["timing_change"] = t
        _plot_timing_change(t, label_a, label_b, out_dir / "plots")

    (out_dir / "runs.json").write_text(
        json.dumps({"a": str(run_a), "label_a": label_a, "b": str(run_b), "label_b": label_b}, indent=2)
    )
    return outputs


def _plot_timing_change(t: pd.DataFrame, label_a: str, label_b: str, plots_dir: Path):
    """Run time against length: aeon in both runs, competitors from run B."""
    colours = _colours(t["toolkit"].unique())
    for (family, process), block in t.groupby(["family", "process"]):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for tk, b in block.groupby("toolkit"):
            b = b.sort_values("n")
            if tk == REFERENCE:
                ax.plot(b["n"], b["median_s_a"], "o--", color=colours[tk], lw=2,
                        label=f"aeon ({label_a})")
                ax.plot(b["n"], b["median_s_b"], "o-", color=colours[tk], lw=2.5,
                        label=f"aeon ({label_b})")
            elif not tk.startswith("aeon"):
                ax.plot(b["n"], b["median_s_b"], "o-", color=colours[tk], lw=1.2,
                        alpha=0.8, label=tk)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("series length")
        ax.set_ylabel("fit + forecast time (s, median)")
        ax.set_title(f"Auto{family.upper()} run time, {process}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        _save(fig, plots_dir, f"timing_{family}_{process}")
