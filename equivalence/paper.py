"""Paper tables and figures: aeon against the other toolkits.

- ``table_simulated.{md,tex}``: pooled accuracy of aeon AutoARIMA and AutoETS
  against each other toolkit on the simulated benchmark (both modes): geometric
  mean MASE ratio aeon / toolkit with 90% interval, the TOST verdict (±5%) and
  the Holm-corrected Wilcoxon signed-rank test.
- ``table_m4.{md,tex}``: aeon AutoETS and AutoARIMA against the published M4
  forecasts of R's ets and auto.arima, by frequency.
- ``fig_timing.{pdf,png}``: median fit-and-forecast time against series length
  for AutoARIMA (ARMA(1,1)) and AutoETS (ETS(A,A,A), m=12), auto mode;
  ``fig_timing_more.{pdf,png}`` the same for ARIMA(1,1,0) and ETS(A,A,N).

    python equivalence/paper.py --simulated equivalence/results/aeon-8585c7bf/simulated \\
        --significance results/aeon-8585c7bf --sweep results/aeon-8585c7bf/sweep-auto-r20 \\
        --m4 equivalence/results/aeon-8585c7bf/m4 --out equivalence/paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# fixed toolkit order, colour and marker (identity never by colour alone)
TOOLKITS = ["aeon", "statsforecast", "sktime", "skforecast", "pmdarima",
            "statsmodels", "smooth"]
COLOURS = dict(zip(TOOLKITS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                              "#e87ba4", "#008300", "#6250d6"]))
MARKERS = dict(zip(TOOLKITS, ["o", "s", "^", "D", "v", "P", "X"]))
LABELS = {"arima": "AutoARIMA", "ets": "AutoETS"}
PROCESS_LABELS = {"arma_11": "ARMA(1,1)", "arima_110": "ARIMA(1,1,0)",
                  "ets_aaa_m12": "ETS(A,A,A), m=12", "ets_aan": "ETS(A,A,N)"}
FREQS = ["yearly", "quarterly", "monthly", "weekly", "daily", "hourly", "all"]


def _toolkit(name: str) -> str:
    """Strip the ``(grid)`` suffix statsmodels carries in auto mode."""
    return name.split(" (")[0]


def simulated_table(tost: pd.DataFrame, sig_dir: Path) -> pd.DataFrame:
    """Pooled ratio, TOST verdict and Holm p of aeon against each toolkit."""
    sig = pd.concat(
        pd.read_csv(sig_dir / mode / "significance.csv").assign(mode=mode)
        for mode in ["auto", "fixed"]
    )
    key = ["mode", "family", "n", "toolkit"]
    t = tost[(tost["scope"] == "all") & ~tost["toolkit"].str.startswith("aeon")]
    s = sig[(sig["scope"] == "all") & ~sig["toolkit"].str.startswith("aeon")]
    df = t.merge(s[key + ["p_holm", "verdict"]], on=key,
                 suffixes=("_tost", "_wilcoxon"))
    df["toolkit"] = df["toolkit"].map(_toolkit)
    return df


def _cell(r) -> str:
    mark = "*" if r["p_holm"] < 0.05 else ""
    return f"{r['ratio']:.3f} ({r['ratio_ci_low']:.3f}-{r['ratio_ci_high']:.3f}){mark}"


def write_simulated(df: pd.DataFrame, out: Path) -> None:
    signif = df[df["p_holm"] < 0.05]
    if not (signif["verdict_wilcoxon"] == "aeon better").all():
        raise ValueError("a toolkit is significantly more accurate than aeon")
    cols = [(f, n) for f in ["arima", "ets"] for n in sorted(df["n"].unique())]
    md, tex = [], []
    head = " | ".join(f"{LABELS[f]} n={n}" for f, n in cols)
    for mode in ["auto", "fixed"]:
        d = df[df["mode"] == mode]
        md += [f"\n**{mode} mode**\n", f"| toolkit | {head} |",
               "|---" * (len(cols) + 1) + "|"]
        tex += [f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{mode} mode}}}} \\\\"]
        for tk in TOOLKITS[1:]:
            cells = []
            for f, n in cols:
                r = d[(d["toolkit"] == tk) & (d["family"] == f) & (d["n"] == n)]
                cells.append(_cell(r.iloc[0]) if len(r) else "-")
            md.append(f"| {tk} | " + " | ".join(cells) + " |")
            tex.append(f"{tk} & " + " & ".join(c.replace("*", "$^*$") for c in cells)
                       + " \\\\")
    note = ("Geometric mean of per-series MASE ratios aeon / toolkit over 600 "
            "series (6 processes x 100), 90% interval; < 1 means aeon is more "
            "accurate. * Wilcoxon signed-rank p < 0.05 after Holm correction "
            "(all such cases have aeon more accurate). TOST with a ±5% margin: "
            f"{(df['verdict_tost'] == 'equivalent').sum()} of {len(df)} equivalent, "
            f"{(df['verdict_tost'] == 'aeon better').sum()} aeon better, "
            f"{(df['verdict_tost'] == 'inconclusive').sum()} inconclusive "
            "(interval below 1, extending past the margin), "
            f"{(df['verdict_tost'] == 'aeon worse').sum()} aeon worse.")
    (out / "table_simulated.md").write_text("\n".join(md + ["", note, ""]))
    spec = "l" + "c" * len(cols)
    (out / "table_simulated.tex").write_text("\n".join(
        [f"% {note}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
         "toolkit & " + " & ".join(f"{LABELS[f]} $n={n}$" for f, n in cols)
         + " \\\\", "\\midrule", *tex, "\\bottomrule", "\\end{tabular}", ""]))


def write_m4(m4_dir: Path, out: Path) -> None:
    owa = pd.read_csv(m4_dir / "owa.csv").pivot(
        index="frequency", columns="method", values="owa")
    tost = pd.read_csv(m4_dir / "tost.csv")
    tost = tost[tost["metric"] == "mase"].set_index(["aeon", "frequency"])
    pairs = [("AutoETS", "M4 ETS", "R ets"), ("AutoARIMA", "M4 ARIMA", "R auto.arima")]
    head = "| frequency | " + " | ".join(
        f"aeon {a} | {lab} | ratio (90% CI) | TOST" for a, _, lab in pairs) + " |"
    md = [head, "|---" * (1 + 4 * len(pairs)) + "|"]
    tex = []
    for f in FREQS:
        cells = []
        for a, ref, _ in pairs:
            r = tost.loc[(a, f)]
            cells += [f"{owa.loc[f, a]:.3f}", f"{owa.loc[f, ref]:.3f}",
                      f"{r['ratio']:.3f} ({r['ratio_ci_low']:.3f}-{r['ratio_ci_high']:.3f})",
                      r["verdict"]]
        md.append(f"| {f} | " + " | ".join(cells) + " |")
        tex.append(f"{f} & " + " & ".join(cells) + " \\\\")
    note = ("OWA relative to Naive2 on the M4 test sets; ratio is the geometric "
            "mean of per-series MASE ratios aeon / published, 90% interval; "
            "TOST margin ±5%. Non-seasonal aeon AutoARIMA is fitted to "
            "M4-deseasonalised series; aeon has no seasonal ARIMA, hence hourly.")
    (out / "table_m4.md").write_text("\n".join(md + ["", note, ""]))
    (out / "table_m4.tex").write_text("\n".join(
        [f"% {note}", "\\begin{tabular}{l" + "cccc" * len(pairs) + "}", "\\toprule",
         *tex, "\\bottomrule", "\\end{tabular}", ""]))


def timing_figure(summary: pd.DataFrame, panels, path: Path) -> None:
    s = summary[~summary["toolkit"].str.startswith("aeon (")].copy()
    s["toolkit"] = s["toolkit"].map(_toolkit)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6),
                             sharey=True)
    for ax, (family, process) in zip(axes, panels):
        g = s[(s["family"] == family) & (s["process"] == process)]
        for tk in TOOLKITS:
            c = g[g["toolkit"] == tk].sort_values("n")
            if c.empty:
                continue
            lw, z = (2.0, 3) if tk == "aeon" else (1.2, 2)
            ax.plot(c["n"], c["median_s"], color=COLOURS[tk], lw=lw, zorder=z,
                    marker=MARKERS[tk], ms=4.5, mec="white", mew=0.6, label=tk)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{LABELS[family]}: {PROCESS_LABELS[process]}", fontsize=10)
        ax.set_xlabel("series length")
        ax.grid(True, which="major", color="0.9", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("fit and forecast time (s, median)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    for ext in ["pdf", "png"]:
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--simulated", type=Path, required=True,
                        help="output of simulated.py (tost.csv)")
    parser.add_argument("--significance", type=Path, required=True,
                        help="run directory holding auto/ and fixed/significance.csv")
    parser.add_argument("--sweep", type=Path, required=True,
                        help="auto-mode sweep directory (timing_summary.csv)")
    parser.add_argument("--m4", type=Path, required=True, help="output of m4.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = simulated_table(pd.read_csv(args.simulated / "tost.csv"), args.significance)
    write_simulated(df, args.out)
    write_m4(args.m4, args.out)
    summary = pd.read_csv(args.sweep / "timing_summary.csv")
    timing_figure(summary, [("arima", "arma_11"), ("ets", "ets_aaa_m12")],
                  args.out / "fig_timing")
    timing_figure(summary, [("arima", "arima_110"), ("ets", "ets_aan")],
                  args.out / "fig_timing_more")
    sweep_info = json.loads((args.sweep / "run_info.json").read_text())["aeon"]
    (args.out / "run_info.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()} | {"sweep_aeon": sweep_info},
        indent=2))
    print((args.out / "table_simulated.md").read_text())
    print((args.out / "table_m4.md").read_text())


if __name__ == "__main__":
    main()
