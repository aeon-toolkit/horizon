"""LSSH-99 league table and the paper's Table (tab:lssh), from the scored results.

Reads ``per_problem.csv`` and ``summary.csv`` written by ``aaltd_score.py`` and
writes, for every forecaster scored:

- ``<results>/league.csv``: relative MAE (geometric mean over series) on the
  sub-daily series, the daily-or-slower series and all series; mean MASE; mean
  rank by MAE; median refit time; problems completed and failed fits.
- ``paper/lssh99_league.md``: the same as a page.
- ``paper/tab_lssh.tex``: the subset shown in the paper.

Sub-daily means a Monash seasonal period of 24 or more other than weekly (52)
or monthly (12): half-hourly, hourly, 10-minute and minutely series.

    python scripts/lssh99_league.py results/aeon-8585c7bf/aaltd --paper paper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from aaltd_experiments import period_of  # noqa: E402

# forecaster -> (display name, group); the order of groups in the page
GROUPS = ["Statistical", "Machine learning", "Deep learning", "Baseline"]
NAMES = {
    "SCUM": ("SCUM", "Statistical"),
    "AutoARIMA": ("AutoARIMA", "Statistical"),
    "ARIMA": ("ARIMA", "Statistical"),
    "AutoETS": ("AutoETS", "Statistical"),
    "ETS": ("ETS", "Statistical"),
    "AutoCES": ("AutoCES", "Statistical"),
    "CES": ("CES", "Statistical"),
    "Theta": ("Theta", "Statistical"),
    "DOTM": ("DOTM", "Statistical"),
    "TAR": ("TAR", "Statistical"),
    "AutoTAR": ("AutoTAR", "Statistical"),
    "TVP": ("TVP", "Statistical"),
    "DrCIFDeltaY": ("DrCIF (Δy)", "Machine learning"),
    "CIFDeltaY": ("CIF (Δy)", "Machine learning"),
    "TSFDeltaY": ("TSF (Δy)", "Machine learning"),
    "RISEDeltaY": ("RISE (Δy)", "Machine learning"),
    "Catch22DeltaY": ("Catch22 (Δy)", "Machine learning"),
    "SummaryDeltaY": ("Summary (Δy)", "Machine learning"),
    "RandFDeltaY": ("Random forest (Δy)", "Machine learning"),
    "RandF": ("Random forest", "Machine learning"),
    "RidgeDiff": ("Ridge (Δ)", "Machine learning"),
    "RegressionForecaster": ("Ridge", "Machine learning"),
    "SETARDiff": ("SETAR (Δ)", "Machine learning"),
    "SETAR": ("SETAR", "Machine learning"),
    "SETARTreeDiff": ("SETARTree (Δ)", "Machine learning"),
    "SETARTree": ("SETARTree", "Machine learning"),
    "SETARForestDiff": ("SETARForest (Δ)", "Machine learning"),
    "SETARForest": ("SETARForest", "Machine learning"),
    "TCNDiff": ("TCN†", "Deep learning"),
    "DeepARDiff": ("DeepAR†", "Deep learning"),
    "NBeatsDiff": ("N-BEATS†", "Deep learning"),
    "Naive": ("Naive", "Baseline"),
    "SeasonalNaive": ("Seasonal naive", "Baseline"),
}
# the paper's table: machine/deep learning block, then statistical block
PAPER = [["DrCIFDeltaY", "RandFDeltaY", "TCNDiff", "CIFDeltaY", "RidgeDiff", "TSFDeltaY",
          "DeepARDiff", "NBeatsDiff", "RegressionForecaster"],
         ["SCUM", "AutoARIMA", "AutoETS", "Theta"]]
TEX_NAMES = {"SCUM": r"\texttt{SCUM}", "AutoARIMA": r"\texttt{AutoARIMA}",
             "AutoETS": r"\texttt{AutoETS}", "Theta": r"\texttt{Theta}",
             "TCNDiff": r"TCN$^\dagger$", "DeepARDiff": r"DeepAR$^\dagger$",
             "NBeatsDiff": r"N-BEATS$^\dagger$"}


def sub_daily(problem: str) -> bool:
    return period_of(problem) not in (7, 12, 52)


def gmean(x: pd.Series) -> float:
    x = x.dropna()
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def league(results: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    per = pd.read_csv(results / "per_problem.csv")
    summary = pd.read_csv(results / "summary.csv", index_col="forecaster")
    per["sub_daily"] = per.problem.map(sub_daily)
    naive = per[(per.forecaster == "Naive") & per.rel_mae.notna()]
    counts = {"sub_daily": int(naive.sub_daily.sum()),
              "daily_or_slower": int((~naive.sub_daily).sum()), "all": len(naive)}
    out = pd.DataFrame({
        "sub_daily": per[per.sub_daily].groupby("forecaster").rel_mae.apply(gmean),
        "daily_or_slower": per[~per.sub_daily].groupby("forecaster").rel_mae.apply(gmean),
        "all": per.groupby("forecaster").rel_mae.apply(gmean),
    })
    out = out.join(summary[["mean_mase", "mean_rank", "median_fit_seconds",
                            "problems", "failed_fits"]])
    unnamed = sorted(set(out.index) - set(NAMES))
    if unnamed:
        raise SystemExit(f"no display name for {unnamed}")
    out.insert(0, "group", [NAMES[f][1] for f in out.index])
    out.insert(0, "name", [NAMES[f][0] for f in out.index])
    return out.sort_values("all"), counts


def fmt_fit(s: float) -> str:
    if s < 0.001:
        return "<0.001"
    return f"{s:.3f}" if s < 0.1 else f"{s:.2f}" if s < 10 else f"{s:.0f}"


def write_page(df: pd.DataFrame, counts: dict[str, int], n_rank: int, path: Path) -> None:
    lines = [
        "# LSSH-99 league table",
        "",
        "Every forecaster run on LSSH-99 for the aeon paper (Section 5.3), aeon commit",
        "8585c7bf. Each forecaster is refitted at each of the last 30 points of each",
        "series and forecasts one step ahead (3,000 fits). **Rel. MAE**: MAE relative",
        "to the naive (last value) forecast, geometric mean over series (below 1:",
        "better than naive). **Mean rank**: by MAE, among the"
        f" {n_rank} forecasters that completed",
        "every series (1 is best). **Fit**: median seconds per refit, one thread.",
        "",
        f"Series: {counts['sub_daily']} sub-daily (half-hourly, hourly, 10-minute,"
        f" minutely), {counts['daily_or_slower']} daily or slower,",
        f"{counts['all']} in all. wind_farms T2 is constant and left out; on one more"
        " series the naive",
        "forecast is exact over the last 30 points, so relative MAE is undefined there",
        "(its MASE and rank are still counted, over 99 series).",
        "",
        "Δy: a time series regressor on windows of 100 values predicting the next change.",
        "Δ: the model on (standardised) first differences. Unmarked: the model on the raw",
        "values. †: input standardised and differenced, last 1,000 points, fitted once.",
        "Non-seasonal and machine learning forecasters are deseasonalised as on M4.",
        "",
        "Generated by `scripts/lssh99_league.py` from `results/aeon-8585c7bf/aaltd/`;",
        "see `paper/README.md`.",
        "",
        "| # | Forecaster | Group | Sub-daily | Daily or slower | All | Mean MASE | Mean rank | Fit (s) |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, (f, r) in enumerate(df.iterrows(), 1):
        name = f"**{r['name']}**" if f == "Naive" else r["name"]
        rank = "–" if pd.isna(r.mean_rank) else f"{r.mean_rank:.1f}"
        lines.append(
            f"| {i} | {name} | {r.group} | {r.sub_daily:.3f} | {r.daily_or_slower:.3f} "
            f"| {r['all']:.3f} | {r.mean_mase:.2f} | {rank} | {fmt_fit(r.median_fit_seconds)} |")
    failed = df[df.failed_fits > 0]
    lines += ["", "## Failures", ""]
    for f, r in failed.iterrows():
        lines.append(f"- {r['name']}: {int(r.failed_fits)} fits failed; scored on"
                     f" {int(r.problems)} series and left out of the ranks. aeon raises a"
                     " singular-matrix error when a regime is all zeros (solar, weather,"
                     " wind series).")
    lines += ["- QUANT (Δy) is not included: it needs torch, which is not installed in"
              " the benchmark environment.", ""]
    path.write_text("\n".join(lines))


def write_tex(df: pd.DataFrame, counts: dict[str, int], path: Path) -> None:
    def name(f: str) -> str:
        if f in TEX_NAMES:
            return TEX_NAMES[f]
        return NAMES[f][0].replace("Δy", r"$\Delta y$").replace("Δ", r"$\Delta$")

    def fit(s: float) -> str:
        return f"{s:.3f}" if s < 0.01 else f"{s:.2f}" if s < 1 else f"{s:.1f}" if s < 10 else f"{s:.0f}"

    rows = []
    for k, block in enumerate(PAPER):
        if k:
            rows.append(r"\midrule")
        for f in block:
            r = df.loc[f]
            rows.append(f"{name(f):<26} & {r.sub_daily:.3f} & {r.daily_or_slower:.3f} "
                        f"& {r['all']:.3f} & {fit(r.median_fit_seconds)} \\\\")
    body = "\n".join(rows)
    path.write_text(rf"""% Generated by scripts/lssh99_league.py from results/aeon-8585c7bf/aaltd
% (aeon 8585c7bf). Table body for tab:lssh in the paper. Needs booktabs.
\begin{{tabular}}{{lcccr}}
\toprule
 & Sub-daily & Daily or slower & All & Fit (s) \\
 & {counts['sub_daily']} & {counts['daily_or_slower']} & {counts['all']} & \\
\midrule
{body}
\bottomrule
\end{{tabular}}
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path, help="output directory of aaltd_score.py")
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    args = parser.parse_args()
    df, counts = league(args.results)
    df.to_csv(args.results / "league.csv")
    n_rank = int(df.mean_rank.notna().sum())
    write_page(df, counts, n_rank, args.paper / "lssh99_league.md")
    write_tex(df, counts, args.paper / "tab_lssh.tex")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(df.drop(columns="group").to_string())


if __name__ == "__main__":
    main()
