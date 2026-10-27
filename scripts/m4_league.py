"""M4 league table and the paper's Table (tab:m4), from the stored M4 forecasts.

Scores every aeon forecaster run on all 100,000 M4 series (result files written
by ``m_comp_experiments.py``) and the published M4 submissions, as in the M4
evaluation: sMAPE and MASE averaged over all forecast points, OWA relative to
Naive2, by frequency and overall. Writes:

- ``<out>/league.csv``: OWA by frequency and overall, overall sMAPE and MASE,
  median fit time, and the number of series whose forecast is not finite or
  explodes (an absolute error above 10 times the largest absolute value of the
  series, training and test).
- ``<out>/owa_all_forecasters.csv``: OWA only (read by other scripts).
- ``paper/m4_league.md``: the league as a page.
- ``paper/tab_m4.tex``: the subset shown in the paper.

    python scripts/m4_league.py --out results/aeon-8585c7bf/m4 --paper paper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "equivalence"))
from m4_score import (FREQS, M4_HOME, PERIOD, read_results, read_tsf_train,  # noqa: E402
                      read_wide, scores)

RESULTS = Path("/local/ajb2u23/Results/Forecasting")
# result directories searched in order; DL runs are in their own directory
DIRS = ["aeon-8585c7bf", "aeon-8585c7bf-dl"]
# forecaster -> (display name, group)
GROUPS = ["Published", "Statistical", "Machine learning", "Deep learning", "Baseline"]
NAMES = {
    "M4 ETS": ("R ets (published)", "Published"),
    "M4 ARIMA": ("R auto.arima (published)", "Published"),
    "M4 Theta": ("Theta (published)", "Published"),
    "M4 036": ("SCUM (published, entry 036)", "Published"),
    "SCUM": ("SCUM", "Statistical"),
    "DOTM": ("DOTM", "Statistical"),
    "AutoCES": ("AutoCES", "Statistical"),
    "Theta": ("Theta", "Statistical"),
    "AutoETS": ("AutoETS", "Statistical"),
    "AutoARIMA": ("AutoARIMA", "Statistical"),
    "CES": ("CES", "Statistical"),
    "ARIMA": ("ARIMA", "Statistical"),
    "ETS": ("ETS", "Statistical"),
    "TVP": ("TVP", "Statistical"),
    "TAR": ("TAR", "Statistical"),
    "AutoTAR": ("AutoTAR", "Statistical"),
    "RandF": ("Random forest", "Machine learning"),
    "DrCIFDeltaY": ("DrCIF (Δy)", "Machine learning"),
    "SETAR": ("SETAR", "Machine learning"),
    "SETARTree": ("SETARTree", "Machine learning"),
    "SETARForest": ("SETARForest", "Machine learning"),
    "DeepARDiff": ("DeepAR†", "Deep learning"),
    "NBeatsDiff": ("N-BEATS†", "Deep learning"),
    "TCNDiff": ("TCN†", "Deep learning"),
    "M4 Naive2": ("Naive2", "Baseline"),
    "NaiveForecaster": ("Naive", "Baseline"),
}
OFFICIAL = ["Naive2", "ETS", "ARIMA", "Theta", "036"]
EXPLODE = 10  # absolute error above this many times the series' largest |value|
# the paper's table, in blocks
PAPER = [["M4 ETS", "M4 ARIMA", "M4 036"],
         ["SCUM", "DOTM", "AutoCES", "Theta", "AutoETS", "AutoARIMA"],
         ["RandF", "DeepARDiff", "NBeatsDiff", "DrCIFDeltaY", "TCNDiff"]]
BLOCKS = [r"\emph{Published}", r"\emph{\texttt{aeon} statistical}",
          r"\emph{\texttt{aeon} machine learning and deep learning}"]
TEX_NAMES = {"M4 ETS": r"R \texttt{ets}", "M4 ARIMA": r"R \texttt{auto.arima}",
             "M4 036": "SCUM", "RandF": "Random forest", "DeepARDiff": r"DeepAR$^\dagger$",
             "NBeatsDiff": r"N-BEATS$^\dagger$", "DrCIFDeltaY": r"DrCIF ($\Delta y$)",
             "TCNDiff": r"TCN$^\dagger$"}


def load_runs() -> dict[str, pd.DataFrame]:
    runs = {}
    for f in NAMES:
        if f.startswith("M4 "):
            continue
        for d in DIRS:
            path = RESULTS / d / "aeon" / f / f"M4_{f}_Test.csv"
            if path.exists():
                runs[f] = read_results(path)
                break
        else:
            raise SystemExit(f"no M4 results for {f}")
        if len(runs[f]) != 100_000:
            raise SystemExit(f"{f}: {len(runs[f])} series, not 100000")
    return runs


def score(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-series sMAPE, MASE and failure flags of every run and submission."""
    official = {f"M4 {k}": read_wide(M4_HOME / "official" / f"submission-{k}.csv")
                for k in OFFICIAL}
    per = []
    for freq in FREQS:
        train = read_tsf_train(freq)
        ids = list(train)
        test = read_wide(M4_HOME / "official" / f"{freq.capitalize()}-test.csv")
        h = test.shape[1]
        actual = test.loc[ids].to_numpy(float)
        m = PERIOD[freq]
        scale = np.array([np.mean(np.abs(y[m:] - y[:-m])) for y in train.values()])
        level = np.maximum([np.max(np.abs(y)) for y in train.values()],
                           np.abs(actual).max(axis=1))
        fc = {}
        for name, df in runs.items():
            sub = df.loc[ids]
            ours = sub[[f"actual_{i + 1}" for i in range(h)]].to_numpy(float)
            if not np.allclose(ours, actual, rtol=1e-9, atol=1e-6):
                raise SystemExit(f"{name} {freq}: test values differ from the official set")
            fc[name] = (sub[[f"pred_{i + 1}" for i in range(h)]].to_numpy(float),
                        sub.train_seconds.to_numpy(float))
        for name, df in official.items():
            fc[name] = (df.loc[ids].iloc[:, :h].to_numpy(float), np.full(len(ids), np.nan))
        for name, (pred, secs) in fc.items():
            s, q = scores(pred, actual, scale)
            with np.errstate(invalid="ignore"):
                err = np.nanmax(np.abs(pred - actual), axis=1)
            per.append(pd.DataFrame({
                "id": ids, "frequency": freq, "method": name, "smape": s, "mase": q,
                "seconds": secs, "nonfinite": ~np.isfinite(pred).all(axis=1),
                "exploded": err > EXPLODE * level}))
    return pd.concat(per, ignore_index=True)


def league(per: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for freq in FREQS + ["all"]:
        d = per if freq == "all" else per[per.frequency == freq]
        g = d.groupby("method")[["smape", "mase"]].mean()
        base = g.loc["M4 Naive2"]
        cols[freq] = (g.smape / base.smape + g.mase / base.mase) / 2
        if freq == "all":
            cols["smape"], cols["mase"] = g.smape, g.mase
    g = per.groupby("method")
    out = pd.DataFrame(cols).join(pd.DataFrame({
        "median_fit_seconds": g.seconds.median(),
        "nonfinite_series": g.nonfinite.sum(),
        "exploded_series": g.exploded.sum()}))
    out.insert(0, "group", [NAMES[f][1] for f in out.index])
    out.insert(0, "name", [NAMES[f][0] for f in out.index])
    out["usable"] = out["all"] < 10  # the five exploding forecasters are far above
    return out.sort_values(["usable", "all"], ascending=[False, True])


def owa_cell(x: float) -> str:
    return f"{x:.3f}" if x < 100 else f"{x:.0f}" if x < 10_000 else f"{x:.0e}".replace("e+", "e")


def fmt_fit(s: float) -> str:
    if np.isnan(s):
        return "–"
    if s < 0.001:
        return "<0.001"
    return f"{s:.3f}" if s < 0.1 else f"{s:.2f}" if s < 10 else f"{s:.0f}"


def write_page(df: pd.DataFrame, n: dict[str, int], path: Path) -> None:
    head = " | ".join(f"{f.capitalize()} ({n[f]:,})" for f in FREQS)
    lines = [
        "# M4 league table",
        "",
        "Every aeon forecaster run on all 100,000 M4 series for the aeon paper",
        "(Section 5.2), aeon commit 8585c7bf, with the published M4 submissions of",
        "Naive2, R's `ets` and `auto.arima`, Theta and SCUM (entry 036). Standard M4",
        "protocol: one origin per series, the competition horizons, sMAPE and MASE",
        "averaged over all forecast points. **OWA**: overall weighted average",
        "relative to Naive2 (below 1: more accurate than Naive2).",
        "",
        "Non-seasonal, machine learning and deep learning forecasters are fitted to",
        "M4-deseasonalised series (classical multiplicative decomposition, 90% ACF",
        "test) and reseasonalised; seasonal models (AutoETS, ETS, SCUM, AutoCES, CES,",
        "DOTM, Naive) get the M4 period. The regression-based forecasters predict one",
        "step and forecast recursively. Δy: the regressor on windows of the series",
        "predicting the next change. †: input standardised and differenced, last",
        "1,000 points, fitted once.",
        "",
        "**Fit**: median seconds to fit one series (single-threaded worker, 60",
        "workers in parallel for the CPU runs). **Exploded**: series whose forecast",
        f"has an absolute error above {EXPLODE} times the largest absolute value of",
        "the series (training and test);",
        "**non-finite**: series with a NaN or infinite forecast (left out of the",
        "means). The last five forecasters explode on some series as shipped, so",
        "their OWA is not meaningful; the count of exploded series is the better",
        "summary.",
        "",
        "Generated by `scripts/m4_league.py`; see `paper/README.md`.",
        "",
        f"| # | Forecaster | Group | {head} | OWA | sMAPE | MASE | Fit (s) | Exploded | Non-finite |",
        "|---:|---|---|" + "---:|" * (len(FREQS) + 6),
    ]
    for i, (f, r) in enumerate(df.iterrows(), 1):
        name = f"**{r['name']}**" if f == "M4 Naive2" else r["name"]
        owa = " | ".join(owa_cell(r[q]) for q in FREQS)
        lines.append(
            f"| {i} | {name} | {r.group} | {owa} | **{owa_cell(r['all'])}** "
            f"| {owa_cell(r.smape)} | {owa_cell(r.mase)} | {fmt_fit(r.median_fit_seconds)} "
            f"| {int(r.exploded_series)} | {int(r.nonfinite_series)} |")
    lines.append("")
    path.write_text("\n".join(lines))


def write_tex(df: pd.DataFrame, n: dict[str, int], path: Path) -> None:
    rows = []
    for k, (label, block) in enumerate(zip(BLOCKS, PAPER)):
        if k:
            rows.append(r"\midrule")
        rows.append(rf"\multicolumn{{8}}{{l}}{{{label}}} \\")
        for f in block:
            name = TEX_NAMES.get(f, rf"\texttt{{{f}}}")
            vals = " & ".join(f"{df.loc[f, q]:.3f}" for q in FREQS + ["all"])
            rows.append(f"{name:<22} & {vals} \\\\")
    counts = " & ".join(f"{n[f]:,}".replace(",", "{,}") for f in FREQS + ["all"])
    body = "\n".join(rows)
    path.write_text(rf"""% Generated by scripts/m4_league.py (aeon 8585c7bf). Table body for tab:m4
% in the paper. Needs booktabs.
\begin{{tabular}}{{lccccccc}}
\toprule
 & Yearly & Quart. & Month. & Weekly & Daily & Hourly & All \\
 & {counts} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    per = score(load_runs())
    df = league(per)
    df.drop(columns="usable").to_csv(args.out / "league.csv")
    owa = df.loc[[f for f in df.index if not f.startswith("M4 ") or f == "M4 Naive2"],
                 FREQS + ["all"]]
    owa.rename_axis("method").to_csv(args.out / "owa_all_forecasters.csv")
    n = per[per.method == "M4 Naive2"].frequency.value_counts().to_dict()
    n["all"] = sum(n.values())
    write_page(df, n, args.paper / "m4_league.md")
    write_tex(df, n, args.paper / "tab_m4.tex")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 250):
        print(df.drop(columns=["group", "usable"]).to_string())


if __name__ == "__main__":
    main()
