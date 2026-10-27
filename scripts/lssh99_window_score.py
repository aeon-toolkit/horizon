"""Score the LSSH-99 training-window experiment (``lssh99_window.py``).

Origins 0-29 are the validation period, 30-59 the test period (the paper's 30
test origins). Per forecaster, each method below gives one forecast per test
origin; per series, its test MAE is divided by the naive (last value) MAE.

- ``full``: the whole series (the paper's setting).
- ``w=<k>``: the last k observations (the whole series where it is shorter).
- ``validation``: per series, the window with the lowest MAE over the
  validation origins.
- ``rolling``: at each test origin, the window with the lowest MAE over the
  previous 30 origins.
- ``global``: one window for every series, the one with the lowest geometric
  mean relative MAE over the validation origins.
- ``average``: the mean of the forecasts from every window fitted (those
  shorter than the history, and the whole series), no selection.
- ``oracle``: per series, the window with the lowest test MAE (not a real
  method: an upper bound on what selecting one window per series can gain).

Writes ``per_problem.csv`` (relative MAE per method and series), ``summary.csv``
(geometric mean relative MAE on the sub-daily, daily-or-slower and all series;
ratio to ``full``; series better / worse than ``full``; Wilcoxon signed-rank p
value against ``full``, Holm-corrected within each forecaster; median fit time),
and ``selected_windows.csv`` (how often each window is chosen).

    python scripts/lssh99_window_score.py \\
        --results /local/ajb2u23/Results/Forecasting/lssh99-window-8585c7bf \\
        --out results/aeon-8585c7bf/lssh99-window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from lssh99_league import sub_daily  # noqa: E402

VALIDATION = 30  # origins 0..29; the rest are test origins
ROLL = 30  # rolling selection looks back this many origins


def gmean(x) -> float:
    x = pd.Series(x).dropna()
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = p.notna().sum()
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        if np.isnan(p[k]):
            adj[k] = np.nan
            continue
        running = max(running, min(1.0, (m - i) * p[k]))
        adj[k] = running
    return pd.Series(adj).reindex(p.index)


def predictions(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wide tables (problem, origin) x window of predictions, and actual/naive.

    In the first table, windows not run because they are at least as long as the
    history take the whole-series prediction; the second holds only the
    predictions actually made.
    """
    ok = rows[(rows.status == "ok") & rows.pred.notna()]
    pred = ok.pivot_table(index=["problem", "origin"], columns="window", values="pred")
    raw = pred.copy()
    full = pred[0]
    for w in pred.columns:
        pred[w] = pred[w].fillna(full) if w else pred[w]
    truth = rows.drop_duplicates(["problem", "origin"]).set_index(
        ["problem", "origin"])[["actual", "naive"]]
    pred = pred.sort_index()
    return pred, raw.loc[pred.index], truth.loc[pred.index]


def score(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    pred, raw, truth = predictions(rows)
    err = pred.sub(truth.actual, axis=0).abs()
    naive_err = (truth.naive - truth.actual).abs()
    origin = err.index.get_level_values("origin")
    val, test = origin < VALIDATION, origin >= VALIDATION
    windows = list(err.columns)
    # candidate order for ties: the whole series first, then longest to shortest
    order = [0] + sorted((w for w in windows if w), reverse=True)

    test_mae = err[test].groupby("problem").mean()
    naive_test = naive_err[test].groupby("problem").mean()
    val_mae = err[val].groupby("problem").mean()
    naive_val = naive_err[val].groupby("problem").mean()

    methods = {"full": test_mae[0]}
    for w in windows[1:]:
        methods[f"w={w}"] = test_mae[w]
    # per-series validation choice (ties go to the longer window)
    choice_val = val_mae[order].idxmin(axis=1)
    methods["validation"] = pd.Series(
        {p: test_mae.loc[p, w] for p, w in choice_val.items()})
    # rolling choice at each test origin
    roll_err, roll_choice = [], []
    for p, e in err.groupby("problem"):
        e = e.droplevel("problem")
        for o in e.index[e.index >= VALIDATION]:
            past = e.loc[o - ROLL:o - 1, order].mean()
            w = past.idxmin()
            roll_err.append((p, e.loc[o, w]))
            roll_choice.append(w)
    methods["rolling"] = pd.DataFrame(roll_err, columns=["problem", "e"]).groupby("problem").e.mean()
    # one window for all series, chosen on validation
    rel_val = val_mae.div(naive_val.where(naive_val > 0), axis=0)
    w_global = max(windows, key=lambda w: -gmean(rel_val[w]))
    methods["global"] = test_mae[w_global]
    # average over the windows fitted (each distinct window counted once)
    avg = raw.mean(axis=1).sub(truth.actual).abs()
    methods["average"] = avg[test].groupby("problem").mean()
    methods["oracle"] = test_mae.min(axis=1)

    rel = pd.DataFrame(methods).div(naive_test.where(naive_test > 0), axis=0)
    chosen = pd.concat([
        choice_val.value_counts().rename("validation"),
        pd.Series(roll_choice).value_counts().rename("rolling"),
        pd.Series({w_global: len(choice_val)}).rename("global"),
        test_mae[order].idxmin(axis=1).value_counts().rename("oracle"),
    ], axis=1).reindex(windows).fillna(0).astype(int)
    fit = rows.groupby("window").fit_seconds.median()
    return rel, chosen, fit


def summarise(rel: pd.DataFrame, fit: pd.Series) -> pd.DataFrame:
    sd = rel.index.map(sub_daily).to_numpy(bool)
    out = []
    for m in rel.columns:
        r, base = rel[m], rel["full"]
        both = r.notna() & base.notna()
        diff = np.log(r[both]) - np.log(base[both])
        p = (wilcoxon(diff).pvalue if m != "full" and (diff != 0).any() else np.nan)
        w = int(m[2:]) if m.startswith("w=") else (0 if m == "full" else None)
        out.append(dict(method=m, sub_daily=gmean(r[sd]), daily_or_slower=gmean(r[~sd]),
                        all=gmean(r), ratio_to_full=gmean(r[both] / base[both]),
                        better=int((diff < 0).sum()), worse=int((diff > 0).sum()),
                        p_value=p, median_fit_seconds=fit.get(w, np.nan) if w is not None else np.nan))
    df = pd.DataFrame(out).set_index("method")
    df["p_holm"] = holm(df.p_value)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    per, summ, sel = [], [], []
    for path in sorted(args.results.glob("*.csv")):
        rows = pd.read_csv(path)
        f = rows.forecaster.iloc[0]
        bad = rows[rows.status != "ok"]
        if len(bad):
            print(f"{f}: {len(bad)} fits not ok (windows {sorted(bad.window.unique())})")
        rel, chosen, fit = score(rows)
        per.append(rel.rename_axis("problem").assign(forecaster=f).reset_index())
        summ.append(summarise(rel, fit).assign(forecaster=f).reset_index())
        sel.append(chosen.rename_axis("window").assign(forecaster=f).reset_index())
    per = pd.concat(per)
    summ = pd.concat(summ)
    pd.concat(sel).to_csv(args.out / "selected_windows.csv", index=False)
    per.to_csv(args.out / "per_problem.csv", index=False)
    summ.to_csv(args.out / "summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        for f, d in summ.groupby("forecaster"):
            print(f"\n== {f}")
            print(d.drop(columns="forecaster").set_index("method").to_string())


if __name__ == "__main__":
    main()
