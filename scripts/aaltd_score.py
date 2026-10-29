"""Score the aaltd rolling-origin one-step forecasts (``aaltd_experiments.py``).

Per problem and forecaster: MAE over the 30 one-step forecasts, MASE (MAE over
the in-sample MAE of the one-step naive forecast on the series before the first
origin) and relative MAE (MAE over the Naive forecaster's MAE). The summary
gives, per forecaster, the geometric mean of relative MAE and the mean MASE over
the problems it completed, its mean rank by MAE (among the forecasters that completed every problem),
and its failed fits. Problems whose training series is constant are left out,
as is relative MAE on problems where the Naive forecast is exact.

``--forecasters`` names a file listing the forecasters to score, one per line
(the paper's fixed set, ``paper/lssh99_forecasters.txt``); a listed forecaster
with fits missing is an error, so later runs in the same directory cannot
change the scores.

    python scripts/aaltd_score.py --results /local/.../aaltd-8585c7bf \\
        --forecasters paper/lssh99_forecasters.txt --out results/aeon-8585c7bf/aaltd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aaltd_experiments import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--forecasters", type=Path,
                        help="file listing the forecasters to score, one per line")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = pd.concat([pd.read_csv(p) for d in args.results for p in sorted(d.rglob("*.csv"))],
                     ignore_index=True)
    rows = rows.drop_duplicates(["forecaster", "problem", "origin"], keep="last")
    origins = rows.origin.max() + 1
    if args.forecasters:
        names = args.forecasters.read_text().split()
        fits = rows.groupby("forecaster").size().reindex(names, fill_value=0)
        expected = rows.problem.nunique() * origins
        if (fits < expected).any():
            raise SystemExit(f"fits missing (of {expected}):\n{fits[fits < expected]}")
        rows = rows[rows.forecaster.isin(names)]
    scale = {}
    for p in rows.problem.unique():
        y = load(p)[: -origins]
        scale[p] = np.mean(np.abs(np.diff(y)))

    constant = sorted(p for p, v in scale.items() if v == 0)
    rows = rows[~rows.problem.isin(constant)]  # nothing to forecast; MASE undefined
    rows["abs_err"] = (rows.pred - rows.actual).abs()
    ok = rows[(rows.status == "ok") & rows.pred.notna()]
    per = (ok.groupby(["forecaster", "problem"])
           .agg(mae=("abs_err", "mean"), n=("abs_err", "size"),
                fit_seconds=("fit_seconds", "mean")).reset_index())
    per = per[per.n == origins]  # problems with every origin forecast
    per["mase"] = per.mae / per.problem.map(scale)
    naive = per[per.forecaster == "Naive"].set_index("problem").mae
    naive = naive[naive > 0]  # relative MAE undefined where Naive is exact
    per["rel_mae"] = per.mae / per.problem.map(naive)
    per.to_csv(args.out / "per_problem.csv", index=False)

    wide = per.pivot(index="problem", columns="forecaster", values="mae")
    # rank the forecasters that finished every problem, on all problems
    full = wide.columns[wide.notna().sum() == wide.shape[0]]
    complete = wide[full].dropna()
    ranks = complete.rank(axis=1).mean().rename("mean_rank")
    failed = (rows[rows.status != "ok"].groupby("forecaster").size()
              .rename("failed_fits"))
    summary = per.groupby("forecaster").agg(
        problems=("problem", "size"),
        gmean_rel_mae=("rel_mae", lambda r: float(np.exp(np.log(r.dropna()).mean()))),
        mean_mase=("mase", "mean"),
        median_fit_seconds=("fit_seconds", "median"),
    ).join(ranks).join(failed).fillna({"failed_fits": 0})
    summary = summary.sort_values("gmean_rel_mae")
    summary.to_csv(args.out / "summary.csv")
    print(f"constant series left out: {constant}")
    print(f"mean rank: {len(full)} forecasters that completed all {len(complete)} problems")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(summary.to_string())


if __name__ == "__main__":
    main()
