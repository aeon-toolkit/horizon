"""DrCIF (Δy) on LSSH-99: deseasonalised, not deseasonalised, adjusted representation.

Three aaltd runs of the same forecaster (30 one-step origins per series):

- ``with``: the paper's run, M4-style deseasonalising up front (``--deseasonalise``);
- ``without``: no seasonal treatment (``scripts/aaltd_noseason.py``);
- ``drcif4``: the prototype with a seasonally adjusted fourth representation
  (``scripts/drcif4.py``, ``scratch/run_drcif4.sh lssh``), if present.

Per series: MAE relative to Naive. Summaries (geometric mean relative MAE,
series better/worse, Wilcoxon on MAE for each pair of runs) for all series, by
the treatment the ``with`` run applied (multiplicative, additive, none) and by
seasonal period.

    python scripts/lssh99_drcif_deseason_compare.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path("/local/ajb2u23/Results/Forecasting")
RUNS = {
    "with": RESULTS / "aaltd-8585c7bf/cpu/DrCIFDeltaY.csv",
    "without": RESULTS / "aaltd-8585c7bf-noseason/cpu/DrCIFDeltaY.csv",
    "drcif4": RESULTS / "aaltd-8585c7bf-drcif4/cpu/DrCIF4DeltaY.csv",
}
NAIVE = RESULTS / "aaltd-8585c7bf/cpu/Naive.csv"
PAIRS = [("with", "without"), ("with", "drcif4"), ("without", "drcif4")]
SUB_DAILY = {24, 48, 144, 1440}
OUT = ROOT / "results" / "aeon-8585c7bf" / "lssh99-drcif-deseason"


def _mae(d: pd.DataFrame) -> pd.Series:
    return (d.pred - d.actual).abs().groupby(d.problem).mean()


def per_series() -> pd.DataFrame:
    runs = {k: pd.read_csv(p) for k, p in RUNS.items() if p.exists()}
    w = runs["with"]
    t = pd.DataFrame({
        "period": w.groupby("problem").period.first(),
        "treatment": w.groupby("problem").deseasonalised.agg(
            lambda s: "/".join(sorted(set(s.astype(str))))),
        "naive": _mae(pd.read_csv(NAIVE)),
    })
    for k, d in runs.items():
        n = d.groupby("problem").size()
        if (n < n.max()).any() or len(n) < len(t):
            raise SystemExit(f"{k}: incomplete ({int(n.sum())} fits)")
        t[k] = _mae(d)
        t[f"rel_{k}"] = t[k] / t.naive
    return t[t.naive > 0]  # relative MAE undefined where Naive is exact


def summary(t: pd.DataFrame) -> pd.DataFrame:
    runs = [k for k in RUNS if k in t]
    groups = [("all", t), ("sub-daily", t[t.period.isin(SUB_DAILY)]),
              ("daily or slower", t[~t.period.isin(SUB_DAILY)])]
    groups += [(f"with: {k}", g) for k, g in t.groupby("treatment")]
    groups += [(f"period {p}", g) for p, g in t.groupby("period")]
    rows = []
    for name, g in groups:
        r = {"group": name, "series": len(g)}
        for k in runs:
            r[f"rel_{k}"] = float(np.exp(np.log(g[f"rel_{k}"]).mean()))
        for a, b in PAIRS:
            if a not in runs or b not in runs:
                continue
            x = g[a] - g[b]
            r[f"{a}<{b}"] = int((x < 0).sum())
            r[f"{a}>{b}"] = int((x > 0).sum())
            r[f"p_{a}_{b}"] = wilcoxon(x).pvalue if (x != 0).sum() > 5 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = per_series()
    t.to_csv(OUT / "per_series.csv")
    s = summary(t)
    s.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
