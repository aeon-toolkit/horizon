"""DrCIF (Δy) with and without M4 deseasonalising on the long M4 series.

Compares the paper's deseasonalised run (``with``) with
``scratch/run_m4_long_noseason.sh`` (``without``: no deseasonalising) and, once
run, the prototype with a seasonally adjusted fourth representation
(``drcif4``: ``scripts/drcif4.py``, ``scratch/run_drcif4.sh m4``) on the series with at least 500 training points and a
seasonal period (monthly, quarterly, hourly). Per series: sMAPE and MASE as in
M4; per frequency: means, OWA against the official Naive2, and a Wilcoxon test
on per-series sMAPE and MASE for each pair of runs. Series are split by whether the 90% seasonality
test fired (multiplicative deseasonalising; all M4 series are positive).

    PYTHONPATH=<aeon worktree> python scripts/m4_long_deseason_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "equivalence"))
from m4_score import M4_HOME, PERIOD, read_results, read_tsf_train, read_wide, scores  # noqa: E402
from m_comp_experiments import _m4_seasonality_test  # noqa: E402

RESULTS = Path("/local/ajb2u23/Results/Forecasting")
RUNS = {
    "with": RESULTS / "aeon-8585c7bf/aeon/DrCIFDeltaY/M4_DrCIFDeltaY_Test.csv",
    "without": RESULTS / "aeon-8585c7bf-noseason-long/aeon/DrCIFDeltaY/M4_DrCIFDeltaY_Test.csv",
    "drcif4": RESULTS / "aeon-8585c7bf-drcif4-long/aeon/DrCIF4DeltaY/M4_DrCIF4DeltaY_Test.csv",
}
PAIRS = [("with", "without"), ("with", "drcif4"), ("without", "drcif4")]
FREQS = ["quarterly", "monthly", "hourly"]
MIN_LENGTH = 500
OUT = ROOT / "results" / "aeon-8585c7bf" / "m4-long-deseason"


def per_series() -> pd.DataFrame:
    runs = {k: read_results(p) for k, p in RUNS.items() if p.exists()}
    naive2 = read_wide(M4_HOME / "official" / "submission-Naive2.csv")
    out = []
    for freq in FREQS:
        train = {k: y for k, y in read_tsf_train(freq).items() if y.size >= MIN_LENGTH}
        ids = list(train)
        test = read_wide(M4_HOME / "official" / f"{freq.capitalize()}-test.csv")
        h = test.shape[1]
        actual = test.loc[ids].to_numpy(float)
        m = PERIOD[freq]
        scale = np.array([np.mean(np.abs(y[m:] - y[:-m])) for y in train.values()])
        fc = {k: r.loc[ids, [f"pred_{i + 1}" for i in range(h)]].to_numpy(float)
              for k, r in runs.items()}
        fc["Naive2"] = naive2.loc[ids].iloc[:, :h].to_numpy(float)
        d = pd.DataFrame({"id": ids, "frequency": freq, "n": [y.size for y in train.values()],
                          "seasonal": [bool(_m4_seasonality_test(y, m)) for y in train.values()]})
        for k, p in fc.items():
            d[f"smape_{k}"], d[f"mase_{k}"] = scores(p, actual, scale)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def summary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [(f, d[d.frequency == f]) for f in FREQS] + [("all", d)]
    groups += [(f"all, seasonal={s}", d[d.seasonal == s]) for s in [True, False]]
    for name, g in groups:
        r = {"group": name, "series": len(g), "seasonal": int(g.seasonal.sum())}
        runs = [k for k in RUNS if f"smape_{k}" in g]
        for k in runs:
            r[f"smape_{k}"] = g[f"smape_{k}"].mean()
            r[f"mase_{k}"] = g[f"mase_{k}"].mean()
            r[f"owa_{k}"] = (r[f"smape_{k}"] / g.smape_Naive2.mean()
                             + r[f"mase_{k}"] / g.mase_Naive2.mean()) / 2
        for a, b in PAIRS:
            if a not in runs or b not in runs:
                continue
            diff = g[f"mase_{a}"] - g[f"mase_{b}"]
            r[f"{a}<{b}"] = int((diff < 0).sum())
            r[f"{a}>{b}"] = int((diff > 0).sum())
            for k in ["smape", "mase"]:
                x = g[f"{k}_{a}"] - g[f"{k}_{b}"]
                r[f"p_{k}_{a}_{b}"] = wilcoxon(x).pvalue if (x != 0).sum() > 5 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = per_series()
    d.to_csv(OUT / "per_series.csv.gz", index=False)
    s = summary(d)
    s.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
