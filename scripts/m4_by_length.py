"""M4 accuracy of the ML/DL forecasters against series length.

Bins the 100,000 M4 series by training length and gives, per bin, the OWA of
each method (relative to Naive2, as in M4) and, as a measure robust to the few
exploding forecasts, the median per-series ratio of the method's MASE to
AutoETS's MASE. Bins are overall and within each frequency, since length and
frequency are confounded (yearly series are short, hourly long).

    python scripts/m4_by_length.py /local/.../aeon-8585c7bf \\
        /local/.../aeon-8585c7bf-dl results/aeon-8585c7bf/m4
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "equivalence"))
from m4_score import FREQS, M4_HOME, PERIOD, read_results, read_tsf_train, read_wide, scores  # noqa: E402

METHODS = ["AutoETS", "Theta", "SCUM", "RandF", "DrCIFDeltaY", "DeepARDiff",
           "NBeatsDiff", "TCNDiff"]
BINS = [0, 30, 50, 100, 200, 400, 1000, np.inf]
LABELS = ["<30", "30-49", "50-99", "100-199", "200-399", "400-999", "1000+"]


def main() -> None:
    *dirs, out = map(Path, sys.argv[1:])
    runs = {}
    for d in dirs:
        for m in METHODS:
            f = d / "aeon" / m / f"M4_{m}_Test.csv"
            if f.exists() and m not in runs:
                runs[m] = read_results(f)
    naive2 = read_wide(M4_HOME / "official" / "submission-Naive2.csv")
    rows = []
    for freq in FREQS:
        train = read_tsf_train(freq)
        ids = list(train)
        test = read_wide(M4_HOME / "official" / f"{freq.capitalize()}-test.csv")
        h = test.shape[1]
        act = test.loc[ids].to_numpy(float)
        m = PERIOD[freq]
        sc = np.array([np.mean(np.abs(train[i][m:] - train[i][:-m])) for i in ids])
        n = np.array([train[i].size for i in ids])
        fc = {k: r.loc[ids][[f"pred_{j + 1}" for j in range(h)]].to_numpy(float)
              for k, r in runs.items()}
        fc["Naive2"] = naive2.loc[ids].iloc[:, :h].to_numpy(float)
        for k, p in fc.items():
            s, q = scores(p, act, sc)
            rows.append(pd.DataFrame({"id": ids, "frequency": freq, "n": n,
                                      "method": k, "smape": s, "mase": q}))
    ps = pd.concat(rows, ignore_index=True)
    ps["length"] = pd.cut(ps.n, BINS, labels=LABELS, right=False)
    ets = ps[ps.method == "AutoETS"].set_index("id").mase
    ps["mase_vs_ets"] = ps.mase / ps.id.map(ets)

    def summarise(d):
        g = d.groupby("method")[["smape", "mase"]].mean()
        owa = (g.smape / g.smape["Naive2"] + g.mase / g.mase["Naive2"]) / 2
        ratio = d.groupby("method").mase_vs_ets.median()
        return pd.DataFrame({"series": d.id.nunique(), "owa": owa,
                             "median_mase_vs_autoets": ratio})

    tables = []
    for freq in ["all"] + FREQS:
        d = ps if freq == "all" else ps[ps.frequency == freq]
        for b, db in d.groupby("length", observed=True):
            if db.id.nunique() >= 50:
                tables.append(summarise(db).assign(frequency=freq, length=b))
    res = pd.concat(tables).reset_index().rename(columns={"index": "method"})
    res = res[["frequency", "length", "series", "method", "owa", "median_mase_vs_autoets"]]
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "owa_by_length.csv", index=False)
    for col in ["owa", "median_mase_vs_autoets"]:
        wide = res.pivot_table(index=["frequency", "length", "series"], columns="method",
                               values=col, observed=True)
        print(f"\n{col}")
        print(wide[[m for m in METHODS + ["Naive2"] if m in wide]].round(3).to_string())


if __name__ == "__main__":
    main()
