"""M4 OWA scored on the first forecast step only, the first three, and all.

Tests whether the ML/DL forecasters fall behind because recursive multi-step
forecasting compounds their errors: if so, they should be close to the
statistical methods at step 1 and drift away over the horizon. Step 1 of each
stored M4 forecast is the one-step forecast from the end of the training series.

    python scripts/m4_by_steps.py results/aeon-8585c7bf/m4
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "equivalence"))
from m4_score import FREQS, M4_HOME, PERIOD, read_results, read_tsf_train, read_wide, scores  # noqa: E402

RESULTS = Path("/local/ajb2u23/Results/Forecasting")
DIRS = ["aeon-8585c7bf", "aeon-8585c7bf-dl"]
METHODS = ["SCUM", "DOTM", "AutoCES", "Theta", "AutoETS", "AutoARIMA", "CES", "ARIMA",
           "ETS", "NaiveForecaster", "RandF", "TVP", "DeepARDiff", "NBeatsDiff",
           "DrCIFDeltaY", "TCNDiff", "TAR", "AutoTAR", "SETAR", "SETARTree", "SETARForest"]
STEPS = {"1": 1, "1-3": 3, "all": None}


def main() -> None:
    out = Path(sys.argv[1])
    runs = {}
    for m in METHODS:
        for d in DIRS:
            f = RESULTS / d / "aeon" / m / f"M4_{m}_Test.csv"
            if f.exists():
                runs[m] = read_results(f)
                break
    naive2 = read_wide(M4_HOME / "official" / "submission-Naive2.csv")
    data = {}
    for freq in FREQS:
        train = read_tsf_train(freq)
        ids = list(train)
        test = read_wide(M4_HOME / "official" / f"{freq.capitalize()}-test.csv")
        m = PERIOD[freq]
        scale = np.array([np.mean(np.abs(train[i][m:] - train[i][:-m])) for i in ids])
        fc = {k: r.loc[ids][[f"pred_{j + 1}" for j in range(test.shape[1])]].to_numpy(float)
              for k, r in runs.items()}
        fc["Naive2"] = naive2.loc[ids].iloc[:, : test.shape[1]].to_numpy(float)
        data[freq] = (test.loc[ids].to_numpy(float), scale, fc)
    table = {}
    for label, k in STEPS.items():
        rows = []
        for freq, (actual, scale, fc) in data.items():
            kk = k or actual.shape[1]
            for name, pred in fc.items():
                s, q = scores(pred[:, :kk], actual[:, :kk], scale)
                rows.append(pd.DataFrame({"frequency": freq, "method": name,
                                          "smape": s, "mase": q}))
        ps = pd.concat(rows)
        for freq in FREQS + ["all"]:
            d = ps if freq == "all" else ps[ps.frequency == freq]
            g = d.groupby("method")[["smape", "mase"]].mean()
            table[(label, freq)] = (g.smape / g.smape["Naive2"]
                                    + g.mase / g.mase["Naive2"]) / 2
    res = pd.DataFrame(table).T
    res.index.names = ["steps", "frequency"]
    res = res[[m for m in METHODS if m in res]]
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "owa_by_steps.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res.T.round(3).to_string())


if __name__ == "__main__":
    main()
