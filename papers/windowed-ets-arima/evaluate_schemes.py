"""Score every window-combination scheme on LSSH-99 and the long M4 series.

The built-in schemes of ``scripts/window_combine.py`` plus variants restricted to
windows of at least 300 observations (and the whole series). Reads the forecast
store; writes results/aeon-8585c7bf/window-combine/{lssh99,m4}_*.

    python papers/windowed-ets-arima/evaluate_schemes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from window_combine import combine  # noqa: E402
from window_store import Store, built_in  # noqa: E402

STORE = "/local/ajb2u23/Results/Forecasting/window-store-8585c7bf"
OUT = ROOT / "results" / "aeon-8585c7bf" / "window-combine"
MIN_WINDOW = 300


def restricted(name: str):
    """A built-in scheme over windows >= MIN_WINDOW and the whole series."""
    def scheme(ctx):
        keep = (ctx["windows"] == 0) | (ctx["windows"] >= MIN_WINDOW)
        oos = ctx.get("oos_mse")
        return combine(ctx["preds"][keep], ctx["insample_mse"][keep],
                       ctx["insample_rel_mae"][keep], None if oos is None else oos[keep])[name]
    return scheme


def schemes():
    s = built_in()
    for n in ["mean", "median", "inv_mse", "inv_oos_mse"]:
        s[f"{n}_min300"] = restricted(n)
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    store = Store(STORE)
    per, summ = store.evaluate_lssh99(schemes())
    per.to_csv(OUT / "lssh99_per_problem.csv", index=False)
    summ.to_csv(OUT / "lssh99_summary.csv", index=False)
    per, tab = store.evaluate_m4(schemes())
    per.to_csv(OUT / "m4_per_series.csv.gz", index=False)
    tab.to_csv(OUT / "m4_owa.csv", index=False)
    import pandas as pd
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(summ.pivot(index="method", columns="forecaster", values="all").sort_values("AutoETS").to_string())
    print(tab.pivot(index="method", columns="forecaster", values="all").sort_values("AutoETS").to_string())


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
