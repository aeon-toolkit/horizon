"""M4 hourly: how accuracy changes over the 48-step horizon.

For each of the 414 hourly series and each forecast step h = 1..48, the sMAPE
and MASE of every stored M4 forecast (aeon runs at 8585c7bf, the deep learners,
the published submissions Naive2, ETS, ARIMA, Theta and SCUM = entry 036) and
of no-skill baselines computed here from the training series:

- ``Naive1``: last value;
- ``SeasonalNaive``: the last 24 values repeated;
- ``Mean28d``: the mean of the last 28 days (a constant: no skill at all);
- ``Profile28d``: the mean of each hour of the day over the last 28 days
  (knows the daily cycle and nothing about the present);
- ``Random28d``: a value drawn at random from the last 28 days, the expected
  error over 50 draws ("random guessing" from the series' own range).

MASE is scaled as in M4 (in-sample seasonal naive error, period 24). Outputs in
``results/aeon-8585c7bf/m4-hourly-horizon/``:

- ``by_step.csv``: per method and step, mean sMAPE, mean MASE, OWA against
  Naive2 at that step, and the share of series better than Naive2 / than
  Profile28d at that step (by absolute error);
- ``by_block.csv``: the same over blocks of steps (1-6, 7-12, 13-24, 25-36, 37-48),
  plus ``gm_rel_mae``: geometric mean over series of the MAE over the block
  relative to Naive2's (robust to the few series that dominate mean MASE), the
  same against SeasonalNaive, and the share of series better than SeasonalNaive;
- ``by_step.png``: per-step OWA against Naive2 and MASE for the main methods.

    PYTHONPATH=<aeon worktree> python scripts/m4_hourly_horizon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "equivalence"))
from m4_score import M4_HOME, read_results, read_tsf_train, read_wide  # noqa: E402

RESULTS = Path("/local/ajb2u23/Results/Forecasting")
DIRS = ["aeon-8585c7bf", "aeon-8585c7bf-dl"]
AEON = ["SCUM", "DOTM", "AutoCES", "Theta", "AutoETS", "AutoARIMA", "CES", "ARIMA",
        "ETS", "NaiveForecaster", "RandF", "TVP", "DeepARDiff", "NBeatsDiff",
        "DrCIFDeltaY", "TCNDiff", "TAR", "AutoTAR", "SETAR", "SETARTree", "SETARForest"]
OFFICIAL = {"Naive2": "Naive2", "M4 ETS": "ETS", "M4 ARIMA": "ARIMA",
            "M4 Theta": "Theta", "M4 SCUM (036)": "036"}
H, M, DAYS = 48, 24, 28
BLOCKS = [(1, 6), (7, 12), (13, 24), (25, 36), (37, 48)]
OUT = ROOT / "results" / "aeon-8585c7bf" / "m4-hourly-horizon"
PLOT = ["Naive2", "SeasonalNaive", "Profile28d", "Mean28d", "Random28d", "Naive1",
        "M4 SCUM (036)", "AutoETS", "AutoARIMA", "Theta", "DrCIFDeltaY", "RandF",
        "TCNDiff"]


def _gmean(r: np.ndarray) -> float:
    r = r[np.isfinite(r) & (r > 0)]
    return float(np.exp(np.log(r).mean()))


def baselines(train: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    out = {k: [] for k in ["Naive1", "SeasonalNaive", "Mean28d", "Profile28d"]}
    for y in train.values():
        recent = y[-DAYS * M:]
        out["Naive1"].append(np.full(H, y[-1]))
        out["SeasonalNaive"].append(np.tile(y[-M:], H // M))
        out["Mean28d"].append(np.full(H, recent.mean()))
        # hour of day of each recent point, aligned so the first forecast is hour 0
        k = recent.size
        hour = (np.arange(k) - k) % M
        prof = np.array([recent[hour == j].mean() for j in range(M)])
        out["Profile28d"].append(np.tile(prof, H // M))
    fc = {k: np.array(v) for k, v in out.items()}
    draws = [np.array([rng.choice(y[-DAYS * M:], H) for y in train.values()])
             for _ in range(50)]
    return fc, draws


def main() -> None:
    train = read_tsf_train("hourly")
    ids = list(train)
    actual = read_wide(M4_HOME / "official" / "Hourly-test.csv").loc[ids].to_numpy(float)
    scale = np.array([np.mean(np.abs(y[M:] - y[:-M])) for y in train.values()])
    fc, draws = baselines(train)
    for name, sub in OFFICIAL.items():
        fc[name] = read_wide(M4_HOME / "official" / f"submission-{sub}.csv").loc[ids].iloc[
            :, :H].to_numpy(float)
    for m in AEON:
        for d in DIRS:
            f = RESULTS / d / "aeon" / m / f"M4_{m}_Test.csv"
            if f.exists():
                fc[m] = read_results(f).loc[ids][[f"pred_{j + 1}" for j in range(H)]].to_numpy(float)
                break

    def err(pred):  # per series x step
        ae = np.abs(actual - pred)
        return 200 * ae / (np.abs(actual) + np.abs(pred)), ae / scale[:, None], ae

    E = {k: err(p) for k, p in fc.items()}
    rs = [err(p) for p in draws]  # random guessing: expected error over the draws
    E["Random28d"] = tuple(np.mean([r[i] for r in rs], axis=0) for i in range(3))

    n2s, n2m, n2a = E["Naive2"]
    prof_a = E["Profile28d"][2]
    sn_a = E["SeasonalNaive"][2]
    rows = []
    for k, (s, q, a) in E.items():
        for h in range(H):
            rows.append({
                "method": k, "step": h + 1,
                "smape": np.nanmean(s[:, h]), "mase": np.nanmean(q[:, h]),
                "owa": (np.nanmean(s[:, h]) / np.nanmean(n2s[:, h])
                        + np.nanmean(q[:, h]) / np.nanmean(n2m[:, h])) / 2,
                "better_than_naive2": np.mean(a[:, h] < n2a[:, h]),
                "better_than_profile": np.mean(a[:, h] < prof_a[:, h]),
            })
    by_step = pd.DataFrame(rows)
    brows = []
    for k, (s, q, a) in E.items():
        for lo, hi in BLOCKS:
            sl = slice(lo - 1, hi)
            brows.append({
                "method": k, "steps": f"{lo}-{hi}",
                "smape": np.nanmean(s[:, sl]), "mase": np.nanmean(q[:, sl]),
                "owa": (np.nanmean(s[:, sl]) / np.nanmean(n2s[:, sl])
                        + np.nanmean(q[:, sl]) / np.nanmean(n2m[:, sl])) / 2,
                "better_than_naive2": np.mean(a[:, sl].mean(1) < n2a[:, sl].mean(1)),
                # robust: geometric mean over series of MAE over the block / Naive2's
                "gm_rel_mae": _gmean(a[:, sl].mean(1) / n2a[:, sl].mean(1)),
                "gm_rel_mae_snaive": _gmean(a[:, sl].mean(1) / sn_a[:, sl].mean(1)),
                "better_than_snaive": np.mean(a[:, sl].mean(1) < sn_a[:, sl].mean(1)),
            })
    by_block = pd.DataFrame(brows)
    OUT.mkdir(parents=True, exist_ok=True)
    by_step.to_csv(OUT / "by_step.csv", index=False)
    by_block.to_csv(OUT / "by_block.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in PLOT:
        d = by_step[by_step.method == k]
        ls = "--" if k in {"SeasonalNaive", "Profile28d", "Mean28d", "Random28d", "Naive1"} else "-"
        axes[0].plot(d.step, d.owa, ls, label=k)
        axes[1].plot(d.step, d.mase, ls, label=k)
    axes[0].axhline(1, color="grey", lw=0.8)
    axes[0].set(xlabel="step (hours ahead)", ylabel="OWA vs Naive2 at that step", ylim=(0, 3))
    axes[1].set(xlabel="step (hours ahead)", ylabel="mean MASE", yscale="log")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("M4 hourly (414 series): accuracy by forecast step")
    fig.tight_layout()
    fig.savefig(OUT / "by_step.png", dpi=150)

    pd.set_option("display.width", 250)
    piv = by_block.pivot(index="method", columns="steps", values="owa")[[f"{a}-{b}" for a, b in BLOCKS]]
    print("OWA vs Naive2 by block of steps\n", piv.sort_values("37-48").round(3).to_string())
    piv = by_block.pivot(index="method", columns="steps", values="mase")[[f"{a}-{b}" for a, b in BLOCKS]]
    for col in ["gm_rel_mae", "gm_rel_mae_snaive", "better_than_snaive"]:
        piv = by_block.pivot(index="method", columns="steps", values=col)[[f"{a}-{b}" for a, b in BLOCKS]]
        print(f"\n{col} by block\n", piv.sort_values("37-48").round(3).to_string())
    print("\nmean MASE by block\n", piv.sort_values("37-48").round(3).to_string())


if __name__ == "__main__":
    main()
