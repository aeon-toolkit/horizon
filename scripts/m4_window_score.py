"""Score the M4 training-window experiment (``m4_window.py``).

Per forecaster and method, sMAPE and MASE of the test forecast as in the M4
evaluation, and OWA relative to the published Naive2 on the same series: by
frequency, by training-length band, and overall. Methods:

- ``full``: the whole training series (the paper's setting);
- ``w=<k>``: the last k points (the whole series where it is shorter);
- ``validation``: per series, the window with the lowest MAE on the M4 holdout
  (the last h training points, forecast from a fit that excludes them);
- ``global``: one window per frequency, the one with the lowest mean holdout
  sMAPE + MASE over that frequency's series;
- ``average``: the mean forecast over the windows fitted (each counted once);
- ``median``: as ``average``, with the median at each step;
- ``oracle``: per series, the window with the lowest test MAE (an upper bound,
  not a method).

Also, per method, the number of series with lower / higher test MASE than the
whole series, and a Wilcoxon signed-rank p value on the log MASE ratio.

    python scripts/m4_window_score.py \\
        --results /local/ajb2u23/Results/Forecasting/m4-window-8585c7bf \\
        --out results/aeon-8585c7bf/m4-window
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "equivalence"))
from m4_score import FREQS, M4_HOME, PERIOD, read_tsf_train, read_wide  # noqa: E402

BANDS = [(500, 1000), (1000, 2000), (2000, 10**6)]


def band(n: int) -> str:
    for lo, hi in BANDS:
        if lo <= n < hi:
            return f"{lo}-{hi - 1}" if hi < 10**6 else f"{lo}+"
    return "<500"


def load_truth(ids_by_freq: dict[str, list[str]]):
    """Test values, holdout values, MASE scale, Naive2 forecast and length per id."""
    naive2 = read_wide(M4_HOME / "official" / "submission-Naive2.csv")
    out = {}
    for freq, ids in ids_by_freq.items():
        train = read_tsf_train(freq)
        test = read_wide(M4_HOME / "official" / f"{freq.capitalize()}-test.csv")
        h, m = test.shape[1], PERIOD[freq]
        for i in ids:
            y = train[i]
            out[i] = dict(freq=freq, n=y.size, test=test.loc[i].to_numpy(float),
                          holdout=y[-h:], naive2=naive2.loc[i].to_numpy(float)[:h],
                          scale=np.mean(np.abs(y[m:] - y[:-m])))
    return out


def smape_mase(pred, actual, scale):
    smape = np.mean(200 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred)))
    return smape, np.mean(np.abs(actual - pred)) / scale


def score(rows: pd.DataFrame, truth: dict) -> pd.DataFrame:
    """Per-series sMAPE and MASE for every method."""
    ok = rows[rows.status == "ok"].copy()
    ok["p"] = ok.pred.map(lambda s: np.array(json.loads(s), dtype=float))
    out = []
    holdout_err = {}
    for i, d in ok.groupby("id"):
        t = truth[i]
        test = d[d.split == "test"].set_index("window").p
        val = d[d.split == "val"].set_index("window").p
        if 0 not in test.index:
            continue
        windows = sorted(test.index)
        mae = {w: np.mean(np.abs(test[w] - t["test"])) for w in windows}
        vmae = {w: np.mean(np.abs(val[w] - t["holdout"])) for w in val.index}
        vsm = {w: smape_mase(val[w], t["holdout"], t["scale"]) for w in val.index}
        holdout_err[i] = vsm
        order = [0] + sorted((w for w in windows if w), reverse=True)  # ties: longer
        methods = {"full": test[0]}
        for w in windows[1:]:
            methods[f"w={w}"] = test[w]
        cand = [w for w in order if w in vmae and w in test.index]
        methods["validation"] = test[min(cand, key=lambda w: vmae[w])] if cand else test[0]
        methods["average"] = np.mean(np.stack(list(test.values)), axis=0)
        methods["median"] = np.median(np.stack(list(test.values)), axis=0)
        methods["oracle"] = test[min(order, key=lambda w: mae[w])]
        for name, pred in methods.items():
            s, q = smape_mase(pred, t["test"], t["scale"])
            out.append(dict(id=i, frequency=t["freq"], n=t["n"], method=name, smape=s, mase=q))
        s2, q2 = smape_mase(t["naive2"], t["test"], t["scale"])
        out.append(dict(id=i, frequency=t["freq"], n=t["n"], method="Naive2", smape=s2, mase=q2))
    per = pd.DataFrame(out)
    # global: one window per frequency, chosen on the holdout
    glob = []
    for freq, d in per[per.method == "full"].groupby("frequency"):
        ids = list(d.id)
        cands = sorted({w for i in ids for w in holdout_err[i]})
        def val_score(w):
            s = [holdout_err[i].get(w, holdout_err[i][0]) for i in ids]
            return np.mean([a for a, _ in s]) + np.mean([b for _, b in s])
        w = min(cands, key=val_score)
        g = per[(per.frequency == freq) & (per.method == (f"w={w}" if w else "full"))]
        # series shorter than w use the whole series
        missing = set(ids) - set(g.id)
        g = pd.concat([g, per[(per.id.isin(missing)) & (per.method == "full")]])
        glob.append(g.assign(method="global", global_window=w))
    per = pd.concat([per, *glob], ignore_index=True)
    # fixed windows longer than a series: the whole series
    full = per[per.method == "full"].set_index("id")
    fixed = [m for m in per.method.unique() if m.startswith("w=")]
    fill = []
    for m in fixed:
        have = set(per[per.method == m].id)
        miss = full.loc[~full.index.isin(have)].reset_index().assign(method=m)
        fill.append(miss)
    return pd.concat([per, *fill], ignore_index=True)


def owa_table(per: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    groups = list(per.groupby(by)) + [("all", per)]
    for key, d in groups:
        g = d.groupby("method")[["smape", "mase"]].mean()
        base = g.loc["Naive2"]
        owa = (g.smape / base.smape + g.mase / base.mase) / 2
        rows.append(owa.rename(key))
    return pd.DataFrame(rows).T


def versus_full(per: pd.DataFrame) -> pd.DataFrame:
    """Per-series MASE against the whole series: better, worse, Wilcoxon p."""
    from scipy.stats import wilcoxon

    wide = per.pivot_table(index="id", columns="method", values="mase")
    out = {}
    for m in wide.columns:
        if m in ("full", "Naive2"):
            continue
        d = np.log(wide[m]) - np.log(wide["full"])
        d = d[np.isfinite(d)]
        p = wilcoxon(d).pvalue if (d != 0).any() else np.nan
        out[m] = dict(better=int((d < 0).sum()), worse=int((d > 0).sum()), p_value=p)
    return pd.DataFrame(out).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.results.glob("*.csv"))
    first = pd.read_csv(files[0], usecols=["id", "frequency"]).drop_duplicates()
    truth = load_truth(first.groupby("frequency").id.apply(list).to_dict())
    per_all, tables = [], []
    for path in files:
        rows = pd.read_csv(path)
        f = rows.forecaster.iloc[0]
        bad = rows[rows.status != "ok"]
        if len(bad):
            print(f"{f}: {len(bad)} fits not ok")
        per = score(rows, truth).assign(forecaster=f)
        per["band"] = per.n.map(band)
        per_all.append(per)
        fr = owa_table(per, "frequency").reindex(columns=[q for q in FREQS if q in set(per.frequency)] + ["all"])
        bd = owa_table(per, "band").drop(columns="all")
        t = fr.join(bd).join(versus_full(per)).assign(forecaster=f)
        t = t.rename_axis("method").reset_index()
        tables.append(t)
        order = (["full"] + sorted([m for m in t.method if m.startswith("w=")], key=lambda m: int(m[2:]))
                 + ["validation", "global", "average", "median", "oracle", "Naive2"])
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
            print(f"\n== {f}  (series: {per[per.method == 'full'].groupby('frequency').size().to_dict()})")
            print(t.set_index("method").drop(columns="forecaster").reindex(order).to_string())
    pd.concat(per_all).to_csv(args.out / "per_series.csv.gz", index=False)
    pd.concat(tables).to_csv(args.out / "owa.csv", index=False)


if __name__ == "__main__":
    main()
