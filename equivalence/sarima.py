"""Equivalence of aeon's SARIMA with reference implementations, order given.

Follows the protocol of the paper's Table on Theta, DOTM and AutoCES: per-series
MASE ratio aeon / reference, TOST on the log ratios with a 5% margin
(``horizon.sweep._tost``), and a paired Wilcoxon signed-rank test, Holm
corrected. Every toolkit is given the same order, so this tests estimation and
forecasting, not model selection. aeon has no automatic SARIMA.

Automatic selection is tested on the simulated series only, ``aeon auto``
(AutoSARIMA) against R ``forecast::auto.arima`` and statsforecast ``AutoARIMA``.
M4 selection is compared with the published forecasts by ``m4.py``.

Two data sets:

- simulated: 12 SARIMA processes x 100 series x lengths 100 and 500;
- M4: all hourly series and a sample of monthly and quarterly series, fitted with
  the airline model (0,1,1)(0,1,1)m and with (1,1,1)(0,1,1)m.

References: R ``stats::arima`` with ``method="CSS"`` (the same objective as aeon)
and ``"CSS-ML"`` (R's default), statsforecast ``ARIMA`` with the same two
methods, and statsmodels ``SARIMAX`` (exact likelihood). R runs in
``sarima_refs.R``.

    python equivalence/sarima.py series --out DIR
    python equivalence/sarima.py python --out DIR
    Rscript equivalence/sarima_refs.R DIR/series.csv DIR/r.csv 8
    python equivalence/sarima.py analyse --out DIR
    python equivalence/sarima.py timing --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import lfilter
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from horizon.sweep import _tost  # noqa: E402

# the Dataset directory of the M4-methods repository (Train/, Test/)
M4_HOME = Path(os.environ.get("AEON_M4_DATASET", "C:/Code/M4-methods/Dataset"))
M4_SETS = {"Hourly": (24, 48, None), "Monthly": (12, 18, 1000), "Quarterly": (4, 8, 1000)}
M4_ORDERS = [(0, 1, 1, 0, 1, 1), (1, 1, 1, 0, 1, 1)]
LENGTHS = [100, 500]
REPLICATES = 100
BURN = 500
TIE_TOL = 1e-9

# name, m, (p, d, q, P, D, Q), phi, theta, Phi, Theta, mean
PROCESSES = [
    ("airline_12", 12, (0, 1, 1, 0, 1, 1), [], [-0.4], [], [-0.6], 0.0),
    ("sar_12", 12, (1, 0, 0, 1, 0, 0), [0.5], [], [0.6], [], 10.0),
    ("sma_12", 12, (0, 0, 1, 0, 0, 1), [], [0.5], [], [0.6], 10.0),
    ("arma_sdiff_12", 12, (1, 0, 1, 0, 1, 1), [0.6], [0.3], [], [-0.5], 0.0),
    ("ar2_diff_12", 12, (2, 1, 0, 0, 1, 1), [0.5, -0.3], [], [], [-0.5], 0.0),
    ("sar_diff_12", 12, (1, 1, 1, 1, 1, 0), [0.4], [-0.6], [-0.4], [], 0.0),
    ("airline_4", 4, (0, 1, 1, 0, 1, 1), [], [-0.5], [], [-0.5], 0.0),
    ("sar_4", 4, (1, 0, 0, 1, 0, 0), [-0.5], [], [0.7], [], 10.0),
    ("arima_sdiff_4", 4, (1, 1, 1, 0, 1, 1), [0.5], [-0.3], [], [-0.6], 0.0),
    ("ma_sar_4", 4, (0, 0, 1, 1, 0, 0), [], [0.6], [0.5], [], 10.0),
    ("arma_sdiff_7", 7, (1, 0, 1, 0, 1, 1), [0.7], [-0.2], [], [-0.7], 0.0),
    ("airline_24", 24, (0, 1, 1, 0, 1, 1), [], [-0.5], [], [-0.7], 0.0),
]


def _poly(coefs, seasonal, m, sign):
    """Coefficients of (1 + sign sum(coefs B^i))(1 + sign sum(seasonal B^(i m)))."""
    ns = np.r_[1.0, sign * np.asarray(coefs, dtype=float)]
    s = np.zeros(len(seasonal) * m + 1)
    s[0] = 1.0
    for j, c in enumerate(seasonal):
        s[(j + 1) * m] = sign * c
    return np.convolve(ns, s)


def simulate(process, n, seed):
    """Sample ``n`` observations of a process, differencing undone."""
    _, m, (_, d, _, _, D, _), phi, theta, Phi, Theta, mean = process
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + BURN)
    w = lfilter(_poly(theta, Theta, m, 1), _poly(phi, Phi, m, -1), e)[BURN:]
    for _ in range(D):
        w = lfilter([1.0], np.r_[1.0, np.zeros(m - 1), -1.0], w)
    for _ in range(d):
        w = np.cumsum(w)
    return w + mean


def _row(group, process, ident, y, h, m, order):
    p, d, q, P, D, Q = order
    return {
        "id": ident, "group": group, "process": process, "n": len(y) - h, "h": h,
        "m": m, "p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q,
        "constant": int(d + D == 0),
        "y": ";".join(f"{v:.10g}" for v in y),
    }


def build_series(out: Path, replicates: int, m4_limit: int | None) -> None:
    rows = []
    for i, proc in enumerate(PROCESSES):
        name, m, order = proc[:3]
        h = max(12, 2 * m)
        for n in LENGTHS:
            for r in range(replicates):
                seed = i * 1_000_000 + n * 1_000 + r
                y = simulate(proc, n + h, seed)
                rows.append(_row("simulated", name, f"{name}_{n}_{r}", y, h, m, order))
    rng = np.random.default_rng(0)
    for name, (m, h, sample) in M4_SETS.items():
        train = pd.read_csv(M4_HOME / "Train" / f"{name}-train.csv", index_col=0)
        test = pd.read_csv(M4_HOME / "Test" / f"{name}-test.csv", index_col=0)
        ids = train.index.to_numpy()
        if m4_limit is not None:
            sample = m4_limit if sample is None else min(sample, m4_limit)
        if sample is not None:
            ids = rng.choice(ids, size=sample, replace=False)
        for sid in ids:
            y = np.r_[train.loc[sid].dropna().to_numpy(float),
                      test.loc[sid].dropna().to_numpy(float)]
            for order in M4_ORDERS:
                tag = "".join(map(str, order))
                rows.append(_row(f"M4 {name}", f"{name}_{tag}", f"{sid}_{tag}", y, h,
                                 m, order))
    pd.DataFrame(rows).to_csv(out / "series.csv", index=False)
    print(len(rows), "series")


def _fit_aeon(row, y):
    from aeon.forecasting.stats import SARIMA

    f = SARIMA(row.p, row.d, row.q, row.P, row.D, row.Q, int(row.m),
               use_constant=bool(row.constant))
    return f.iterative_forecast(y, int(row.h)), f.aic_


def _fit_statsforecast(row, y, method):
    from statsforecast.models import ARIMA

    f = ARIMA(order=(row.p, row.d, row.q), season_length=int(row.m),
              seasonal_order=(row.P, row.D, row.Q),
              include_mean=bool(row.constant), include_drift=False, method=method)
    return f.forecast(y, int(row.h))["mean"], np.nan


def _fit_aeon_auto(row, y):
    from aeon.forecasting.stats import AutoSARIMA

    f = AutoSARIMA(seasonal_period=int(row.m))
    return f.iterative_forecast(y, int(row.h)), f.fit_aic_


def _fit_statsforecast_auto(row, y):
    from statsforecast.models import AutoARIMA

    return AutoARIMA(season_length=int(row.m)).forecast(y, int(row.h))["mean"], np.nan


def _fit_statsmodels(row, y):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    fit = SARIMAX(y, order=(row.p, row.d, row.q),
                  seasonal_order=(row.P, row.D, row.Q, int(row.m)),
                  trend="c" if row.constant else "n").fit(disp=False)
    return fit.forecast(int(row.h)), np.nan


TOOLKITS = {
    "aeon": _fit_aeon,
    "statsforecast CSS": lambda r, y: _fit_statsforecast(r, y, "CSS"),
    "statsforecast CSS-ML": lambda r, y: _fit_statsforecast(r, y, "CSS-ML"),
    "statsmodels": _fit_statsmodels,
    "aeon auto": _fit_aeon_auto,
    "statsforecast auto": _fit_statsforecast_auto,
}
# selection is tested on simulated series; M4 is in m4.py
AUTO_TOOLKITS = {"aeon auto", "statsforecast auto"}
AUTO_PAIRS = ["R auto.arima", "statsforecast auto"]


def _run_one(row, toolkit):
    import warnings

    warnings.filterwarnings("ignore")
    values = np.array(row.y.split(";"), dtype=float)
    y = values[: -int(row.h)]
    start = time.perf_counter()
    try:
        fc, aic = TOOLKITS[toolkit](row, y)
        fc, error = np.asarray(fc, dtype=float), ""
        if not np.all(np.isfinite(fc)):
            error = "non-finite forecast"
    except Exception as e:  # recorded, a failure is a result
        fc, aic, error = np.full(int(row.h), np.nan), np.nan, repr(e)[:200]
    return {"id": row.id, "toolkit": toolkit, "aic": aic,
            "time": time.perf_counter() - start, "error": error,
            "fc": ";".join(f"{v:.10g}" for v in fc)}


def run_python(out: Path, jobs: int) -> None:
    series = pd.read_csv(out / "series.csv")
    tasks = [
        (row, t) for row in series.itertuples() for t in TOOLKITS
        if t not in AUTO_TOOLKITS or row.group == "simulated"
    ]
    rows = Parallel(n_jobs=jobs, batch_size=8, verbose=5)(
        delayed(_run_one)(row, t) for row, t in tasks
    )
    pd.DataFrame(rows).to_csv(out / "python.csv", index=False)


def _errors(row, fc):
    values = np.array(row.y.split(";"), dtype=float)
    h, m = int(row.h), int(row.m)
    train, test = values[:-h], values[-h:]
    scale = np.mean(np.abs(train[m:] - train[:-m]))
    return (np.mean(np.abs(test - fc)) / scale,
            200 * np.mean(np.abs(test - fc) / (np.abs(test) + np.abs(fc))))


def _at_r_params(row, coef):
    """aeon's objective at R's CSS estimates, in aeon's parametrisation."""
    from aeon.forecasting.stats._sarima import _difference, _get_active_lags, _get_diff_poly
    from aeon.forecasting.utils._loss_functions import _sarima_model

    coef = np.array(str(coef).split(";"), dtype=float)
    p, q, P, Q = row.p, row.q, row.P, row.Q
    phi, theta = coef[:p], coef[p : p + q]
    Phi, Theta = coef[p + q : p + q + P], coef[p + q + P : p + q + P + Q]
    params = np.r_[phi, theta, Phi, Theta]
    if row.constant:
        # R's mean mu, aeon's c = mu (1 - sum(phi))(1 - sum(Phi))
        c = coef[-1] * (1 - phi.sum()) * (1 - Phi.sum())
        params = np.r_[c, params]
    ar, ma = _get_active_lags(p, q, P, Q, int(row.m))
    model = np.array([row.constant, p, q, P, Q, row.m, len(ar), len(ma), *ar, *ma],
                     dtype=np.int32)
    y = np.array(row.y.split(";"), dtype=float)[: -int(row.h)]
    data = _difference(y, _get_diff_poly(row.d, row.D, int(row.m)))
    return _sarima_model(params, data, model)[0]


def _signed_rank(ours, ref):
    d = ours - ref
    nz = d[np.abs(d) > TIE_TOL]
    if nz.size == 0:
        return 1.0, 0.0
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    pos, neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return float(wilcoxon(nz).pvalue), float((pos - neg) / (pos + neg))


def _compare(block, ref, margin, alpha, ours="aeon"):
    pair = block[[f"mase_{ours}", f"mase_{ref}"]].dropna()
    ours, theirs = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
    ok = (ours > 0) & (theirs > 0)
    t = _tost(np.log(ours[ok] / theirs[ok]), margin, alpha)
    if t["p_tost"] < alpha:
        verdict = "equivalent"
    elif t["ratio_ci_high"] < 1 / (1 + margin):
        verdict = "aeon better"
    elif t["ratio_ci_low"] > 1 + margin:
        verdict = "aeon worse"
    else:
        verdict = "inconclusive"
    p, r = _signed_rank(ours, theirs)
    return {**t, "verdict": verdict, "p_wilcoxon": p, "rank_biserial": r,
            "aeon_mean_mase": ours.mean(), "ref_mean_mase": theirs.mean(),
            "dropped": len(block) - len(pair)}


def analyse(out: Path, margin: float, alpha: float) -> None:
    series = pd.read_csv(out / "series.csv").set_index("id")
    fc = pd.concat([pd.read_csv(out / "python.csv"), pd.read_csv(out / "r.csv")])
    fc["error"] = fc["error"].fillna("")
    rows = []
    for rec in fc.itertuples():
        row = series.loc[rec.id]
        mase = smape = np.nan
        if rec.error == "":
            mase, smape = _errors(row, np.array(str(rec.fc).split(";"), dtype=float))
        rows.append({"id": rec.id, "toolkit": rec.toolkit, "mase": mase,
                     "smape": smape, "time": rec.time, "error": rec.error})
    scored = pd.DataFrame(rows)
    scored.to_csv(out / "per_series.csv.gz", index=False)

    failures = scored.assign(failed=scored.error != "").join(
        series[["group"]], on="id"
    ).groupby(["group", "toolkit"]).failed.sum().unstack(fill_value=0)
    failures.to_csv(out / "failures.csv")

    wide = scored.pivot(index="id", columns="toolkit", values="mase")
    wide.columns = [f"mase_{c}" for c in wide.columns]
    wide = wide.join(series[["group", "process", "n"]])
    wide["band"] = np.where(wide.group == "simulated", wide.n.astype(str), wide.group)
    refs = [c[5:] for c in wide.columns if c.startswith("mase_") and c != "mase_aeon"
            and c[5:] not in AUTO_TOOLKITS and c[5:] not in AUTO_PAIRS]
    res = []
    for ref in refs:
        for band, g in wide.groupby("band"):
            res.append({"scope": "pooled", "band": band, "process": "all",
                        "reference": ref, **_compare(g, ref, margin, alpha)})
            for proc, gg in g.groupby("process"):
                res.append({"scope": "per process", "band": band, "process": proc,
                            "reference": ref, **_compare(gg, ref, margin, alpha)})
    for ref in [r for r in AUTO_PAIRS if f"mase_{r}" in wide.columns]:
        for band, g in wide[wide.group == "simulated"].groupby("band"):
            res.append({"scope": "pooled", "band": f"auto {band}", "process": "all",
                        "reference": ref,
                        **_compare(g, ref, margin, alpha, "aeon auto")})
            for proc, gg in g.groupby("process"):
                res.append({"scope": "per process", "band": f"auto {band}",
                            "process": proc, "reference": ref,
                            **_compare(gg, ref, margin, alpha, "aeon auto")})
    res = pd.DataFrame(res)
    res["p_holm"] = np.nan
    for _, idx in res.groupby(["scope", "band"]).groups.items():
        res.loc[idx, "p_holm"] = multipletests(res.loc[idx, "p_wilcoxon"], method="holm")[1]
    res["difference"] = np.where(
        res.p_holm >= alpha, "no difference",
        np.where(res.rank_biserial < 0, "aeon better", "reference better"),
    )
    res.to_csv(out / "tost.csv", index=False)

    # Does aeon reach at least as low an objective as R's CSS optimiser?
    r_css = pd.read_csv(out / "r.csv").query("toolkit == 'R CSS' and error != error")
    aeon = fc[fc.toolkit == "aeon"].set_index("id")
    obj = []
    for rec in r_css.itertuples():
        row = series.loc[rec.id]
        at_r = _at_r_params(row, rec.coef)
        obj.append({"id": rec.id, "group": row.group, "aeon_aic": aeon.loc[rec.id, "aic"],
                    "aic_at_r": at_r})
    obj = pd.DataFrame(obj)
    obj["diff"] = obj.aeon_aic - obj.aic_at_r
    obj.to_csv(out / "objective.csv.gz", index=False)
    finite = obj[np.isfinite(obj["diff"])]
    summary = finite.groupby("group")["diff"].describe(percentiles=[0.5, 0.95, 0.99])
    summary["share_within_0.01"] = finite.groupby("group")["diff"].apply(
        lambda d: float(np.mean(d <= 0.01))
    )
    summary.to_csv(out / "objective_summary.csv")

    with pd.option_context("display.width", 220, "display.max_rows", 500,
                           "display.float_format", "{:.3f}".format):
        print(res[res.scope == "pooled"][
            ["band", "reference", "n_series", "dropped", "ratio", "ratio_ci_low",
             "ratio_ci_high", "verdict", "difference", "rank_biserial"]
        ].to_string(index=False))
        pp = res[res.scope == "per process"]
        print(pp.groupby(["reference", "verdict"]).size().unstack(fill_value=0))
        print(pp.groupby(["reference", "difference"]).size().unstack(fill_value=0))
        print(pp[pp.verdict != "equivalent"][
            ["band", "process", "reference", "ratio", "ratio_ci_low", "ratio_ci_high",
             "verdict", "difference"]
        ].to_string(index=False))
        print(failures)
        print(summary)


def timing(out: Path, repeats: int = 5) -> None:
    """Fit and forecast one airline series, one thread, JIT excluded."""
    rows = []
    proc = PROCESSES[0]
    for n in [100, 1000, 10000]:
        y = simulate(proc, n, 42 + n)
        row = pd.Series(dict(p=0, d=1, q=1, P=0, D=1, Q=1, m=12, h=12, constant=0))
        row = argparse.Namespace(**row.to_dict())
        for name, fn in TOOLKITS.items():
            fn(row, y[:50 + proc[1] * 3])  # JIT and imports
            times = []
            for _ in range(1 if n == 10000 and name == "statsmodels" else repeats):
                start = time.perf_counter()
                fn(row, y)
                times.append(time.perf_counter() - start)
            rows.append({"toolkit": name, "n": n, "median_s": np.median(times)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "timing_python.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["series", "python", "analyse", "timing"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--m4-limit", type=int, default=None,
                        help="series per M4 frequency, for a smoke test")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "series":
        import aeon

        repo = str(Path(aeon.__file__).parents[1])
        git = ["git", "-C", repo]
        (args.out / "run_info.json").write_text(json.dumps({
            "aeon_head": subprocess.run(git + ["rev-parse", "HEAD"],
                                        capture_output=True, text=True).stdout.strip(),
            "uncommitted_changes": bool(subprocess.run(
                git + ["status", "--porcelain"], capture_output=True, text=True
            ).stdout.strip()),
        }, indent=2))
        build_series(args.out, args.replicates, args.m4_limit)
    elif args.stage == "python":
        run_python(args.out, args.jobs)
    elif args.stage == "analyse":
        analyse(args.out, args.margin, args.alpha)
    else:
        timing(args.out)


if __name__ == "__main__":
    main()
