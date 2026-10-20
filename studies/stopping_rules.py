"""Nelder-Mead stopping rules on aeon's ARIMA and ETS losses, single runs.

Needs the aeon branch with the multi-start helpers. Compares, from aeon's
starting points:

V0: value spread < tol, checked only after a shrink step (aeon 1.6.0).
V1: value spread < tol, checked every iteration.
V2: V1 and every vertex within a relative 1e-6 of the best.

    python studies/stopping_rules.py <out.csv>
"""

import sys

import numpy as np
import pandas as pd
from numba import njit

from horizon.simulate import ALL_PROCESSES  # noqa: E402

from aeon.forecasting.utils._nelder_mead import (  # noqa: E402
    _arima_simplex_around,
    _ets_r_style_start,
    _ets_simplex_around,
    _hannan_rissanen,
    _init_arima_simplex,
    _init_ets_simplex,
    dispatch_loss,
)


@njit(cache=True)
def converged(points, values, tol, xtol, rule):
    if not np.isfinite(values[0]):
        return False
    if np.max(np.abs(values - values[0])) >= tol:
        return False
    if rule == 2:
        for i in range(1, len(points)):
            for j in range(points.shape[1]):
                if abs(points[i, j] - points[0, j]) > xtol * (1.0 + abs(points[0, j])):
                    return False
    return True


@njit(cache=True)
def nm(loss_id, points, data, model, max_iter, rule, tol=1e-6, xtol=1e-6):
    points = points.copy()
    values = np.empty(len(points))
    evals = 0
    for i in range(len(points)):
        values[i] = dispatch_loss(loss_id, points[i].copy(), data, model)
        evals += 1
    for _ in range(max_iter):
        order = np.argsort(values)
        points = points[order]
        values = values[order]
        if rule > 0 and converged(points, values, tol, xtol, rule):
            break
        c = points[:-1].sum(axis=0) / len(points[:-1])
        r = c + (c - points[-1])
        rv = dispatch_loss(loss_id, r, data, model)
        evals += 1
        if values[0] <= rv < values[-2]:
            points[-1] = r
            values[-1] = rv
            continue
        if rv < values[0]:
            e = c + 2 * (r - c)
            ev = dispatch_loss(loss_id, e, data, model)
            evals += 1
            if ev < rv:
                points[-1] = e
                values[-1] = ev
            else:
                points[-1] = r
                values[-1] = rv
            continue
        k = c - 0.5 * (c - points[-1])
        kv = dispatch_loss(loss_id, k, data, model)
        evals += 1
        if kv < values[-1]:
            points[-1] = k
            values[-1] = kv
            continue
        for i in range(1, len(points)):
            points[i] = points[0] - 0.5 * (points[0] - points[i])
            values[i] = dispatch_loss(loss_id, points[i], data, model)
            evals += 1
        if rule == 0 and np.isfinite(values[0]) and np.max(np.abs(values - values[0])) < tol:
            break
    return values.min(), evals


rows = []
for name in ["arma_11", "ar_2", "ma_2", "white_noise", "arima_110", "arima_011_drift"]:
    d = 1 if name in ("arima_110", "arima_011_drift") else 0
    for n in (100, 500, 2000):
        for r in range(8):
            y = ALL_PROCESSES[name].sample(n, 9000 + r)
            x = np.diff(y, n=d) if d else y
            for p, q in [(0, 0), (1, 1), (2, 0), (0, 2), (2, 2), (3, 2)]:
                model = np.array([1, p, q], dtype=np.int32)
                k = 1 + p + q
                pts = np.empty((k + 1, k))
                _init_arima_simplex(pts, x, model, 0.1 / (k + 1))
                starts = [("mean", pts)]
                hr = _hannan_rissanen(x, model)
                if np.all(np.isfinite(hr)):
                    starts.append(("hr", _arima_simplex_around(hr, x, model)))
                for sname, s in starts:
                    for rule in (0, 1, 2):
                        v, ev = nm(0, s, x, model, 500, rule)
                        rows.append(dict(family="arima", process=name, n=n, rep=r, model=f"{p}{q}",
                                         start=sname, rule=rule, loss=v, evals=ev))
for name in ["ets_ann", "ets_aan", "ets_adn", "ets_ana_m4", "ets_aaa_m12", "ets_mam_m12"]:
    sp = ALL_PROCESSES[name].seasonal_period
    for n in (100, 500, 2000):
        for r in range(8):
            y = ALL_PROCESSES[name].sample(n, 9000 + r)
            for e in (1, 2):
                for t, damped in ((0, 0), (1, 0), (1, 1)):
                    for s in ((0, 1, 2) if sp > 1 else (0,)):
                        model = np.array([e, t, s, sp if s else 1, damped], dtype=np.int32)
                        k = 1 + (t != 0) * (1 + damped) + (s != 0)
                        pts = np.empty((k + 1, k))
                        _init_ets_simplex(pts, model)
                        for sname, st in (("stock", pts), ("R", _ets_simplex_around(_ets_r_style_start(model), model))):
                            for rule in (0, 1, 2):
                                v, ev = nm(1, st, y, model, 500, rule)
                                rows.append(dict(family="ets", process=name, n=n, rep=r, model=f"{e}{t}{damped}{s}",
                                                 start=sname, rule=rule, loss=v, evals=ev))

df = pd.DataFrame(rows)
df.to_csv(sys.argv[1], index=False)
key = ["family", "process", "n", "rep", "model", "start"]
w = df.pivot_table(index=key, columns="rule", values=["loss", "evals"])
w = w[np.isfinite(w[("loss", 0)])]
for fam, g in w.groupby(level="family"):
    print(f"== {fam}: {len(g)} runs")
    for rule in (1, 2):
        dl = g[("loss", rule)] - g[("loss", 0)]
        saved = 1 - g[("evals", rule)].sum() / g[("evals", 0)].sum()
        print(f"  V{rule}: evals saved {saved:6.1%} | loss worse than V0 by >0.01: {(dl > 0.01).mean():6.2%}, "
              f">1: {(dl > 1).mean():6.2%} | better by >0.01: {(dl < -0.01).mean():6.2%} | max worse {dl.max():.3f}")
    for n, gn in g.groupby(level="n"):
        print(f"    n={n}: V0 evals/run {gn[('evals', 0)].mean():.0f}, V1 {gn[('evals', 1)].mean():.0f}, V2 {gn[('evals', 2)].mean():.0f}")
