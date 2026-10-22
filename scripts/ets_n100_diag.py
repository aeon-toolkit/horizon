"""Where aeon AutoETS spends its time at n=100 (aeon 8585c7bf).

For the timing-experiment series (same seeds as ``horizon run --only timing``)
of each ETS process, refit every candidate model that ``auto_ets`` tries and
record, for each of the three Nelder-Mead starts, the number of loss
evaluations, iterations, whether it hit the iteration limit, and the value
reached. The Nelder-Mead loop is replicated in Python around aeon's compiled
loss so the counts are exact; times come from the compiled functions.

    PYTHONPATH=/home/ajb2u23/Code/aeon-bench-next python scripts/ets_n100_diag.py OUT.csv
"""

import sys
import time

import numpy as np
import pandas as pd
from aeon.forecasting.stats import ETS, AutoETS
from aeon.forecasting.utils._nelder_mead import (
    _ets_num_params,
    _ets_r_style_start,
    _ets_simplex_around,
    _fit_ets_parameters,
    _init_ets_simplex,
    _simplex_converged,
    dispatch_loss,
)

from horizon.experiments import _seed, horizon_for
from horizon.simulate import get_processes

N, REPEATS, OFFSET, MAX_ITER = 100, 100, 10_000_000, 500


def nm_counted(points, data, model, tol=1e-6, max_iter=MAX_ITER):
    """aeon's _nelder_mead_from_simplex, counting evaluations and iterations."""
    points = points.copy()
    evals = 0

    def f(x):
        nonlocal evals
        evals += 1
        return dispatch_loss(1, x.copy(), data, model)

    values = np.array([f(p) for p in points])
    it = 0
    for it in range(1, max_iter + 1):
        order = np.argsort(values)
        points, values = points[order], values[order]
        if _simplex_converged(points, values, tol):
            return points[0], values[0], evals, it - 1, False
        c = points[:-1].sum(axis=0) / len(points[:-1])
        r = c + (c - points[-1])
        rv = f(r)
        if len(values) > 1 and not np.isnan(rv) and values[0] <= rv < values[-2]:
            points[-1], values[-1] = r, rv
            continue
        if not np.isnan(rv) and rv < values[0]:
            e = c + 2 * (r - c)
            ev = f(e)
            if not np.isnan(ev) and ev < rv:
                points[-1], values[-1] = e, ev
            else:
                points[-1], values[-1] = r, rv
            continue
        k = c - 0.5 * (c - points[-1])
        kv = f(k)
        if not np.isnan(kv) and kv < values[-1]:
            points[-1], values[-1] = k, kv
            continue
        for i in range(1, len(points)):
            points[i] = points[0] - 0.5 * (points[0] - points[i])
            values[i] = f(points[i])
    b = np.argmin(values)
    return points[b], values[b], evals, max_iter, True


def candidates(y, sp):
    """Models auto_ets tries (restrict=True, additive trends, damped or not)."""
    pos = bool(np.all(y > 0))
    seasonal = sp > 1 and len(y) > 4 * sp
    out = []
    for e in (1, 2):
        if e == 2 and not pos:
            continue
        for t, d in ((0, 0), (1, 0), (1, 1)):
            for s in range(3 if seasonal else 1):
                if s == 2 and not pos:
                    continue
                if e == 1 and s == 2:
                    continue
                out.append(np.array([e, t, s, sp if s else 1, d], dtype=np.int32))
    return out


def label(m):
    t = {0: "N", 1: "A", 2: "M"}[m[1]] + ("d" if m[1] and m[4] else "")
    return f"{'AM'[m[0] - 1]}{t}{'NAM'[m[2]]}"


def main(out):
    all_names = [p.name for p in get_processes()]
    rows = []
    for proc in get_processes("ets"):
        sp = proc.seasonal_period
        pidx = all_names.index(proc.name)
        for r in range(REPEATS):
            y = proc.sample(N, random_state=_seed(pidx, N, r, OFFSET))
            y = np.asarray(y, dtype=np.float64)
            t0 = time.perf_counter()
            f = AutoETS(seasonal_period=sp if sp > 1 else None).fit(y)
            total = time.perf_counter() - t0
            sel = np.array([f.error_type_, f.trend_type_, f.seasonality_type_,
                            f.seasonal_period_, int(f.damped_trend_)], dtype=np.int32)
            t0 = time.perf_counter()
            ETS(f.error_type_, f.trend_type_, f.seasonality_type_,
                f.seasonal_period_, damped_trend=f.damped_trend_).fit(y)
            refit = time.perf_counter() - t0
            for m in candidates(y, sp):
                k = _ets_num_params(m)
                t0 = time.perf_counter()
                _, ref_v = _fit_ets_parameters(y, m, max_iter=MAX_ITER)
                cand_s = time.perf_counter() - t0
                pts = np.empty((k + 1, k))
                _init_ets_simplex(pts, m)
                starts = [pts, _ets_simplex_around(_ets_r_style_start(m), m)]
                res = []
                for st in starts:
                    res.append(nm_counted(st, y, m))
                best = min(res, key=lambda z: z[1])
                res.append(nm_counted(_ets_simplex_around(best[0], m), y, m))
                for i, (x, v, ev, it, cap) in enumerate(res):
                    rows.append(dict(
                        process=proc.name, repeat=r, total_s=total, refit_s=refit,
                        selected=label(sel), candidate=label(m), n_params=k,
                        candidate_s=cand_s, start=["default", "r_style", "restart"][i],
                        evals=ev, iters=it, hit_limit=cap, value=v, aeon_value=ref_v))
        print(proc.name, "done", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main(sys.argv[1])
