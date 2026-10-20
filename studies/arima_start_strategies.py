"""Nelder-Mead starting strategies for ARIMA on aeon's CSS loss.

Works with any aeon (uses only ``_arima_fit``). Copies aeon's Nelder-Mead loop
and runs it from different starting simplices: aeon 1.6.0's default, the
constant at the series mean, Hannan-Rissanen estimates, and combinations with
a restart. The gap is the loss reached minus the best loss from a scipy
multi-start. Note that the CSS loss in aeon 1.6.0 is unconstrained, so the
reference optimum can be non-invertible.

    python studies/arima_start_strategies.py <out.csv>
"""

import sys
import warnings

warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

from aeon.datasets import load_airline
from aeon.forecasting.utils._loss_functions import _arima_fit

from horizon.simulate import ALL_PROCESSES  # noqa: E402


@njit(cache=True)
def nm_from(points, data, model, max_iter, tol=1e-6):
    """aeon's nelder_mead loop, verbatim, from a supplied starting simplex."""
    points = points.copy()
    values = np.empty(len(points))
    for i in range(len(points)):
        v = _arima_fit(points[i].copy(), data, model)
        values[i] = v if np.isfinite(v) else np.inf
    for _ in range(max_iter):
        order = np.argsort(values)
        points = points[order]
        values = values[order]
        centre = points[:-1].sum(axis=0) / len(points[:-1])
        refl = centre + (centre - points[-1])
        rv = _arima_fit(refl, data, model)
        rv = rv if np.isfinite(rv) else np.inf
        if values[0] <= rv < values[-2]:
            points[-1] = refl
            values[-1] = rv
            continue
        if rv < values[0]:
            exp = centre + 2 * (refl - centre)
            ev = _arima_fit(exp, data, model)
            ev = ev if np.isfinite(ev) else np.inf
            if ev < rv:
                points[-1] = exp
                values[-1] = ev
            else:
                points[-1] = refl
                values[-1] = rv
            continue
        con = centre - 0.5 * (centre - points[-1])
        cv = _arima_fit(con, data, model)
        cv = cv if np.isfinite(cv) else np.inf
        if cv < values[-1]:
            points[-1] = con
            values[-1] = cv
            continue
        for i in range(1, len(points)):
            points[i] = points[0] - 0.5 * (points[0] - points[i])
            v = _arima_fit(points[i], data, model)
            values[i] = v if np.isfinite(v) else np.inf
        if np.isfinite(values[0]) and np.max(np.abs(values - values[0])) < tol:
            break
    b = np.argmin(values)
    return points[b], values[b]


def lagmat(x, lags, start):
    return np.column_stack([x[start - k : len(x) - k] for k in range(1, lags + 1)])


def hannan_rissanen(x, c, p, q):
    """Hannan-Rissanen estimates in aeon's layout [c?, phi..., theta...]."""
    n = len(x)
    if p == 0 and q == 0:
        return np.array([x.mean()]) if c else np.zeros(0)
    if q == 0:
        X = lagmat(x, p, p)
        if c:
            X = np.column_stack([np.ones(len(X)), X])
        return np.linalg.lstsq(X, x[p:], rcond=None)[0]
    m = int(min(max(p + q, np.ceil(10 * np.log10(n))), n // 4))
    X = np.column_stack([np.ones(n - m), lagmat(x, m, m)])
    e = np.zeros(n)
    e[m:] = x[m:] - X @ np.linalg.lstsq(X, x[m:], rcond=None)[0]
    s = m + max(p, q)
    cols = [lagmat(x, p, s)] if p else []
    cols.append(lagmat(e, q, s))
    X = np.column_stack(cols)
    if c:
        X = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X, x[s:], rcond=None)[0]


def simplex_generic(k, s):
    pts = np.full((k + 1, k), s)
    for i in range(k):
        pts[i + 1, i] = s * 1.2
    return pts


def simplex_mean(x, c, p, k, s):
    pts = simplex_generic(k, s)
    if c:
        const = x.mean() * (1 - p * s)
        pts[:, 0] = const
        pts[1, 0] = const + 0.1 * x.std() + 1e-3
    return pts


def simplex_around(x0, x, c):
    k = len(x0)
    pts = np.tile(x0, (k + 1, 1))
    for i in range(k):
        step = 0.1 * x.std() + 1e-3 if (c and i == 0) else 0.1
        pts[i + 1, i] += step
    return pts


def run(x, c, p, q, max_iter):
    model = np.array([c, p, q], dtype=np.int32)
    k = c + p + q
    if k == 0:
        return None
    s = 0.1 / (c + p + q + 1)
    hr = hannan_rissanen(x, c, p, q)
    out = {}
    out["stock"] = nm_from(simplex_generic(k, s), x, model, max_iter)
    out["mean_init"] = nm_from(simplex_mean(x, c, p, k, s), x, model, max_iter)
    out["hr"] = nm_from(simplex_around(hr, x, c), x, model, max_iter)
    out["hr+restart"] = nm_from(simplex_around(out["hr"][0], x, c), x, model, max_iter)
    out["best(mean,hr)"] = min(out["mean_init"], out["hr"], key=lambda t: t[1])
    out["best(mean,hr)+restart"] = min(
        out["best(mean,hr)"],
        nm_from(simplex_around(out["best(mean,hr)"][0], x, c), x, model, max_iter),
        key=lambda t: t[1],
    )

    def loss(z):
        v = _arima_fit(np.asarray(z, dtype=np.float64), x, model)
        return v if np.isfinite(v) else 1e300

    best_found = min(out.values(), key=lambda t: t[1])[0]
    starts = [hr, np.r_[[x.mean()] * c, [0.0] * (p + q)], best_found]
    ref = min(
        minimize(
            loss,
            z,
            method="Nelder-Mead",
            options={"maxiter": 8000, "xatol": 1e-8, "fatol": 1e-8, "adaptive": True},
        ).fun
        for z in starts
    )
    ref = min([ref] + [v[1] for v in out.values()])
    return {name: v[1] - ref for name, v in out.items()}



def work(case):
    src, n, r, x, p, q = case
    res = run(x, 1, p, q, 200)
    return [
        {"max_iter": 200, "source": src, "n": n, "rep": r, "order": f"({p},{q})", "strategy": k, "gap": v}
        for k, v in res.items()
    ]


if __name__ == "__main__":
    from multiprocessing import Pool

    cases = []
    for name in ["arma_11", "ar_2", "ma_2", "white_noise", "arima_110", "arima_011_drift"]:
        d = 1 if name in ("arima_110", "arima_011_drift") else 0
        for n in (100, 500):
            for r in range(15):
                y = ALL_PROCESSES[name].sample(n, 5000 + r)
                x = np.diff(y, n=d) if d else y
                for p, q in [(1, 1), (2, 0), (0, 2), (2, 1), (3, 2)]:
                    cases.append((f"sim:{name}", n, r, x, p, q))
    air = np.diff(load_airline().astype(float))
    for p, q in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (3, 2)]:
        cases.append(("airline_d1", 143, 0, air, p, q))
    with Pool(8) as pool:
        rows = [row for rs in pool.map(work, cases, chunksize=4) for row in rs]
    df = pd.DataFrame(rows)
    df.to_csv(sys.argv[1], index=False)
    g = df.groupby("strategy")["gap"]
    print(pd.DataFrame({
        "fail(gap>0.1)": g.apply(lambda s: (s > 0.1).mean()),
        "fail(gap>2)": g.apply(lambda s: (s > 2).mean()),
        "fail(gap>10)": g.apply(lambda s: (s > 10).mean()),
        "median": g.median(), "p95": g.quantile(0.95)}).round(3).to_string())
    print(df[df.source == "airline_d1"].pivot_table(index="order", columns="strategy", values="gap").round(2).to_string())
