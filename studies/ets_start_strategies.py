"""Nelder-Mead starting strategies for ETS on aeon's ETS loss.

Needs the aeon branch with ``_fit_ets_parameters`` (the "aeon_impl" strategy).
Fits every AutoETS candidate to each ETS process from aeon's default simplex,
an R forecast::ets style start, and combinations with a restart. The gap is
the loss reached minus the best loss from a grid search followed by scipy
Nelder-Mead from the best grid points.

    python studies/ets_start_strategies.py <out.csv>
"""

import itertools
import sys
import warnings

warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

from aeon.forecasting.utils._loss_functions import _ets_fit
from aeon.forecasting.utils._nelder_mead import _fit_ets_parameters, _init_ets_simplex

from horizon.simulate import ALL_PROCESSES  # noqa: E402


@njit(cache=True)
def f(x, data, model):
    v = _ets_fit(x, data, model)[0]
    return v if np.isfinite(v) else np.inf


@njit(cache=True)
def nm_from(points, data, model, max_iter, tol=1e-6):
    """aeon's nelder_mead loop, verbatim, from a supplied starting simplex."""
    points = points.copy()
    values = np.empty(len(points))
    for i in range(len(points)):
        values[i] = f(points[i].copy(), data, model)
    for _ in range(max_iter):
        order = np.argsort(values)
        points = points[order]
        values = values[order]
        centre = points[:-1].sum(axis=0) / len(points[:-1])
        refl = centre + (centre - points[-1])
        rv = f(refl, data, model)
        if values[0] <= rv < values[-2]:
            points[-1] = refl
            values[-1] = rv
            continue
        if rv < values[0]:
            exp = centre + 2 * (refl - centre)
            ev = f(exp, data, model)
            if ev < rv:
                points[-1] = exp
                values[-1] = ev
            else:
                points[-1] = refl
                values[-1] = rv
            continue
        con = centre - 0.5 * (centre - points[-1])
        cv = f(con, data, model)
        if cv < values[-1]:
            points[-1] = con
            values[-1] = cv
            continue
        for i in range(1, len(points)):
            points[i] = points[0] - 0.5 * (points[0] - points[i])
            values[i] = f(points[i], data, model)
        if np.isfinite(values[0]) and np.max(np.abs(values - values[0])) < tol:
            break
    b = np.argmin(values)
    return points[b], values[b]


def layout(model, alpha, beta, phi, gamma):
    x = [alpha]
    if model[1]:
        x += [beta, phi]
    if model[2]:
        x += [gamma]
    return np.array(x, dtype=np.float64)


def simplex_around(x0, model):
    """Each vertex moves one parameter towards the interior by 10% of its range."""
    k = len(x0)
    pts = np.tile(x0, (k + 1, 1))
    for i in range(k):
        is_phi = model[1] and i == 2
        pts[i + 1, i] = x0[i] - 0.08 if is_phi else x0[i] * 1.5 + 0.01
    return pts


def run(data, model, max_iter):
    k = 1 + 2 * (model[1] != 0) + (model[2] != 0)
    stock = np.empty((k + 1, k))
    _init_ets_simplex(stock, model)
    r_x0 = layout(model, 0.2, 0.02, 0.978, 0.04)
    low_x0 = layout(model, 0.05, 0.005, 0.95, 0.05)
    out = {}
    out["stock"] = nm_from(stock, data, model, max_iter)
    out["R_start"] = nm_from(simplex_around(r_x0, model), data, model, max_iter)
    out["stock+restart"] = min(
        out["stock"],
        nm_from(simplex_around(out["stock"][0], model), data, model, max_iter),
        key=lambda t: t[1],
    )
    out["best(stock,R)"] = min(out["stock"], out["R_start"], key=lambda t: t[1])
    low = nm_from(simplex_around(low_x0, model), data, model, max_iter)
    out["best(stock,R,low)"] = min(out["best(stock,R)"], low, key=lambda t: t[1])
    out["aeon_impl"] = _fit_ets_parameters(data, model, max_iter)
    b = out["best(stock,R)"]
    out["best(stock,R)+restart"] = min(
        b, nm_from(simplex_around(b[0], model), data, model, max_iter), key=lambda t: t[1]
    )

    def loss(x):
        v = f(np.asarray(x, dtype=np.float64), data, model)
        return v if np.isfinite(v) else 1e300

    grid = [
        layout(model, a, bf * a, ph, gf * (1 - a))
        for a, bf, ph, gf in itertools.product(
            [0.02, 0.1, 0.3, 0.6, 0.9], [0.05, 0.5], [0.85, 0.98], [0.05, 0.5]
        )
    ]
    grid = [g for g in {tuple(g): g for g in grid}.values()]
    gl = np.array([loss(g) for g in grid])
    starts = [grid[i] for i in np.argsort(gl)[:4]] + [v[0] for v in out.values()]
    ref = min(
        minimize(loss, z, method="Nelder-Mead", options={"maxiter": 20000, "adaptive": True}).fun
        for z in starts
    )
    ref = min([ref, low[1]] + [v[1] for v in out.values()])
    return {name: v[1] - ref for name, v in out.items()}


# Fit every candidate AutoETS would try (true and misspecified) on each process.
candidates = [(e, t, s) for e in (1, 2) for t in (0, 1) for s in (0, 1, 2)]
rows = []
for name in ["ets_ann", "ets_aan", "ets_adn", "ets_ana_m4", "ets_aaa_m12", "ets_mam_m12"]:
    sp = ALL_PROCESSES[name].seasonal_period
    for n in (100, 500):
        for r in range(15):
            y = ALL_PROCESSES[name].sample(n, 7000 + r)
            for e, t, s in candidates:
                if s and sp == 1:
                    continue
                model = np.array([e, t, s, sp if s else 1], dtype=np.int32)
                res = run(y, model, 200)
                for k, v in res.items():
                    rows.append(
                        {"process": name, "n": n, "rep": r, "model": f"{e}{t}{s}", "strategy": k, "gap": v}
                    )
    print("done", name, flush=True)

df = pd.DataFrame(rows)
df = df[np.isfinite(df["gap"])]
df.to_csv(sys.argv[1], index=False)
g = df.groupby("strategy")["gap"]
print(
    pd.DataFrame(
        {
            "fail(gap>0.1)": g.apply(lambda s: (s > 0.1).mean()),
            "fail(gap>2)": g.apply(lambda s: (s > 2).mean()),
            "fail(gap>10)": g.apply(lambda s: (s > 10).mean()),
            "median": g.median(),
            "p95": g.quantile(0.95),
        }
    ).round(3).to_string()
)
