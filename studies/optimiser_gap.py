"""How far aeon's ARIMA/ETS fits are from the optimum of aeon's own loss.

Works with any aeon. For the true model of each process (100 ARIMA, 50 ETS
series of length 500) it compares the loss aeon reaches with the best found by
scipy Nelder-Mead from several starting points on the same loss, and records
forecast MASE and AutoARIMA/AutoETS model selection. Run once per aeon version
(different environments, or ``PYTHONPATH`` pointing at another aeon checkout):

    python studies/optimiser_gap.py <label> <out.csv>
"""

import itertools
import sys
import warnings

warnings.simplefilter("ignore")

import aeon  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA as SM  # noqa: E402

from aeon.forecasting.stats import ARIMA, ETS, AutoARIMA, AutoETS  # noqa: E402
from aeon.forecasting.stats._ets import _numba_predict  # noqa: E402
from aeon.forecasting.utils._loss_functions import _arima_fit, _ets_fit  # noqa: E402
from horizon.experiments import score  # noqa: E402
from horizon.simulate import ALL_PROCESSES, TRUE_MODELS, TRUE_SPECS  # noqa: E402

label, out = sys.argv[1], sys.argv[2]
print(label, "aeon from", aeon.__file__, flush=True)
N, R_ARIMA, R_ETS = 500, 100, 50
rows = []


def nm(loss, x0):
    r = minimize(
        loss,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8, "adaptive": True},
    )
    return r.x, r.fun


# ----------------------------------------------------------------- ARIMA
for name in ["arma_11", "ar_2", "ma_2", "white_noise"]:
    p, d, q = TRUE_SPECS[name]["order"]
    model = np.array([1, p, q], dtype=np.int32)
    for r in range(R_ARIMA):
        y = ALL_PROCESSES[name].sample(N + 12, 1000 + r)
        tr, te = y[:N], y[N:]
        f = ARIMA(p, d, q, use_constant=True)
        pred = f.iterative_forecast(tr, prediction_horizon=12)
        aeon_x = f._parameters.copy()
        aeon_loss = _arima_fit(aeon_x, tr, model)

        def loss(x):
            v = _arima_fit(np.asarray(x, dtype=np.float64), tr, model)
            return v if np.isfinite(v) else 1e300

        # multi-start: from aeon's answer and from the statsmodels MLE
        sm = SM(tr, order=(p, d, q), trend="c").fit().params
        mean, ar, ma = sm[0], sm[1 : 1 + p], sm[1 + p : 1 + p + q]
        x_sm = np.r_[mean * (1 - ar.sum()), ar, ma]
        best = min(nm(loss, aeon_x)[1], nm(loss, x_sm)[1], aeon_loss)

        auto = AutoARIMA(max_p=3, max_d=2, max_q=2)
        pa = auto.iterative_forecast(tr, prediction_horizon=12)
        sel = f"ARIMA({auto.p_},{auto.d_},{auto.q_})"
        rows.append(
            {
                "variant": label,
                "family": "arima",
                "process": name,
                "replicate": r,
                "fixed_loss": aeon_loss,
                "best_loss": best,
                "gap": aeon_loss - best,
                "fixed_mase": score(tr, te, pred, 1)["mase"],
                "multistart_mase": np.nan,
                "auto_correct": sel == TRUE_MODELS[name],
                "auto_mase": score(tr, te, pa, 1)["mase"],
            }
        )
    print(label, name, "done", flush=True)


# ----------------------------------------------------------------- ETS
def ets_forecast(x, data, model, h):
    st = _ets_fit(np.asarray(x, dtype=np.float64), data, model)
    level, trend, season, n_t = st[1], st[2], st[3], st[4]
    phi = x[2] if model[1] != 0 else 1.0
    return np.array(
        [
            _numba_predict(model[1], model[2], level, trend, season, phi, i, n_t, model[3])
            for i in range(1, h + 1)
        ]
    )


code = {None: 0, "add": 1, "mul": 2}
for name in ["ets_aan", "ets_adn", "ets_aaa_m12", "ets_mam_m12"]:
    spec, sp = TRUE_SPECS[name], ALL_PROCESSES[name].seasonal_period
    h = max(12, 2 * sp)
    et, tt, st = code[spec["error"]], code[spec["trend"]], code[spec["seasonal"]]
    model = np.array([et, tt, st, sp if st else 1], dtype=np.int32)
    # grid of starting points in aeon's parameter layout [alpha, (beta, phi), (gamma)]
    grid = []
    for a, bf, ph, gf in itertools.product(
        [0.05, 0.2, 0.5], [0.05, 0.3], [0.8, 0.9, 0.98], [0.05, 0.3]
    ):
        x = [a]
        if tt:
            x += [bf * a, ph]
        if st:
            x += [gf * (1 - a)]
        grid.append(x)
    grid = [list(g) for g in dict.fromkeys(map(tuple, grid))]
    for r in range(R_ETS):
        y = ALL_PROCESSES[name].sample(N + h, 2000 + r)
        tr, te = y[:N], y[N:]
        f = ETS(et, tt, st, sp if st else 1)
        pred = f.iterative_forecast(tr, prediction_horizon=h)
        aeon_loss = _ets_fit(np.asarray(f.parameters_, dtype=np.float64), tr, model)[0]

        def loss(x):
            v = _ets_fit(np.asarray(x, dtype=np.float64), tr, model)[0]
            return v if np.isfinite(v) else 1e300

        # grid search, then Nelder-Mead from the 3 best grid points and aeon's
        gl = np.array([loss(g) for g in grid])
        starts = [grid[i] for i in np.argsort(gl)[:3]] + [list(f.parameters_)]
        fits = [nm(loss, s) for s in starts]
        bx, bl = min(fits, key=lambda t: t[1])
        best = min(bl, aeon_loss)
        ms_pred = ets_forecast(bx, tr, model, h)

        auto = AutoETS(seasonal_period=sp if sp > 1 else None)
        pa = auto.iterative_forecast(tr, prediction_horizon=h)
        letters = {0: "N", 1: "A", 2: "M"}
        sel = (
            f"ETS({ {1: 'A', 2: 'M'}[auto.error_type_] },"
            f"{letters[auto.trend_type_]},{letters[auto.seasonality_type_]})"
        )
        truth = TRUE_MODELS[name].replace("Ad", "A")  # aeon has no damped label
        rows.append(
            {
                "variant": label,
                "family": "ets",
                "process": name,
                "replicate": r,
                "fixed_loss": aeon_loss,
                "best_loss": best,
                "gap": aeon_loss - best,
                "fixed_mase": score(tr, te, pred, sp)["mase"],
                "multistart_mase": score(tr, te, ms_pred, sp)["mase"],
                "auto_correct": sel == truth,
                "auto_mase": score(tr, te, pa, sp)["mase"],
            }
        )
    print(label, name, "done", flush=True)

pd.DataFrame(rows).to_csv(out, index=False)
