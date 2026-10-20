"""Final loss and time of the full multi-start ARIMA/ETS parameter fits.

Needs the aeon branch with ``_fit_arima_parameters``/``_fit_ets_parameters``.
Writes one row per fit; run once per code version on the same cases and
compare the CSVs (used to evaluate the stopping rule and fastmath changes):

    python studies/fit_loss_and_time.py <label> <out.csv>
"""
import sys, time
import numpy as np, pandas as pd
from horizon.simulate import ALL_PROCESSES
import aeon
from aeon.forecasting.utils._nelder_mead import _fit_arima_parameters, _fit_ets_parameters
print(sys.argv[1], aeon.__file__, flush=True)
cases = []
for name in ["arma_11", "ar_2", "ma_2", "white_noise", "arima_110", "arima_011_drift"]:
    d = 1 if name in ("arima_110", "arima_011_drift") else 0
    for n in (100, 500, 2000):
        for r in range(8):
            y = ALL_PROCESSES[name].sample(n, 9000 + r); x = np.diff(y, n=d) if d else y
            for p, q in [(0, 0), (1, 1), (2, 0), (0, 2), (2, 2), (3, 2)]:
                cases.append(("arima", name, n, r, f"{p}{q}", x, np.array([1, p, q], dtype=np.int32)))
for name in ["ets_ann", "ets_aan", "ets_adn", "ets_ana_m4", "ets_aaa_m12", "ets_mam_m12"]:
    sp = ALL_PROCESSES[name].seasonal_period
    for n in (100, 500, 2000):
        for r in range(8):
            y = ALL_PROCESSES[name].sample(n, 9000 + r)
            for e in (1, 2):
                for t, dmp in ((0, 0), (1, 0), (1, 1)):
                    for s in ((0, 1, 2) if sp > 1 else (0,)):
                        cases.append(("ets", name, n, r, f"{e}{t}{dmp}{s}", y, np.array([e, t, s, sp if s else 1, dmp], dtype=np.int32)))
# compile
_fit_arima_parameters(cases[0][5], cases[0][6], 200, 0.05); _fit_ets_parameters(cases[-1][5], cases[-1][6], 200)
rows = []
for fam, name, n, r, m, x, model in cases:
    t0 = time.perf_counter()
    if fam == "arima":
        _, v = _fit_arima_parameters(x, model, 500, 0.1 / (model.sum() + 1))
    else:
        _, v = _fit_ets_parameters(x, model, 500)
    rows.append(dict(family=fam, process=name, n=n, rep=r, model=m, loss=v, sec=time.perf_counter() - t0))
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
