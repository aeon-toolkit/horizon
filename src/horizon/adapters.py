"""Uniform adapters over the automatic ARIMA/ETS forecasters of several toolkits.

Every adapter exposes the same contract::

    preds = adapter(y, h, sp)   # y 1d np.ndarray, h horizon, sp seasonal period
    preds.shape == (h,)

so the benchmark code never has to know whether a toolkit wants a numpy array,
a pandas Series with a RangeIndex, or a ``ForecastingHorizon``. One call does
model selection, parameter estimation and forecasting, i.e. it is the whole
"fit and forecast" cost that a user would pay.

Fairness notes
--------------
* aeon's :class:`~aeon.forecasting.stats.AutoARIMA` is non-seasonal, so every
  competitor is configured ``seasonal=False`` for the ARIMA comparison and the
  ARIMA DGPs contain no seasonality.
* All auto-ARIMA searches are given the same order limits (:data:`MAX_P`,
  :data:`MAX_D`, :data:`MAX_Q`), including aeon, rather than each toolkit's
  defaults.
* For ETS, the *true* seasonal period of the DGP is given to every toolkit, so
  no implementation is penalised for failing to detect it. aeon is additionally
  entered as ``aeon (auto sp)``, which estimates the period from the data.
* aeon's AutoETS searches error x {none, additive} trend x seasonality but no
  damped trend. Competitors keep their default candidate sets (which include
  damped trends) because that is what a user of each toolkit gets.
* statsmodels has no built-in automatic selection, so the grid search that a
  user would have to write themselves is included here and timed as part of
  the method. Those entries are labelled ``(grid)``.
* Every adapter records the chosen model in ``selected_model`` using a shared
  format, ``ARIMA(p,d,q)`` or ``ETS(E,T,S)``, so selections can be compared.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

__all__ = ["Adapter", "MAX_P", "MAX_D", "MAX_Q", "get_adapters", "availability_table"]

# Shared auto-ARIMA search limits, applied to every toolkit including aeon.
MAX_P = 3
MAX_D = 2
MAX_Q = 2


def _filter_kwargs(func: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs that ``func`` does not accept.

    The competitor toolkits rename and remove constructor arguments between
    releases. Filtering against the live signature means the benchmark keeps
    working rather than dying with a ``TypeError`` on a minor upgrade.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _ets_label(error, trend, seasonal, damped=False) -> str:
    """Format an ETS spec as ``ETS(E,T,S)`` using Hyndman's letters."""

    def letter(x):
        x = "" if x is None else str(x).strip().lower()
        if x in ("", "n", "none"):
            return "N"
        return "M" if x.startswith("m") else "A"

    t = letter(trend)
    if t != "N" and damped:
        t += "d"
    return f"ETS({letter(error)},{t},{letter(seasonal)})"


# aeon's auto_ets encodes components as ints: error 1=add, 2=mul; trend and
# seasonality 0=none, 1=add, 2=mul.
_AEON_ERROR = {1: "A", 2: "M"}
_AEON_COMPONENT = {0: "N", 1: "A", 2: "M"}


@dataclass
class Adapter:
    """A single toolkit/estimator entry in the comparison.

    Parameters
    ----------
    name : str
        Label used in tables, plots and file names.
    package : str
        Importable module that must be present for this adapter to run.
    family : str
        ``"arima"`` or ``"ets"``.
    fn : callable
        ``fn(y, h, sp) -> np.ndarray`` of shape ``(h,)``.
    notes : str
        Short description of what is actually being called.
    mode : str
        ``"auto"`` for automatic model selection, ``"fixed"`` to fit the model
        given in ``spec`` (the true generating model, see
        :data:`simulate.TRUE_SPECS`). Fixed adapters must have ``spec`` set
        before they are called.
    """

    name: str
    package: str
    family: str
    fn: Callable[[np.ndarray, int, int], np.ndarray] = None
    notes: str = ""
    selected_model: str = ""
    mode: str = "auto"
    spec: dict | None = None

    def available(self) -> bool:
        """Return True if the backing package can be imported."""
        import importlib

        try:
            importlib.import_module(self.package)
        except Exception:
            return False
        return True

    def __call__(self, y: np.ndarray, h: int, sp: int) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self.fn(self, y, h, sp)
        preds = np.asarray(preds, dtype=np.float64).ravel()
        if preds.shape != (h,):
            raise ValueError(
                f"{self.name} returned {preds.shape} forecasts, expected ({h},)"
            )
        return preds

    def warmup(self, sp: int = 12) -> None:
        """Run once on a short series to pay any JIT compilation cost up front.

        aeon and statsforecast both compile with numba on first call. Timing
        that compilation would make a one-off cost look like model fitting, so
        it is paid here and excluded from all reported run times.
        """
        n = max(8 * max(sp, 1), 60)
        rng = np.random.default_rng(0)
        # numba specialises on code paths as well as types, so exercise a
        # stationary series, a random walk (d > 0) and, if seasonal, a
        # seasonal series. Otherwise the first integrated series in the timed
        # runs pays a compilation cost.
        stationary = 100.0 + rng.normal(0, 1, n)
        walk = 200.0 + np.cumsum(rng.normal(0.05, 1, n))
        series = [stationary, walk]
        if sp > 1:
            season = 10.0 * np.sin(2 * np.pi * np.arange(n) / sp)
            series.append(stationary + season)
        for y in series:
            self(np.abs(y) + 1.0, 2, sp)


# ---------------------------------------------------------------------------
# aeon
# ---------------------------------------------------------------------------


def _aeon_arima(self, y, h, sp):
    from aeon.forecasting.stats import AutoARIMA

    f = AutoARIMA(max_p=MAX_P, max_d=MAX_D, max_q=MAX_Q)
    preds = f.iterative_forecast(y, prediction_horizon=h)
    self.selected_model = f"ARIMA({f.p_},{f.d_},{f.q_})"
    return preds


def _aeon_ets(self, y, h, sp):
    from aeon.forecasting.stats import AutoETS

    # aeon treats seasonal_period=None as "estimate from the data"; an integer
    # is a request for a seasonal model, which only makes sense when sp > 1.
    f = AutoETS(seasonal_period=sp if sp > 1 else None)
    preds = f.iterative_forecast(y, prediction_horizon=h)
    self.selected_model = _aeon_ets_label(f)
    return preds


def _aeon_ets_label(f):
    trend = _AEON_COMPONENT[f.trend_type_]
    # aeon >= the estimation fixes reports damped_trend_; before, any trend
    # was damped (phi always estimated)
    if trend != "N" and getattr(f, "damped_trend_", True):
        trend += "d"
    return (
        f"ETS({_AEON_ERROR[f.error_type_]},{trend},"
        f"{_AEON_COMPONENT[f.seasonality_type_]})"
    )


def _aeon_ets_auto_sp(self, y, h, sp):
    from aeon.forecasting.stats import AutoETS

    f = AutoETS(seasonal_period=None)
    preds = f.iterative_forecast(y, prediction_horizon=h)
    self.selected_model = f"{_aeon_ets_label(f)} m={f.seasonal_period_}"
    return preds


# ---------------------------------------------------------------------------
# statsforecast (Nixtla)
# ---------------------------------------------------------------------------


def _statsforecast_arima(self, y, h, sp):
    from statsforecast.models import AutoARIMA

    kwargs = _filter_kwargs(
        AutoARIMA.__init__,
        dict(
            season_length=1,
            seasonal=False,
            max_p=MAX_P,
            max_d=MAX_D,
            max_q=MAX_Q,
            max_order=MAX_P + MAX_Q,
            stepwise=True,
        ),
    )
    m = AutoARIMA(**kwargs)
    m.fit(y)
    preds = m.predict(h=h)["mean"]
    arma = getattr(m, "model_", {}).get("arma", None)
    if arma is not None:
        # statsforecast stores arma as (p, q, P, Q, m, d, D)
        self.selected_model = f"ARIMA({arma[0]},{arma[5]},{arma[1]})"
    return preds


def _statsforecast_ets(self, y, h, sp):
    from statsforecast.models import AutoETS

    kwargs = _filter_kwargs(
        AutoETS.__init__, dict(season_length=max(sp, 1), model="ZZZ")
    )
    m = AutoETS(**kwargs)
    m.fit(y)
    preds = m.predict(h=h)["mean"]
    method = getattr(m, "model_", {}).get("method", None)
    if method is not None:
        # e.g. "ETS(A,Ad,N)", already in the shared format
        self.selected_model = str(method)
    return preds


# ---------------------------------------------------------------------------
# sktime
# ---------------------------------------------------------------------------


def _as_sktime_series(y):
    import pandas as pd

    return pd.Series(y, index=pd.RangeIndex(len(y)), name="y")


def _sktime_arima(self, y, h, sp):
    import numpy as np
    from sktime.forecasting.arima import AutoARIMA

    kwargs = _filter_kwargs(
        AutoARIMA.__init__,
        dict(
            seasonal=False,
            sp=1,
            max_p=MAX_P,
            max_d=MAX_D,
            max_q=MAX_Q,
            max_order=MAX_P + MAX_Q,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            n_jobs=1,
        ),
    )
    f = AutoARIMA(**kwargs)
    f.fit(_as_sktime_series(y))
    preds = f.predict(fh=np.arange(1, h + 1))
    order = f.get_fitted_params().get("order")
    if order is not None:
        self.selected_model = "ARIMA({},{},{})".format(*order)
    return np.asarray(preds)


def _sktime_ets(self, y, h, sp):
    import numpy as np
    from sktime.forecasting.ets import AutoETS

    kwargs = _filter_kwargs(
        AutoETS.__init__,
        dict(
            auto=True,
            sp=max(sp, 1),
            seasonal="add" if sp > 1 else None,
            information_criterion="aic",
            allow_multiplicative_trend=False,
            n_jobs=1,
        ),
    )
    f = AutoETS(**kwargs)
    f.fit(_as_sktime_series(y))
    preds = f.predict(fh=np.arange(1, h + 1))
    res = getattr(f, "_fitted_forecaster", None)
    if res is not None:
        m = res.model
        self.selected_model = _ets_label(m.error, m.trend, m.seasonal, m.damped_trend)
    return np.asarray(preds)


# ---------------------------------------------------------------------------
# skforecast
# ---------------------------------------------------------------------------


def _skforecast_arima(self, y, h, sp):
    import numpy as np
    from skforecast.stats import Arima

    kwargs = _filter_kwargs(
        Arima.__init__,
        dict(
            order=None,
            seasonal_order=None,
            m=1,
            seasonal=False,
            max_p=MAX_P,
            max_d=MAX_D,
            max_q=MAX_Q,
            max_order=MAX_P + MAX_Q,
            stepwise=True,
        ),
    )
    est = Arima(**kwargs)
    est.fit(y=_as_sktime_series(y))
    preds = est.predict(steps=h)
    arma = getattr(est, "arma_", None)
    if arma is not None:
        # same layout as statsforecast: (p, q, P, Q, m, d, D)
        self.selected_model = f"ARIMA({arma[0]},{arma[5]},{arma[1]})"
    return np.asarray(preds).ravel()


def _skforecast_ets(self, y, h, sp):
    import numpy as np
    from skforecast.stats import Ets

    kwargs = _filter_kwargs(
        Ets.__init__,
        dict(
            m=max(sp, 1),
            model="ZZZ",
            seasonal=sp > 1,
            ic="aic",
            allow_multiplicative_trend=False,
        ),
    )
    est = Ets(**kwargs)
    est.fit(y=_as_sktime_series(y))
    preds = est.predict(steps=h)
    cfg = getattr(getattr(est, "model_", None), "config", None)
    if cfg is not None:
        self.selected_model = _ets_label(cfg.error, cfg.trend, cfg.season, cfg.damped)
    return np.asarray(preds).ravel()


# ---------------------------------------------------------------------------
# pmdarima
# ---------------------------------------------------------------------------


def _pmdarima_arima(self, y, h, sp):
    import numpy as np
    from pmdarima import auto_arima

    kwargs = _filter_kwargs(
        auto_arima,
        dict(
            seasonal=False,
            m=1,
            max_p=MAX_P,
            max_d=MAX_D,
            max_q=MAX_Q,
            max_order=MAX_P + MAX_Q,
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
            n_jobs=1,
        ),
    )
    model = auto_arima(y, **kwargs)
    self.selected_model = "ARIMA({},{},{})".format(*model.order)
    return np.asarray(model.predict(n_periods=h))


# ---------------------------------------------------------------------------
# statsmodels, with the model selection a user would have to write themselves
# ---------------------------------------------------------------------------


def _statsmodels_arima(self, y, h, sp):
    """Hyndman-Khandakar style selection over a full (p, d, q) grid by AIC.

    statsmodels has no auto-ARIMA. ``d`` is chosen by repeated KPSS testing as
    aeon does, then every ``(p, q)`` within the shared limits is fitted and the
    lowest AIC wins. The whole search is timed, because that is what using
    statsmodels for this task actually costs.
    """
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import kpss

    d = 0
    work = y.copy()
    while d < MAX_D:
        try:
            p_value = kpss(work, regression="c", nlags="auto")[1]
        except Exception:
            break
        if p_value >= 0.05:  # fail to reject stationarity
            break
        work = np.diff(work, n=1)
        d += 1

    best_aic, best_res, best_order = np.inf, None, None
    trend = "c" if d == 0 else "n"
    for p in range(MAX_P + 1):
        for q in range(MAX_Q + 1):
            try:
                res = ARIMA(y, order=(p, d, q), trend=trend).fit()
            except Exception:
                continue
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic, best_res, best_order = res.aic, res, (p, d, q)
    if best_res is None:
        self.selected_model = "failed"
        return np.full(h, float(np.mean(y)))
    self.selected_model = "ARIMA({},{},{})".format(*best_order)
    return np.asarray(best_res.forecast(steps=h))


def _statsmodels_ets(self, y, h, sp):
    """Grid search over the ETS taxonomy by AIC using ``ETSModel``.

    Mirrors the candidate set aeon's AutoETS searches: additive or
    multiplicative error, none or additive (optionally damped) trend, and none,
    additive or multiplicative seasonality. Multiplicative components are only
    tried on strictly positive data, as statsmodels requires.
    """
    import numpy as np
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    positive = bool(np.min(y) > 0)
    errors = ["add", "mul"] if positive else ["add"]
    seasonals = [None]
    if sp > 1 and len(y) > 2 * sp:
        seasonals += ["add"] + (["mul"] if positive else [])

    best_aic, best_res, best_spec = np.inf, None, None
    for error in errors:
        for trend, damped in [(None, False), ("add", False), ("add", True)]:
            for seasonal in seasonals:
                try:
                    model = ETSModel(
                        y,
                        error=error,
                        trend=trend,
                        damped_trend=damped,
                        seasonal=seasonal,
                        seasonal_periods=sp if seasonal else None,
                    )
                    res = model.fit(disp=False)
                except Exception:
                    continue
                if np.isfinite(res.aic) and res.aic < best_aic:
                    best_aic, best_res = res.aic, res
                    best_spec = (error, trend, damped, seasonal)
    if best_res is None:
        self.selected_model = "failed"
        return np.full(h, float(np.mean(y)))
    self.selected_model = _ets_label(*best_spec[:2], best_spec[3], best_spec[2])
    return np.asarray(best_res.forecast(steps=h))


# ---------------------------------------------------------------------------
# Fixed order: fit the true generating model, no model selection
# ---------------------------------------------------------------------------
#
# ARIMA ``spec["constant"]`` is a mean when d == 0 and a drift when d > 0, and
# each toolkit is given whichever of its own flags expresses that.
#
# aeon's ETS estimates the damping parameter phi whenever there is a trend, so
# it cannot fit an undamped trend. For the undamped trend processes aeon fits a
# damped trend while the competitors fit the exact model. This is a property of
# aeon, recorded in ``selected_model`` as ``Ad``.


def _arima_label(spec):
    return "ARIMA({},{},{})".format(*spec["order"])


def _ets_letters(spec):
    """Hyndman model string, e.g. "MAM", and the damped flag."""
    e = _ets_label(spec["error"], spec["trend"], spec["seasonal"])[4:-1]
    return e.replace(",", ""), bool(spec["damped"])


def _fixed_ets_label(spec):
    return _ets_label(spec["error"], spec["trend"], spec["seasonal"], spec["damped"])


def _aeon_arima_fixed(self, y, h, sp):
    from aeon.forecasting.stats import ARIMA

    p, d, q = self.spec["order"]
    f = ARIMA(p=p, d=d, q=q, use_constant=self.spec["constant"])
    preds = f.iterative_forecast(y, prediction_horizon=h)
    self.selected_model = _arima_label(self.spec)
    return preds


def _aeon_ets_fixed(self, y, h, sp):
    from aeon.forecasting.stats import ETS

    code = {None: 0, "add": 1, "mul": 2}
    spec = self.spec
    kwargs = dict(
        error_type=code[spec["error"]],
        trend_type=code[spec["trend"]],
        seasonality_type=code[spec["seasonal"]],
        seasonal_period=sp if spec["seasonal"] else 1,
    )
    # aeon versions without damped_trend always damp (estimate phi)
    can_undamp = "damped_trend" in inspect.signature(ETS.__init__).parameters
    if can_undamp:
        kwargs["damped_trend"] = spec["damped"]
    f = ETS(**kwargs)
    preds = f.iterative_forecast(y, prediction_horizon=h)
    self.selected_model = _ets_label(
        spec["error"],
        spec["trend"],
        spec["seasonal"],
        damped=spec["damped"] if can_undamp else True,
    )
    return preds


def _statsforecast_arima_fixed(self, y, h, sp):
    from statsforecast.models import ARIMA

    order, const = self.spec["order"], self.spec["constant"]
    m = ARIMA(
        order=order,
        include_mean=const and order[1] == 0,
        include_drift=const and order[1] > 0,
    )
    m.fit(y)
    self.selected_model = _arima_label(self.spec)
    return m.predict(h=h)["mean"]


def _statsforecast_ets_fixed(self, y, h, sp):
    from statsforecast.models import AutoETS

    model, damped = _ets_letters(self.spec)
    m = AutoETS(season_length=max(sp, 1), model=model, damped=damped)
    m.fit(y)
    self.selected_model = str(m.model_.get("method", _fixed_ets_label(self.spec)))
    return m.predict(h=h)["mean"]


def _sktime_arima_fixed(self, y, h, sp):
    from sktime.forecasting.arima import ARIMA

    f = ARIMA(
        order=self.spec["order"],
        with_intercept=self.spec["constant"],
        suppress_warnings=True,
    )
    f.fit(_as_sktime_series(y))
    self.selected_model = _arima_label(self.spec)
    return np.asarray(f.predict(fh=np.arange(1, h + 1)))


def _sktime_ets_fixed(self, y, h, sp):
    from sktime.forecasting.ets import AutoETS

    spec = self.spec
    f = AutoETS(
        auto=False,
        error=spec["error"],
        trend=spec["trend"],
        damped_trend=spec["damped"],
        seasonal=spec["seasonal"],
        sp=max(sp, 1),
    )
    f.fit(_as_sktime_series(y))
    self.selected_model = _fixed_ets_label(spec)
    return np.asarray(f.predict(fh=np.arange(1, h + 1)))


def _skforecast_arima_fixed(self, y, h, sp):
    from skforecast.stats import Arima

    order, const = self.spec["order"], self.spec["constant"]
    est = Arima(
        order=order,
        seasonal_order=(0, 0, 0),
        m=1,
        fit_intercept=const and order[1] == 0,
        include_drift=const and order[1] > 0,
    )
    est.fit(y=_as_sktime_series(y))
    arma = getattr(est, "arma_", None)
    self.selected_model = (
        f"ARIMA({arma[0]},{arma[5]},{arma[1]})"
        if arma is not None
        else _arima_label(self.spec)
    )
    return np.asarray(est.predict(steps=h)).ravel()


def _skforecast_ets_fixed(self, y, h, sp):
    from skforecast.stats import Ets

    model, damped = _ets_letters(self.spec)
    est = Ets(m=max(sp, 1), model=model, damped=damped)
    est.fit(y=_as_sktime_series(y))
    cfg = getattr(getattr(est, "model_", None), "config", None)
    self.selected_model = (
        _ets_label(cfg.error, cfg.trend, cfg.season, cfg.damped)
        if cfg is not None
        else _fixed_ets_label(self.spec)
    )
    return np.asarray(est.predict(steps=h)).ravel()


def _pmdarima_arima_fixed(self, y, h, sp):
    from pmdarima import ARIMA

    model = ARIMA(
        order=self.spec["order"],
        with_intercept=self.spec["constant"],
        suppress_warnings=True,
    )
    model.fit(y)
    self.selected_model = _arima_label(self.spec)
    return np.asarray(model.predict(n_periods=h))


def _statsmodels_arima_fixed(self, y, h, sp):
    from statsmodels.tsa.arima.model import ARIMA

    order, const = self.spec["order"], self.spec["constant"]
    # statsmodels: "c" is a mean for d == 0; for d > 0 a drift is trend "t"
    trend = ("c" if order[1] == 0 else "t") if const else "n"
    res = ARIMA(y, order=order, trend=trend).fit()
    self.selected_model = _arima_label(self.spec)
    return np.asarray(res.forecast(steps=h))


def _statsmodels_ets_fixed(self, y, h, sp):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    spec = self.spec
    res = ETSModel(
        y,
        error=spec["error"],
        trend=spec["trend"],
        damped_trend=spec["damped"],
        seasonal=spec["seasonal"],
        seasonal_periods=sp if spec["seasonal"] else None,
    ).fit(disp=False)
    self.selected_model = _fixed_ets_label(spec)
    return np.asarray(res.forecast(steps=h))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: list[Adapter] = [
    # --- AutoARIMA ---------------------------------------------------------
    Adapter("aeon", "aeon", "arima", _aeon_arima, "aeon.forecasting.stats.AutoARIMA"),
    Adapter(
        "statsforecast",
        "statsforecast",
        "arima",
        _statsforecast_arima,
        "statsforecast.models.AutoARIMA",
    ),
    Adapter(
        "sktime", "sktime", "arima", _sktime_arima, "sktime.forecasting.arima.AutoARIMA"
    ),
    Adapter(
        "skforecast", "skforecast", "arima", _skforecast_arima, "skforecast.stats.Arima"
    ),
    Adapter("pmdarima", "pmdarima", "arima", _pmdarima_arima, "pmdarima.auto_arima"),
    Adapter(
        "statsmodels (grid)",
        "statsmodels",
        "arima",
        _statsmodels_arima,
        "statsmodels ARIMA + AIC grid search written here",
    ),
    # --- AutoETS -----------------------------------------------------------
    Adapter("aeon", "aeon", "ets", _aeon_ets, "aeon.forecasting.stats.AutoETS"),
    Adapter(
        "aeon (auto sp)",
        "aeon",
        "ets",
        _aeon_ets_auto_sp,
        "aeon AutoETS estimating the seasonal period",
    ),
    Adapter(
        "statsforecast",
        "statsforecast",
        "ets",
        _statsforecast_ets,
        "statsforecast.models.AutoETS",
    ),
    Adapter(
        "sktime", "sktime", "ets", _sktime_ets, "sktime.forecasting.ets.AutoETS(auto)"
    ),
    Adapter("skforecast", "skforecast", "ets", _skforecast_ets, "skforecast.stats.Ets"),
    Adapter(
        "statsmodels (grid)",
        "statsmodels",
        "ets",
        _statsmodels_ets,
        "statsmodels ETSModel + AIC grid search written here",
    ),
]

_FIXED = [
    ("arima", "aeon", "aeon", _aeon_arima_fixed, "aeon ARIMA"),
    ("arima", "statsforecast", "statsforecast", _statsforecast_arima_fixed, "ARIMA"),
    ("arima", "sktime", "sktime", _sktime_arima_fixed, "sktime ARIMA (pmdarima)"),
    ("arima", "skforecast", "skforecast", _skforecast_arima_fixed, "Arima(order)"),
    ("arima", "pmdarima", "pmdarima", _pmdarima_arima_fixed, "pmdarima.ARIMA"),
    ("arima", "statsmodels", "statsmodels", _statsmodels_arima_fixed, "ARIMA"),
    ("ets", "aeon", "aeon", _aeon_ets_fixed, "aeon ETS (phi always estimated)"),
    ("ets", "statsforecast", "statsforecast", _statsforecast_ets_fixed, "AutoETS(model)"),
    ("ets", "sktime", "sktime", _sktime_ets_fixed, "AutoETS(auto=False)"),
    ("ets", "skforecast", "skforecast", _skforecast_ets_fixed, "Ets(model)"),
    ("ets", "statsmodels", "statsmodels", _statsmodels_ets_fixed, "ETSModel"),
]
_REGISTRY += [
    Adapter(name, pkg, fam, fn, f"fixed order: {notes}", mode="fixed")
    for fam, name, pkg, fn, notes in _FIXED
]


def get_adapters(
    family: str,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    only_available: bool = True,
    mode: str = "auto",
) -> list[Adapter]:
    """Return fresh adapter instances for ``family``.

    Parameters
    ----------
    family : {"arima", "ets"}
        Which comparison to build.
    mode : {"auto", "fixed"}
        Automatic model selection, or fit the true model (set ``spec`` on each
        returned adapter before calling it).
    include : list of str or None
        If given, keep only adapters with these names.
    exclude : list of str or None
        Drop adapters with these names.
    only_available : bool
        If True, silently drop adapters whose package is not installed.

    Returns
    -------
    list of Adapter
    """
    out = []
    for a in _REGISTRY:
        if a.family != family or a.mode != mode:
            continue
        if include and a.name not in include:
            continue
        if exclude and a.name in exclude:
            continue
        fresh = Adapter(a.name, a.package, a.family, a.fn, a.notes, mode=a.mode)
        if only_available and not fresh.available():
            continue
        out.append(fresh)
    return out


def availability_table() -> "Any":
    """Return a DataFrame of every registered adapter and whether it can run."""
    import pandas as pd

    rows = []
    for a in _REGISTRY:
        version = ""
        if a.available():
            import importlib

            version = getattr(importlib.import_module(a.package), "__version__", "")
        rows.append(
            {
                "mode": a.mode,
                "family": a.family,
                "name": a.name,
                "package": a.package,
                "version": version,
                "available": a.available(),
                "notes": a.notes,
            }
        )
    return pd.DataFrame(rows)
