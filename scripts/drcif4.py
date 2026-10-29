"""Prototype: DrCIF with a fourth, seasonally adjusted representation.

DrCIF extracts interval features from three representations of each window:
the series, its first differences and its periodogram. ``DrCIF4Regressor``
adds a fourth, the window of the seasonally adjusted series, so the forest
chooses between raw and adjusted values instead of having the series
deseasonalised up front (as ``--deseasonalise`` does, which on LSSH-99 hurts
DrCIF on series with zeros).

The input is two channels per window: the series (channel 0) and the adjusted
series (channel 1). The first three representations use channel 0 only, the
fourth channel 1 only. ``SeasonalTargetDifferenceForecaster`` builds the two
channels: classical decomposition of the whole training series as on M4
(``_m4_seasonal_indices``; multiplicative, or additive if the series is not
strictly positive), with no seasonality test, whenever the series has at least
three seasons. Otherwise channel 1 is a copy of channel 0. The target is the
next change of the raw series, as ``DrCIFDeltaY``.

Importing this module registers ``DrCIF4DeltaY`` with ``m_comp_experiments``.
Run it without ``--deseasonalise``:

    PYTHONPATH=<aeon worktree> python scripts/drcif4.py aaltd --forecasters DrCIF4DeltaY ...
    PYTHONPATH=<aeon worktree> python scripts/drcif4.py m4long --min-length 500 ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import FunctionTransformer

sys.path.insert(0, str(Path(__file__).parent))

import m_comp_experiments as mc  # noqa: E402
from aeon.regression.interval_based import DrCIFRegressor  # noqa: E402
from aeon.transformations.collection import PeriodogramTransformer  # noqa: E402
from aeon.utils.numba.general import first_order_differences_3d  # noqa: E402

NAME = "DrCIF4DeltaY"


# module level, so the transformers pickle for spawned workers
def _series(X):
    return X[:, :1]


def _differences(X):
    return first_order_differences_3d(np.ascontiguousarray(X[:, :1]))


def _periodogram(X):
    return PeriodogramTransformer().fit_transform(np.ascontiguousarray(X[:, :1]))


def _adjusted(X):
    return X[:, 1:2]


class DrCIF4Regressor(DrCIFRegressor):
    """DrCIF on (series, adjusted series) windows with four representations."""

    def __init__(self, base_estimator=None, n_estimators=200, n_intervals=(4, "sqrt-div"),
                 min_interval_length=3, max_interval_length=0.5, att_subsample_size=10,
                 time_limit_in_minutes=None, contract_max_n_estimators=500,
                 use_pycatch22="deprecated", random_state=None, n_jobs=1,
                 parallel_backend=None):
        super().__init__(
            base_estimator=base_estimator, n_estimators=n_estimators,
            n_intervals=n_intervals, min_interval_length=min_interval_length,
            max_interval_length=max_interval_length, att_subsample_size=att_subsample_size,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            use_pycatch22=use_pycatch22, random_state=random_state, n_jobs=n_jobs,
            parallel_backend=parallel_backend)
        self.series_transformers = [
            FunctionTransformer(func=f, validate=False)
            for f in (_series, _differences, _periodogram, _adjusted)
        ]


class SeasonalTargetDifferenceForecaster(mc.TargetDifferenceRegressionForecaster):
    """Next-change regression on windows of the series and the adjusted series."""

    def __init__(self, window: int, regressor, period: int):
        super().__init__(window, regressor)
        self.period = int(period)

    def _adjust(self, y):
        """The adjusted series for ``y`` (a training series or its extension)."""
        if self.figure_ is None:
            return y
        f = self.figure_[np.arange(y.size) % self.period]
        return y - f if self.additive_ else y / f

    def _channels(self, y):
        return np.stack([y, self._adjust(y)])

    def fit(self, y):
        y = mc._as_float_array(y)
        self.figure_, self.additive_ = None, False
        m = self.period
        if m > 1 and y.size >= 3 * m:
            self.additive_ = bool(np.min(y) <= 0)
            seasonal_in, _ = mc._m4_seasonal_indices(y, m, 1, self.additive_)
            figure = seasonal_in[:m]
            if np.all(np.isfinite(figure)) and (self.additive_ or np.all(figure > 0)):
                self.figure_ = figure
        return super().fit(y)

    def _x(self, X):
        return X  # already (n, 2, window)

    def _training_data(self, y):
        c = self._channels(y)
        X = np.lib.stride_tricks.sliding_window_view(c, self.window, axis=1)[:, :-1]
        target = y[self.window:] - y[self.window - 1:-1]
        return np.ascontiguousarray(X.transpose(1, 0, 2)), target

    def iterative_forecast(self, y, prediction_horizon: int) -> np.ndarray:
        history = mc._as_float_array(y)
        forecasts = np.empty(int(prediction_horizon), dtype=np.float64)
        for i in range(int(prediction_horizon)):
            x = self._channels(history)[:, -self.window:][None]
            nxt = history[-1] + float(self.regressor_.predict(x)[0])
            forecasts[i] = nxt
            history = np.append(history, nxt)
        return forecasts

    def get_params(self, deep: bool = True):
        return {**super().get_params(deep), "period": self.period,
                "representations": "series, differences, periodogram, adjusted series",
                "deseasonalised": (None if getattr(self, "figure_", None) is None
                                   else "additive" if self.additive_ else True)}


_build = mc._build_aeon_forecaster


def _build_with_drcif4(name, y, h, period):
    if name == NAME:
        return SeasonalTargetDifferenceForecaster(
            window=max(2, min(100, int(y.size) - 3)),
            regressor=DrCIF4Regressor(n_estimators=50, random_state=0, n_jobs=1),
            period=period)
    return _build(name, y, h, period)


mc._build_aeon_forecaster = _build_with_drcif4
mc.FORECASTER_ALIASES[NAME] = NAME

if __name__ == "__main__":
    mode = sys.argv.pop(1)
    if "--deseasonalise" in sys.argv:
        sys.exit("drcif4.py runs without --deseasonalise")
    if mode == "aaltd":
        import aaltd_experiments
        aaltd_experiments.main()
    elif mode == "m4long":
        import m4_long
        raise SystemExit(mc.main())
    else:
        sys.exit(f"unknown mode {mode!r}: aaltd or m4long")
