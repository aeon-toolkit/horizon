"""Interval forests on levels: the next value from a window of 100 values.

The level counterparts of ``DrCIFDeltaY``, ``CIFDeltaY`` and ``TSFDeltaY``
(same regressors and settings: 50 trees, seed 0, one thread), as
``RandF`` is of ``RandFDeltaY``: inputs are windows of 100 consecutive values
(fewer on short series), the target is the next value, and forecasts recurse.

Importing this module registers ``DrCIFLevels``, ``CIFLevels`` and
``TSFLevels`` with ``m_comp_experiments`` (also as deseasonalised forecasters,
like the other machine learning forecasters). Run as ``aaltd_experiments.py``:

    PYTHONPATH=<aeon worktree> python scripts/levels_regressors.py --deseasonalise \\
        --forecasters DrCIFLevels --n-jobs 32 --out <dir>
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import m_comp_experiments as mc  # noqa: E402

LEVELS = {"DrCIFLevels": "DrCIFDeltaY", "CIFLevels": "CIFDeltaY", "TSFLevels": "TSFDeltaY"}


class LevelRegressionForecaster(mc.TargetDifferenceRegressionForecaster):
    """Regress the next value on a window of levels and forecast recursively."""

    def fit(self, y):
        super().fit(y)
        # the base class adds the prediction to the last value; here it is the value
        last = mc._as_float_array(y)[self.window - 1 : -1]
        self.fitted_values_ = self.fitted_values_ - last
        self.residuals_ = mc._as_float_array(y)[self.window:] - self.fitted_values_
        return self

    def _training_data(self, y):
        X = np.lib.stride_tricks.sliding_window_view(y, self.window)[:-1]
        return X, y[self.window:]

    def iterative_forecast(self, y, prediction_horizon: int) -> np.ndarray:
        history = mc._as_float_array(y)
        forecasts = np.empty(int(prediction_horizon), dtype=np.float64)
        for i in range(int(prediction_horizon)):
            x = self._x(history[-self.window:].reshape(1, -1))
            forecasts[i] = float(self.regressor_.predict(x)[0])
            history = np.append(history, forecasts[i])
        return forecasts

    def get_params(self, deep: bool = True):
        return {**super().get_params(deep), "target": "next value"}


_build = mc._build_aeon_forecaster


def _build_levels(name, y, h, period):
    if name in LEVELS:
        if name == "DrCIFLevels":  # as DrCIFDeltaY in _build_aeon_forecaster
            from aeon.regression.interval_based import DrCIFRegressor
            regressor = DrCIFRegressor(n_estimators=50, random_state=0, n_jobs=1)
        else:
            module, cls, kwargs = mc.DELTAY_REGRESSORS[LEVELS[name]]
            regressor = getattr(importlib.import_module(f"aeon.regression.{module}"), cls)(
                **kwargs)
        return LevelRegressionForecaster(
            window=max(2, min(100, int(y.size) - 3)), regressor=regressor)
    return _build(name, y, h, period)


mc._build_aeon_forecaster = _build_levels
mc.FORECASTER_ALIASES.update({n: n for n in LEVELS})
mc.DESEASONALISED_AEON_FORECASTERS = mc.DESEASONALISED_AEON_FORECASTERS | set(LEVELS)

if __name__ == "__main__":
    import aaltd_experiments
    aaltd_experiments.main()
