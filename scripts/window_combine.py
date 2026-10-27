"""Ways to combine the forecasts of one forecaster fitted to several training windows.

Used by ``lssh99_window_score.py`` and ``m4_window_score.py``. ``preds`` is a
(windows, steps) array of forecasts made at one origin; the scores are per
window, aligned with its rows:

- ``insample_mse``, ``insample_rel_mae``: in-sample fit per point (rel_mae: MAE
  over the in-sample naive MAE on the same points), so comparable across window
  lengths;
- ``oos_mse``: out-of-sample error of each window before this origin (LSSH-99:
  the previous 30 one-step errors; M4: the holdout), or None.

Each scheme returns one forecast per step.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import trim_mean

REFERENCE_N = 100  # length the per-point log-likelihood is scaled to (akaike)


def _weighted(preds: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() == 0:
        return preds.mean(axis=0)
    return (w[:, None] * preds).sum(axis=0) / w.sum()


def combine(preds: np.ndarray, insample_mse: np.ndarray, insample_rel_mae: np.ndarray,
            oos_mse: np.ndarray | None = None) -> dict[str, np.ndarray]:
    preds = np.atleast_2d(preds)
    out = {
        "mean": preds.mean(axis=0),
        "median": np.median(preds, axis=0),
        "trimmed": trim_mean(preds, 0.2, axis=0) if len(preds) >= 5 else np.median(preds, axis=0),
        "inv_mse": _weighted(preds, 1 / insample_mse),
        "inv_rel_mae": _weighted(preds, 1 / insample_rel_mae),
        "inv_rel_mae2": _weighted(preds, 1 / insample_rel_mae ** 2),
    }
    # Akaike-style: -2 log L per point ~ log MSE, scaled to a common length
    lm = np.log(insample_mse)
    ok = np.isfinite(lm)
    a = np.zeros_like(lm)
    a[ok] = np.exp(-0.5 * REFERENCE_N * (lm[ok] - lm[ok].min()))
    out["akaike_n100"] = _weighted(preds, a)
    r = np.where(np.isfinite(insample_rel_mae), insample_rel_mae, np.inf)
    out["best_insample"] = preds[int(np.argmin(r))]
    if oos_mse is not None:
        out["inv_oos_mse"] = _weighted(preds, 1 / oos_mse)
        o = np.where(np.isfinite(oos_mse), oos_mse, np.inf)
        top = np.argsort(o)[: max(1, len(o) // 2)]
        out["median_best_half"] = np.median(preds[top], axis=0)
    return out
