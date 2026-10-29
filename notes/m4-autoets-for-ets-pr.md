# M4 AutoETS results for the ETS PR (aeon 439ed8bc)

From the horizon session, 2026-10-21. aeon AutoETS from branch
`fix/forecasting-arima-ets-estimation` at 439ed8bc was run on all 100,000 M4
series and scored against the official M4 benchmark forecasts. Two issues are
relevant to the ETS PR. Neither showed up in the simulated benchmark.

## Setup

- Run: `scripts/m_comp_experiments.py --datasets M4 --forecasters AutoETS
  --backends aeon` (horizon repo), aeon imported from the
  `/home/ajb2u23/Code/aeon-bench` worktree at 439ed8bc. The true seasonal
  period is given (1, 4, 12, 1, 1, 24) and no deseasonalisation is applied.
- Output: `/local/ajb2u23/Results/Forecasting/aeon-439ed8bc/aeon/AutoETS/`
  (`M4_AutoETS_{Test,Train}.csv`, plus `M4_AutoETS_scores_per_series.csv` with
  per-series sMAPE/MASE for every method below).
- Scoring: `scripts/m4_score.py` in horizon, following the M4 evaluation script
  (sMAPE and MASE averaged over forecast points, OWA relative to Naive2). The
  official `submission-ETS.csv` (R `forecast::ets`) and `submission-Naive2.csv`
  score exactly the published values (ETS sMAPE 12.725, MASE 1.680,
  OWA 0.908), so the scoring is right.

## Results (OWA; aeon 439ed8bc excludes series Q5018, see issue 1)

| | R ets (M4) | aeon 439ed8bc | aeon June 2026 | statsforecast June |
|---|---|---|---|---|
| Yearly | 0.903 | 0.905 | 0.871 | 0.816 |
| Quarterly | 0.890 | 0.893 | 0.907 | 0.888 |
| Monthly | 0.914 | 0.931 | 0.942 | 0.925 |
| Weekly | 0.931 | 0.938 | 0.948 | 0.934 |
| Daily | 0.996 | 1.001 | 1.001 | 0.988 |
| Hourly | 0.852 | 0.779 | 0.788 | 0.803 |
| All | 0.908 | 0.914 | 0.907 | 0.878 |

Overall MASE 1.679 against R's 1.680; sMAPE 12.89 against 12.73. The June
runs used the older aeon code on branch `ajb/experiments`; their exact
settings were not checked, so treat the June statsforecast column with care.
Its Yearly result is much better than R's, and that is unexplained.

## Issue 1: forecasts blow up with additive error and multiplicative seasonality

Including Q5018, aeon's quarterly MASE is 32.9 instead of 1.17, and that one
series causes all of it (MASE 761,368; R ets 0.20).

Repro (series saved at `/home/ajb2u23/Code/horizon/notes/m4_Q5018_train.npy`,
the 70 training values of M4 Q5018):

```python
import numpy as np
from aeon.forecasting.stats import AutoETS
y = np.load("/home/ajb2u23/Code/horizon/notes/m4_Q5018_train.npy")
f = AutoETS(seasonal_period=4).fit(y)
# error_type_=1, trend_type_=1, damped_trend_=True, seasonality_type_=2: A,Ad,M
f.iterative_forecast(y, 8)
# [1.99e+02, 2.44e+02, 4.55e+09, 3.09e+02, 2.20e+02, 2.57e+02, 4.69e+09, 3.14e+02]
# actual: 330 370 340 390 1750 1210 470 410
```

The series runs at 3000-8000 and then drops to about 400 for its last six
points. One seasonal state explodes, so every fourth forecast is around 4.5e9.
Fitted parameters: alpha 0.300, beta 0.300 (at the beta <= alpha bound),
phi 0.561, gamma 0.140; AIC 1156.3.

R's `ets` leaves out the additive-error, multiplicative-season models (A,N,M,
A,A,M, A,Ad,M) by default (`restrict = TRUE`) because they are numerically
unstable. aeon's AutoETS includes them and chose one for 8,456 of 100,000 M4
series (8.5%: 6,084 monthly, 2,288 quarterly, 84 hourly). Apart from Q5018
these score about the same as R (monthly mean MASE 0.946 vs 0.923, quarterly
median 0.889 vs 0.877), so the cost is mainly this one catastrophic forecast.

Possible fixes: exclude A,*,M from the AutoETS search by default, as R does
(perhaps behind an option like `restrict`); and/or reject candidate fits whose
states or forecasts are non-finite or explode. The first matches the reference
implementation and also makes the search cheaper.

## Issue 2: Yearly is worse than the June aeon because fewer trends are damped

On Yearly, sMAPE is 15.59 on the branch against 14.89 for the June aeon
(OWA 0.905 vs 0.871). It is still on par with R (15.36, OWA 0.903).

| Yearly, series with a trend | branch | June |
|---|---|---|
| share of series with a trend | 84% | 74% |
| share of those damped (phi < 0.999) | 15% | 49% |
| phi, 10th / 25th percentile | 0.945 / 1.0 | 0.694 / 0.907 |

Mean sMAPE where the branch fits an undamped trend (16,516 series): branch
15.02, R ets 14.64. Where it fits a damped trend (2,885 series): 13.47 against
13.69.

This looks like the cost of fixing 1.6.0's always-estimated phi: that bug
happened to suit M4 Yearly, where damped trends are known to do well. An
example of the new behaviour is Y17486 (21 points, falling at the end). The
branch chooses M,A,N with alpha = beta = 0.98 (beta at the alpha bound) and
extrapolates a steep rise (7654 to 21291 against actuals 4129 to 2450; MASE
51.9 vs R 10.1). Things to check: whether the AIC comparison between damped
and undamped candidates matches R (parameter count, phi bounds 0.8-0.98 in R),
and the beta upper bound. In R, `ets` restricts phi to [0.8, 0.98] and uses
`bounds = "both"`. This is not investigated; it may be acceptable as it
stands.

This differs from the simulated benchmark in horizon, where the branch picked
A,Ad,N for A,N,N series more often than competitors did. Each problem may come
from a different part of the selection.

## Minor (horizon side, not aeon)

The model label in `m_comp_experiments.py` Train files drops damping (Q5018 is
labelled "AAM" but fitted A,Ad,M). The phi in `model_params_json` is right.

# Addendum (2026-10-21 12:45): M4 AutoARIMA, KPSS lag rule

aeon AutoARIMA (439ed8bc) on all M4, with M4-style deseasonalisation for
quarterly/monthly/hourly (`--deseasonalise`, checked to reproduce the official
Naive2 exactly). Output in
`/local/ajb2u23/Results/Forecasting/aeon-439ed8bc/aeon/AutoARIMA/`.

OWA: all 0.966 (statsforecast AutoARIMA June run 0.907, R ets 0.908). By
frequency: yearly 1.001, quarterly 0.920, monthly 0.950, weekly 0.894, daily
1.126, hourly 0.710. Yearly and daily are no better than Naive2. Those
frequencies have m = 1, so deseasonalisation plays no part there.

Cause: the d chosen. aeon picks d = 0 with a constant (mean-reverting) on 32% of
yearly and 9% of daily series. `AutoARIMA._fit` chooses d by repeated
`kpss_test(x)` with the default `lags = int(sqrt(n))`. R's
`forecast::ndiffs` (and statsforecast) use `urca::ur.kpss(type = "mu",
use.lag = trunc(3 * sqrt(n) / 13))`, about 4x fewer lags (yearly n = 30: aeon 5,
R 1). More lags inflate the long-run variance, so trending series pass as
stationary.

Evidence (recomputed with aeon's own `kpss_test`; the aeon-lag recomputation
matches the run's chosen d on 100% of series):

| | d=0 | d=1 | d=2 |
|---|---|---|---|
| yearly, aeon lags | 32.2% | 53.3% | 14.5% |
| yearly, `lags=trunc(3*sqrt(n)/13)` | 9.4% | 62.7% | 27.5% |
| daily, aeon lags | 9.2% | 85.6% | 5.2% |
| daily, R lags | 0.9% | 92.9% | 6.2% |

The 5,249 yearly series that aeon leaves at d = 0 but the R rule differences
have mean sMAPE 27.9 (aeon) against 20.0 (statsforecast), and they account for
94% of aeon's yearly sMAPE gap to statsforecast. The 347 such daily series
account for all of the daily gap. Where aeon picks d >= 1, it is level with
statsforecast.

Suggested fix: in `AutoARIMA._fit`, call `kpss_test(x, lags=int(3 *
np.sqrt(len(x)) / 13))`, or change `kpss_test`'s default lag rule to match
R/urca "short". Smaller differences from urca: gamma_0 divides by n - k
rather than n. Not yet verified: rerunning with the change.

History of the KPSS lag rule: unchanged since AutoARIMA was added. `kpss_test`
came in with AutoARIMA in #2861 (4b982a6cd, 2025-09-02) with
`lags = int(np.sqrt(n))`, and that line is identical in 1.6.0 (b694b1c03) and on
the fix branch (439ed8bc). The only later commit touching it, #3602
(2026-07-06), added tests (one calls `kpss_test(y, lags=5)`) without changing the
default. None of the four ARIMA commits on the fix branch touch the d
selection. So this is a pre-existing issue, not a regression from the branch.
The simulated horizon benchmark (n = 100, 500) did not expose it; M4's short
smooth yearly series do.
