# SARIMA / AutoSARIMA: review and M4 results (aeon c5615dd85)

From the interval-based session in `~/Code/aeon`, 2026-10-31. It follows
`SARIMA_HANDOVER.md` (on `origin/main`; this checkout had not merged it when
this note was written). The review is of `ajb/sarima` at **c5615dd85**. The
M4 runs below go further than the handover's planned runs: fixed orders on all
of M4, and AutoSARIMA on all of M4.

**Summary.** Fixed-order `SARIMA` is equivalent (TOST ±5%) to statsforecast and
to the published M4 ARIMA on every frequency. `AutoSARIMA` is equivalent to the
published ARIMA overall and better on hourly, but on monthly it is behind both R
and aeon's existing non-seasonal `AutoARIMA`. Changes 2 and 3 below are the
likely causes; change 1 is needed for the paper's "same objective" wording.

## Changes to make, in priority order

### 1. Condition only on the AR lags, as R's CSS does

`_sarima_lags` (`aeon/forecasting/utils/_extract_paras.py`) returns
`start = max(largest AR lag, largest MA lag)`, and `_sarima_model` /
`_arima_css_aic` drop the first `start` residuals. R's `arima(method = "CSS")`
(and statsforecast's port) condition on `p + P*m` only, with pre-sample
residuals set to 0 for the MA recursion. For (0,1,1)(0,1,1)12 aeon drops 13
extra observations, which is a lot on M4 monthly (minimum length 42).

Evidence, from all of M4 at statsforecast's orders. "Extra" is the number of
residuals aeon drops beyond R's `p + P*m`. The difference is the largest
absolute ARMA-coefficient difference between aeon and statsforecast's CSS fit.

| group | series | median difference | difference > 0.2 |
|---|---|---|---|
| no extra | 36,302 | 0.000 | 4.5% |
| extra <= 20% of the differenced series | 40,445 | 0.029 | 12.2% |
| extra > 20% | 1,463 | 0.178 | 44.2% |

With the same conditioning the estimates match R exactly. Accuracy is already
equivalent, so this is about the claim of an identical objective, and about
efficiency on short seasonal series. Suggested change: start the recursion at
the largest AR lag (residuals before it stay 0), and count the variance over
`n - start_ar`. `_valid` in `AutoSARIMA` and `_difference_checked` use the
same `start` for their length checks, so update those too.

Main's `ARIMA` uses `max(p, q)` in the same way. Changing ARIMA would change
its existing results, so treat it as a separate decision. For non-seasonal
`q <= 2` the effect is small.

### 2. AutoSARIMA: no constant when d + D >= 2

`AutoSARIMA._search` tries `constant_term in [0, 1]` at every neighbour,
whatever `d + D` is. R allows a mean only when `d + D == 0` and drift only when
`d + D == 1`. With `d + D == 2`, a constant puts a quadratic trend into the
forecasts.

Evidence, from 1,000 monthly M4 series (seed 0): 21 fits have a constant with
`d + D == 2`. Their mean MASE is 1.356, against 1.020 for R and 1.014 for
statsforecast. The 10 series with the largest gap to R account for 62% of the
whole monthly gap. In `_valid` or `_search`, reject `constant_term == 1` when
`self.d_ + self.D_ >= 2`.

### 3. AutoSARIMA: AICc, and R's order limit

R's `auto.arima` uses AICc and `max.order = 5` (p + q + P + Q <= 5). aeon uses
AIC with no total limit. In the same monthly sample, the 213 fits with
p + q + P + Q > 5 have mean MASE 0.854, against 0.803 for R. Try both,
separately, and rerun the M4 AutoSARIMA (about 6 minutes on 40 workers). AICc
was already an untested candidate for the yearly AutoARIMA gap (README).

AutoSARIMA picks the same D as statsforecast on 96-100% of series and the same
d on 97-100%, but the same full order on only 18-39%. So the differences are in
the ARMA search, not in the differencing.

### 4. statsforecast CSS reference (handover issue 1)

The problem persists in statsforecast 2.1.1. `ARIMA(..., method="CSS")` gives
correct coefficients but wrong forecasts (zeros, or the wrong level). It is not
an aeon problem. A working reference: take the CSS coefficients and pass them
as `fixed=coef` (the dict) to `ARIMA(..., method="CSS-ML")`, which then only
runs the Kalman filter to forecast. `m4_sarima.py` does this (`fit_sf_css`).

### 5. Speed

These numbers are the baseline for the other speed-ups. Single thread, median
seconds per series, fit plus 24-step forecast, simulated airline-model series
(m = 12):

| n | aeon SARIMA | sf ARIMA CSS | sf ARIMA CSS-ML | aeon AutoSARIMA | sf AutoARIMA |
|---|---|---|---|---|---|
| 100 | 0.0008 | 0.0018 | 0.0129 | 0.079 | 0.67 |
| 1,000 | 0.0028 | 0.0025 | 0.0887 | 0.31 | 1.64 |
| 10,000 | 0.0271 | 0.0154 | 0.8501 | 1.21 | 27.1 |

statsforecast's CSS fit is faster than aeon's SARIMA from 1,000 observations (1.7x at 10,000). Each aeon fit runs Nelder-Mead
three times (default simplex, Hannan-Rissanen, restart), so that is the place
to look. AutoSARIMA's time depends on the search path: n = 10,000 was faster
than 5,000 on these series. On M4, the median AutoSARIMA time per series was
0.62 s hourly and 0.14 s monthly, under 40 workers. Change 2 also roughly
halves the neighbour fits when `d + D >= 2`.

## M4 results at c5615dd85

OWA (lower is better). "Fixed" fits each series at the order, mean and drift
that statsforecast's AutoARIMA recorded for it (`arma` and `coef` in its result
file).

| | fixed SARIMA | sf ARIMA CSS (same order) | sf AutoARIMA | AutoSARIMA | aeon AutoARIMA (deseasonalised) | M4 ARIMA (R) |
|---|---|---|---|---|---|---|
| Yearly | 0.907 | 0.907 | 0.906 | 0.919 | 0.918 | 0.892 |
| Quarterly | 0.902 | 0.903 | 0.900 | 0.910 | 0.894 | 0.898 |
| Monthly | 0.906 | 0.904 | 0.904 | 0.929 | 0.918 | 0.903 |
| Weekly | 0.876 | 0.872 | 0.868 | 0.924 | 0.902 | 0.932 |
| Daily | 1.007 | 1.006 | 1.006 | 1.008 | 1.012 | 1.044 |
| Hourly | 0.581 | 0.578 | 0.578 | 0.566 | 0.692 | 0.577 |
| **All** | **0.909** | **0.908** | **0.907** | **0.925** | **0.915** | **0.903** |

TOST (±5%, geometric mean of per-series MASE ratios):

- **Fixed SARIMA** is equivalent to sf CSS, to sf AutoARIMA and to the M4 ARIMA
  on every frequency. The one exception is weekly against the M4 ARIMA:
  inconclusive at 0.952 (0.925-0.981), with aeon better. Overall ratio against
  the M4 ARIMA: 1.002.
- **AutoSARIMA** is equivalent to the M4 ARIMA on every frequency (overall
  1.017, monthly 1.016, hourly 0.988). Against sf AutoARIMA, weekly is
  inconclusive at 1.048.
- Fit failures: 0 in every run. Before the rewrite (32b8cb85), fixed SARIMA
  had 2,658 non-invertible and 1,154 non-stationary fits on M4; at c5615dd85 it
  has none.

The aeon AutoARIMA column is from `results/aeon-8585c7bf/m4`. Its weekly
figure (0.902) differs from AutoSARIMA with period 1 (0.924), and the two
should be the same algorithm. The difference may come from main's changes since
8585c7bf, but this was not checked.

## Other review points (minor)

- The seasonal root check (`margin ** m` on a polynomial in B^m) is correct. It
  is equivalent to R's check on the expanded polynomial.
- `_seasonal_strength` uses STL with seasonal window 11, which matches the
  first window of `mstl`, used by R's `seas.heuristic`. R's `mstl` iterates
  twice; this is close enough, and D agrees with statsforecast on 96-100% of
  series.
- `ARIMA._predict` undifferences from `self._series` and not from the `y`
  passed in (as the handover says). This is not fixed and not tested.

## Files and rerunning

Everything, including copies of the scripts, is in
`/local/ajb2u23/Results/Forecasting/sarima-review-c5615dd8/`:

- `full-c5615dd8/`: fixed SARIMA at sf orders, all of M4, with per-frequency
  pickles (forecasts, coefficients, times, root checks), `summary.csv` and
  `per_series.csv.gz`.
- `auto-c5615dd8/`: AutoSARIMA on all of M4 (forecasts, orders, times).
- `timing/`: run times, `median_seconds.csv`, and `timing_sarima.png`/`.pdf`.
- Scripts:
  - `m4_sarima.py`: `fit`, then `score`. Run from `~/Code` with
    `PYTHONPATH=<aeon worktree>`.
  - `auto_m4.py`: AutoSARIMA on all of M4.
  - `timing_sarima.py`: `run --method ...`, then `plot`.
  - `make_tables.py`: the markdown tables.
  - `auto_check_m.py`: the monthly breakdown for changes 2 and 3.
- The scripts import `m4_score` and `horizon.sweep` from this repository. They
  were run in the `aeon-fcbench` env with statsforecast 2.1.1, against the
  worktree `/home/ajb2u23/Code/aeon-sarima` (detached at c5615dd85), with a
  fresh numba cache.
- PR-ready tables and the figure: `~/Code/aeon/local/prs/sarima-results.md`
  and `timing_sarima.png`.

To check changes 1-3, rerun each on a new commit in a clean worktree, then
compare with the tables above:

```bash
m4_sarima.py fit --sf-css      # 2 min
OUT=auto-<commit> auto_m4.py    # 6 min
m4_sarima.py score
make_tables.py
```
