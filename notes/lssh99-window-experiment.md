# ARIMA and ETS on long series: fit to a window, not the whole series?

Started 2026-10-27. aeon 8585c7bf (the 1.7 forecasting code). Not part of the
paper's fixed results.

## Questions

1. Do the statistical forecasters forecast LSSH-99 better when fitted to the last
   `w` observations than to the whole series (up to 10,000 points)?
2. Can `w` be chosen automatically, from data before the forecast origin, so that
   it beats the whole series?

## Protocol

- `scripts/lssh99_window.py`: AutoARIMA, AutoETS, ARIMA, ETS (aeon defaults, as
  in the paper; AutoARIMA, ARIMA and ETS M4-deseasonalised, AutoETS seasonal with
  the Monash period), all 100 LSSH-99 series.
- Last 60 origins, refitted at each, one step ahead. Origins 0-29 are validation,
  30-59 are the paper's 30 test origins. At window 0 (whole series) the test
  forecasts are identical to the paper's runs (checked in the pilot).
- Windows 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500 and the
  whole series. A window at least as long as the history is the whole series.
- Each worker compiles the numba code before its first timed fit.
- Raw rows: `/local/ajb2u23/Results/Forecasting/lssh99-window-8585c7bf/`;
  run script `scratch/run_lssh99_window.sh`, log `scratch/run_lssh99_window.log`.

## Methods compared (`scripts/lssh99_window_score.py`)

Relative MAE (vs naive) on the test origins, geometric mean over series; ratio to
the whole series; series better/worse; Wilcoxon signed-rank against the whole
series, Holm-corrected per forecaster.

- fixed windows, and the whole series (baseline);
- `validation`: per series, the window with the lowest validation MAE;
- `rolling`: per test origin, the window with the lowest MAE over the previous 30
  origins;
- `global`: one window for all series, chosen on the validation origins;
- `average`: mean forecast over all windows (combination across estimation
  windows, no selection);
- `oracle`: best window per series on the test origins (upper bound only).

## Related work (references to check before citing)

- Pesaran and Timmermann (2007), selection of the estimation window in the
  presence of breaks, Journal of Econometrics.
- Pesaran and Pick (2011), forecast combination across estimation windows,
  Journal of Business & Economic Statistics.
- Inoue, Jin and Rossi (2017), rolling window selection for out-of-sample
  forecasting with time-varying parameters, Journal of Econometrics.

## Caveats

- Timings are measured with other jobs on the machine (GPU runs, DrCIF), so
  absolute times are noisy; relative times across windows are what matter.
- A short window can change which models AutoETS selects, not just their
  parameters: in the pilot, solar (period 144) gave a seasonal AAA model at 1000
  and 2000 points but non-seasonal AAN at 500. The rule aeon applies has not been
  checked. The selected model is recorded per row (`model`), so this can be
  analysed.

## Results (2026-10-27)

Run 17:35-18:24, 4 x 63,187 fits, none failed. Scores:
`results/aeon-8585c7bf/lssh99-window/{summary,per_problem,selected_windows}.csv`
(`scripts/lssh99_window_score.py`). Relative MAE vs naive on the 30 test origins,
geometric mean over 98 series; ratio = geometric mean of method / whole series.

| | whole series | best fixed w | rolling selection | average over windows | oracle |
|---|---|---|---|---|---|
| AutoARIMA | 0.918 | 0.846 (w=500) | 0.800 (ratio 0.872) | 0.834 (0.909; 66 better / 32 worse, Holm p = 0.002) | 0.745 |
| AutoETS | 0.927 | 0.859 (w=1000) | 0.808 (0.871; Holm p = 0.019) | 0.818 (0.882; 68 / 30, Holm p < 0.001) | 0.734 |
| ARIMA (1,1) | 0.932 | 0.862 (w=300) | 0.823 (0.883) | 0.881 (0.945) | 0.787 |
| ETS (SES) | 1.023 | 1.005 (w=200-300) | 1.006 | 1.014 | 0.990 |

`average` counts each window fitted once (windows longer than the history are
not run); an earlier scoring that filled them with the whole series gave
0.835 / 0.816 / 0.868 / 1.014.

1. **Q1: yes for AutoARIMA, AutoETS and ARMA(1,1)**: windows of about 300-1000
   points beat the whole series by 6-8% overall. No single fixed window is
   significant after Holm correction (13 windows x 98 series). SES gains 1-2%.
2. **Q2: yes**. Choosing the window at each origin from the previous 30 one-step
   errors (`rolling`) gains 12-13% for AutoARIMA and AutoETS (significant for
   AutoETS). Averaging the forecasts over the windows gains 9-12% for AutoARIMA
   and AutoETS and is the most reliable (significant for both, ~70% of series
   better); for ARMA(1,1) it gains 5% (not significant). Per-series choice
   on a fixed validation block is weaker (11-12%, not significant). The oracle
   (best window per series, chosen on the test origins) gains 16-21%, so room
   remains.
3. **Where**: almost all of the gain is on sub-daily series, above all hourly
   (period 24, 21 series: AutoARIMA 0.774 -> 0.549 rolling) and minutely wind
   (1440, 5 series: whole series worse than naive, 1.05-1.49, windows ~0.63, even
   SES 0.999 -> 0.628). On daily/weekly/monthly the gains are small (rolling helps
   monthly and weekly a little). Likely reason for hourly: the deseasonalised
   models estimate the daily profile from the window, i.e. locally, not averaged
   over 10,000 hours.
4. **Cost**: a 500-1000 point window is 2-4x cheaper per fit than the whole series
   (AutoARIMA median 0.20-0.35 s vs 0.84 s; AutoETS 0.33-0.53 s vs 1.10 s), but
   rolling selection and averaging as run here fit all 13 windows. Hence the plan
   (part A) to do this without full refits.
5. On LSSH-99 these would move AutoARIMA (0.800) and AutoETS (0.808) from 0.92-0.93
   to ahead of SCUM (0.843) and level with the best ML forecasters (DrCIF 0.781,
   RandF (Δy) 0.791). Not a fair league comparison: the other forecasters were
   not given windows.

**Caveats**
- Below 300 points the hourly series are not deseasonalised (the M4 test checks
  periods up to 10 log10 n lags: 23 at 200 points, 24.8 at 300), so the
  window < 300 results on hourly data mix two effects. From 300 up the
  deseasonalisation decision mostly matches the whole series.
- The minutely wind result (SES 37% better than naive one step ahead) is
  surprising and comes from 5 series of one dataset: inspect before relying on it.
- Selection uses 30 one-step errors; the per-series validation choice is noisy.

## Plan and ideas (user thinking aloud, 2026-10-27; none started)

Order: get the window results first (above). If no window beats the whole series,
the ideas below matter only for cost, not accuracy.

### 0. New component: windowed forecasters in a SCUM-like ensemble (user, 2026-10-27)

**PARKED (user, 2026-10-27):** focus first on better AutoETS/AutoARIMA by window
selection, written up as a technical report (`reports/windowed-arima-ets.md`):
(1) experiments that fitting all the data is generally worse, (2) an algorithm
that chooses the window, i.e. up to the `Windower` wrapper.

Goal: a stats ensemble like SCUM whose members are fitted to recent windows.
Design sketch (not built):
- a training-window wrapper, e.g. `TrainWindowForecaster(forecaster, window)`,
  that fits and predicts with the last `window` points (0/None: whole series);
- then the existing `EnsembleForecaster` (mean / median) composes them, so no new
  ensemble class is needed: `EnsembleForecaster([TrainWindowForecaster(AutoETS(), w)
  for w in windows], averaging_method="mean")` is the `average` method of the
  experiment; a SCUM variant takes the median over members x windows, or the median
  of the members' window averages.
- `rolling` selection needs stored past one-step errors per window, so it is a
  stateful forecaster, harder to express in fit/predict; averaging is simpler and
  almost as good on LSSH-99.
Offline test before building: run AutoCES and DOTM through `lssh99_window.py`
(AutoETS and AutoARIMA are done), then form SCUM-like combinations from the stored
forecasts (median over members of full-series forecasts should reproduce SCUM;
check against the paper's SCUM run). Same on the M4 long series (`m4_window.py`).

### A. Choose the window without refitting the whole model

1. **Fit once, roll the state forward.** With fixed parameters each new point
   costs one filter step. aeon `ARIMA._predict(y)` already applies fitted phi/theta
   to a new series; aeon `ETS._predict` ignores `y` and returns the stored
   `forecast_`, so ETS needs a filter-only predict. Experiment: estimate once, roll
   forward, compare with refitting at every origin (accuracy and time).
2. **One pass scores every window (ETS).** For fixed theta the one-step errors on
   the last w points barely depend on where the filter started (start-up effect
   decays like (1 - alpha)^w), so one forward pass plus suffix sums gives the
   windowed loss for every w. Over a grid of theta: loss table L(theta, w), argmin
   per w, no Nelder-Mead per window. Select the model structure once or rarely;
   re-estimate only parameters.
3. **Incremental parameter updates.** AR terms: recursive least squares (exact,
   ~p^2 per point; sliding window by downdating, or a forgetting factor). MA terms:
   Hannan-Rissanen (long AR, then regression; linear, so RLS-friendly) or
   recursive prediction-error methods. ETS: warm-start Nelder-Mead from the
   previous origin with a small simplex; measure evaluations saved.
4. **Forgetting factor instead of a hard window**: one continuous parameter
   lambda, naturally incremental, tunable on past one-step errors. Caution: aeon
   TVP (Kalman, time-varying AR) was poor on LSSH-99 (rel. MAE 1.060).
5. Selecting w from 30 one-step errors is noisy: compare with the average across
   windows (already in the scorer). If averaging matches selection, maintain a few
   windows incrementally and average.

### B. Strip back to simple fixed models first

Test the cheap schemes on the simplest models before AutoETS/AutoARIMA:
- ETS defaults = SES, ETS(A,N,N), one parameter: grid over alpha (e.g. 50 values),
  one pass each, exact best alpha for every window.
- ARIMA defaults = ARMA(1,1), no constant, on the deseasonalised series: tests
  Hannan-Rissanen + RLS against CSS.
- AR(1), AR(2): exact sliding-window RLS; add to `lssh99_window.py` by building
  aeon `ARIMA(p, q=0)` directly (do not add names to `m_comp_experiments.py` while
  the GPU jobs import it).
Then check whether gains carry over to AutoETS/AutoARIMA.

### C. Long AR as a substitute for ARMA

Question: is a long AR(k) equivalent to ARMA asymptotically, and faster to fit?
Experiment sketch: on LSSH-99 (long; asymptotics favourable) and M4 (short;
contrast), AR(k) fitted by least squares (closed form), k chosen by AIC over
1..K via Levinson-Durbin (all orders in one pass) or fixed k in {5, 10, 20, 50},
against ARMA(1,1) and AutoARIMA; same protocol; accuracy and fit time.
Note: aeon's `ARIMA(p, 0, 0)` estimates a pure AR by Nelder-Mead on CSS
(`_fit_arima_parameters`, 200 iterations), not least squares, so a long AR in aeon
is currently slow and probably badly converged. Conditional least squares for a
pure AR is exactly OLS: a closed-form path for q = 0 is a candidate aeon change.

Existing aeon pieces (checked at 8585c7bf): `ARCoefficientTransformer`
(collection transformer, Burg via `statsmodels.regression.linear_model.burg`, so a
statsmodels dependency, not native); `AutoCorrelationSeriesTransformer` (native
ACF); `_is_stationary_ar` in `forecasting/utils/_loss_functions.py` (Levinson-Durbin
step-down). An `ARForecaster` (user: "should be easy", 2026-10-27) needs: estimate
(OLS = CSS, Burg, or Yule-Walker via Levinson-Durbin from the ACF; Yule-Walker is
biased near a unit root, Burg/OLS better for forecasting), order by AIC over 1..K
(Levinson-Durbin gives all orders in one pass), predict = dot product with the
last k values, so `predict(y)` rolls forward with no refit. Native numba Burg
would drop the statsmodels dependency of the transformer too.

`AutoCorrelationSeriesTransformer` (`aeon/transformations/series/_acf.py`, the
user's, PR #1380) is native numba, but each lag is a Pearson correlation of
`x[:-k]` with `x[k:]`, each segment with its own mean and variance. That is not
the standard (biased) ACF, which uses one mean and divides by n * c0; only the
standard one guarantees a positive definite Toeplitz matrix, so Levinson-Durbin
on this ACF could give non-stationary AR coefficients. For Yule-Walker use the
standard ACF (or Burg / OLS). Also: its docstring documents an `adjusted`
parameter that `__init__` does not have, and `StatsModelsACF` / `StatsModelsPACF`
in the same file are deprecated "for removal in v1.7.0".
The user may build the AR forecaster in a separate conversation.
