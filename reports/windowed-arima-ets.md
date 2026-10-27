# Fitting AutoARIMA and AutoETS to recent data: training windows for long series

**Status: outline and partial draft (2026-10-27).** aeon commit 8585c7bf (the
forecasting code released in aeon 1.7). Experiment notes and the running plan:
`notes/lssh99-window-experiment.md`.

The report has two parts:

1. **Experiments** showing that, on long series, the statistical forecasters are
   generally more accurate when fitted to recent data than to the whole series.
2. **An algorithm**, a forecaster wrapper (`Windower`), that chooses the training
   window automatically from the data before the forecast origin.

Out of scope for now: windowed members in a SCUM-like ensemble (parked), and
cheaper incremental updates (later work, plan part A).

## 1. Summary

(to write when Part 1 is complete)

## 2. Motivation

- Long series (thousands of points) are unlikely to follow one model with fixed
  parameters throughout: levels, seasonal profiles and dynamics drift.
  Fitting to all of the data then averages over regimes that no longer apply.
- Fitting to less data is also cheaper.
- The cost of windowing is estimation variance: on a stationary series the whole
  series is the efficient choice. A good window balances the two.
- Related work in econometrics on choosing or combining estimation windows under
  structural change (references to check before citing): Pesaran and Timmermann
  (2007); Pesaran and Pick (2011); Inoue, Jin and Rossi (2017).

## 3. Part 1: is it better not to fit all the data?

### 3.1 LSSH-99, one-step rolling origin (done)

Protocol: the paper's LSSH-99 protocol, 30 test origins, refitted at each, one
step ahead, plus 30 earlier origins for validation. Forecasters as in the paper
(AutoARIMA, ARIMA and ETS M4-deseasonalised; AutoETS seasonal). Windows 50 to
7,500 and the whole series. `scripts/lssh99_window.py`,
`scripts/lssh99_window_score.py`, results `results/aeon-8585c7bf/lssh99-window/`.

Relative MAE vs naive, geometric mean over 98 series:

| | whole series | best fixed window | rolling selection | average over windows | oracle |
|---|---|---|---|---|---|
| AutoARIMA | 0.918 | 0.846 (500) | 0.800 | 0.834 | 0.745 |
| AutoETS | 0.927 | 0.859 (1000) | 0.808 | 0.818 | 0.734 |
| ARIMA (ARMA(1,1)) | 0.932 | 0.862 (300) | 0.823 | 0.881 | 0.787 |
| ETS (SES) | 1.023 | 1.005 (200-300) | 1.006 | 1.014 | 0.990 |

- Windows of 300-1,000 points beat the whole series by 6-8% (no single window
  significant after Holm correction over 13 windows).
- Rolling selection (window with the lowest error over the previous 30 origins)
  gains 12-13% for AutoARIMA and AutoETS; averaging over windows 9-12%,
  significant for both, about 70% of series better.
- The gain is almost all on sub-daily series: hourly (AutoARIMA 0.774 -> 0.549)
  and minutely wind (whole series worse than naive; windows about 0.63).

Caveats to resolve before writing up: hourly series are not deseasonalised
below 300 points (M4 test limit of 10 log10 n lags); the minutely wind result
(5 series, one dataset) needs inspecting.

### 3.2 M4, series with at least 500 training points (running)

5,070 series (2,946 daily, 1,403 monthly, 414 hourly, 252 weekly, 43 quarterly,
12 yearly). Standard M4 protocol: one origin, competition horizons, OWA against
Naive2 on the same series. Window chosen on the M4 holdout (last h training
points). `scripts/m4_window.py`, `scripts/m4_window_score.py`, results
`results/aeon-8585c7bf/m4-window/`. By frequency and by length band
(500-999, 1,000-1,999, 2,000+).

(results to come)

### 3.3 Simulated controls (to do)

To show when windowing helps and when it costs:

- stationary processes (horizon's ARIMA/ETS processes at long lengths): the whole
  series should win; measures the variance cost of each window;
- processes with a structural break or slowly drifting parameters: a window
  should win, by an amount that depends on the break size and position.

### 3.4 Cost

Median fit time per window (already recorded): a 500-1,000-point window is 2-4x
cheaper than the whole series on LSSH-99.

## 4. Part 2: choosing the window automatically

### 4.1 The `Windower` wrapper (design, not built)

`Windower(forecaster, windows=None, strategy="select", n_validation=30)`:

- **candidates**: a geometric grid of windows from a minimum length (enough for
  the model, e.g. a few seasonal periods) up to the whole series;
- **`strategy="select"`**: in `fit`, a rolling-origin validation over the last
  `n_validation` points of the training series: at each validation origin fit a
  clone to each candidate window, record the one-step error; choose the window
  with the lowest mean error; refit on that window. `predict(y)` uses the last
  `w*` points of `y`;
- **`strategy="average"`**: fit a clone to each candidate window; forecast the
  mean (no selection, no validation fits);
- deseasonalisation: decide on the whole series, window only the fit (avoids the
  test switching off on short windows).

### 4.2 Cost and cheaper variants

Selection as above costs `n_validation x len(windows)` fits per `fit` (e.g. 30 x
10). Options to evaluate: validate every s-th origin; fix the model structure
from one AutoETS/AutoARIMA search and re-estimate parameters only; warm starts.

### 4.3 Evaluation

The wrapper run as an ordinary forecaster on LSSH-99, the M4 long series and the
simulated controls: accuracy against the whole series and against the offline
rolling/average results of Part 1, and fit time.

## 5. Open questions

- Does selection on multi-step holdout errors (M4) behave like selection on
  one-step errors (LSSH-99)?
- Minimum window per model and seasonal period.
- Is a forgetting factor (exponential down-weighting) better than a hard window?
