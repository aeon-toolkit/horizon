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

### 2.1 Related work (literature search 2026-10-27)

Checked against abstracts and bibliographic records; full texts were not read,
so verify details before citing.

**Choosing or combining estimation windows (econometrics).** The idea is well
established for regression and AR models of macroeconomic and financial series:

- Pesaran and Timmermann (2007, J. Econometrics 137:134-161): the window trades
  the bias of using pre-break data against the variance of a shorter sample;
  they propose cross-validation to choose a single window, and weighting or
  pooling of forecasts from windows of different lengths.
- Pesaran and Pick (2011, JBES 29:307-318), "Forecast combination across
  estimation windows": averaging one model's forecasts over windows lowers bias
  and RMSFE for all but the smallest breaks (equity index futures). Applied to
  SARIMA for monthly air travel by Jungmittag (2016, J. Forecasting 35:373-380):
  averaged-window forecasts generally beat single windows. Assenmacher-Wesche and
  Pesaran (Swiss VECX* models; NIER, SNB WP 2008-03): averaging over windows is at
  least as effective as averaging over models, the two complement each other, and
  alternative weighting schemes add little over equal weights.
- Clark and McCracken (2009, Int. Econ. Rev. 50(2)): combining recursive
  (expanding) and rolling forecasts is at least as accurate as either; the
  expanding window is often hard to beat.
- Inoue, Jin and Rossi (2017, J. Econometrics 196:55-67): choose the rolling
  window by minimising estimated forecast loss, analogous to bandwidth selection.
- Tian and Anderson (2014, IJF 30:161-175): weight windows by the probability
  that each point is the most recent break (reversed-order CUSUM).
- Down-weighting instead of a hard window: Pesaran, Pick and Pranovich (2013,
  J. Econometrics 177:134-152) derive MSFE-optimal observation weights (close to
  exponential under continuous breaks); Giraitis, Kapetanios and Price (2013,
  J. Econometrics 177:153-170) choose the discount rate of rolling / exponentially
  weighted estimators by cross-validation, on 97 US macro series.
- Rossi and Inoue (2012, JBES 30:432-453): evaluation that is robust to the window,
  since repeating tests over windows invites data snooping (relevant to how we
  report fixed-window results).

**Outside econometrics.**

- Petersen et al. (2026, Br. J. Math. Stat. Psychol. 79:341-361): 21 training
  windows for OLS and random forests on non-stationary experience-sampling data.
  The simple average over windows and an ensemble super learner did best;
  selecting a single window by cross-validation was poor; the expanding window was
  best only when coefficients were stable. Close to our finding that averaging is
  more reliable than per-series selection.
- Hyndman and Athanasopoulos, FPP3, "Very long and very short time series":
  "Most time series models do not work well for very long time series"; for
  short-horizon forecasts "one simple approach is to throw away the earliest
  observations and only fit a model to the most recent observations". Advice, no
  method for choosing how many.
- Concept drift (machine learning): ADWIN (Bifet and Gavalda 2007) adapts a window
  online by detecting changes in the mean; aimed at drift detection, not forecast
  accuracy. Cerqueira et al.'s arbitrated / dynamic ensembles weight forecasters
  over time; different models, not windows.

**Combination methods.**

- The forecast combination puzzle (Stock and Watson 2004; Smith and Wallis 2009;
  Claeskens et al., ~2016): estimated weights rarely beat equal weights because
  of estimation error. This predicts that fit-based weights over windows will not
  beat the plain mean or median by much.
- Jose and Winkler (2008, IJF 24:163-169): trimmed (10-30%) and Winsorised means
  slightly more accurate and more robust than the mean; supports the median /
  trimmed mean over windows.
- Kolassa (2011, IJF 27:238-251): Akaike weights over ETS models beat selection by
  an information criterion (M1, M3), as in `smooth::es(model="CCC")`. Across
  models on one sample; AIC is not comparable across windows of different length,
  hence our per-point normalisation.
- Bergmeir, Hyndman and Benitez (2016, IJF 32:303-312): bagged ETS (STL + block
  bootstrap) beats ETS on M3; Petropoulos, Hyndman and Bergmeir attribute the gain
  mainly to model uncertainty (different models selected per bootstrap series).
  Window averaging likely gains partly the same way: AutoETS/AutoARIMA select
  different models on different windows.

**What appears to be new here** (no prior work found; absence of evidence only):

- a large-scale evaluation of window selection and averaging for *automatic*
  ETS and ARIMA (model selection on each window) on benchmark long series
  (LSSH-99, the long M4 series), rather than fixed regression/AR models on macro
  series;
- the median across windows (prior work averages with the mean or weights);
- the interaction with M4-style deseasonalisation and seasonal minimum lengths
  (short windows switch the seasonality test off);
- a reusable wrapper in a toolkit: no forecasting library found offers
  combination across estimation windows as a component.

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

Run 18:35-18:57, 4 x 87,142 fits, none failed. OWA against Naive2 on the same
5,070 series (below 1: better than Naive2).

**All windows (50 to 7,500).** Short windows are harmful on hourly series:
below 300 points the M4 seasonality test no longer covers lag 24, so the
non-seasonal models are fitted without deseasonalising (AutoARIMA hourly OWA
2.3-2.7 at 50-200 points), and AutoETS fits seasonal models to 50 points (8.1).
These windows also wreck the average over windows (AutoARIMA 0.975, AutoETS
1.027, worse than the whole series). Results with all windows:
`results/aeon-8585c7bf/m4-window/`.

**Windows of at least 300 points** (`results/aeon-8585c7bf/m4-window-min300/`):

| | | Yearly | Quart. | Monthly | Weekly | Daily | Hourly | All | better / worse than whole series |
|---|---|---|---|---|---|---|---|---|---|
| AutoARIMA | whole series | 1.019 | 0.998 | 0.917 | 0.878 | 1.007 | 0.692 | 0.920 | |
| | validation | 1.010 | 0.990 | 0.876 | 0.860 | 1.007 | 0.578 | 0.897 | 2,054 / 1,730 |
| | average | 1.008 | 0.995 | 0.879 | 0.841 | 0.999 | 0.631 | 0.897 | 2,929 / 2,055 |
| | oracle | 0.972 | 0.946 | 0.798 | 0.715 | 0.912 | 0.530 | 0.812 | |
| AutoETS | whole series | 0.990 | 0.939 | 0.869 | 0.963 | 1.002 | 0.788 | 0.945 | |
| | validation | 0.955 | 0.945 | 0.857 | 0.928 | 0.999 | 0.768 | 0.934 | 1,898 / 1,739 |
| | average | 0.987 | 0.936 | 0.848 | 0.947 | 0.999 | 0.795 | 0.935 | 2,773 / 2,210 |
| | oracle | 0.948 | 0.890 | 0.755 | 0.838 | 0.956 | 0.595 | 0.849 | |

Series: 12 yearly, 43 quarterly, 1,403 monthly, 252 weekly, 2,946 daily,
414 hourly. Better / worse: series with lower / higher test MASE than the whole
series; all four differences are significant (Wilcoxon, p <= 0.001).

- Overall the gain is small: 2.5% for AutoARIMA, 1% for AutoETS (ARIMA 3%;
  SES none). It is concentrated where LSSH-99 found it: hourly (AutoARIMA 16%
  with validation selection), monthly (4-5%) and weekly (2-4%).
- Daily series, 58% of the subset, show no effect for any method. M4 daily series
  are close to random walks (Naive2 is hard to beat), so the window cannot help.
- Selection on the single M4 holdout (h points) and averaging give similar
  gains here; on LSSH-99 averaging was more reliable than a per-series validation
  choice. The oracle shows large headroom, but with one multi-step test period per
  series it is very optimistic.
- Smaller than on LSSH-99 (12-13%): the M4 subset is shorter (median training
  length 1,430, quartiles 700 and 4,196, against 1,008-10,000 on LSSH-99), multi-step, and dominated by daily series.

Consequences for the algorithm: a minimum window (here 300; in general enough for
the seasonal test and for seasonal models, e.g. several periods), and the
deseasonalisation decision taken on the whole series.

### 3.2a Mean or median over windows

The averages above are means. The median of the window forecasts (each step,
for M4) is as good or better, and robust to bad windows:

| | LSSH-99 rel. MAE: whole / mean / median | M4 OWA, windows >= 300: whole / mean / median | M4 OWA, all windows: mean / median |
|---|---|---|---|
| AutoARIMA | 0.918 / 0.834 / **0.813** | 0.920 / 0.897 / 0.899 | 0.975 / **0.905** |
| AutoETS | 0.927 / 0.818 / **0.804** | 0.945 / 0.935 / **0.930** | 1.027 / **0.926** |
| ARIMA | 0.932 / 0.881 / **0.846** | 1.056 / 1.035 / 1.035 | 1.196 / 1.064 |
| ETS (SES) | 1.023 / 1.014 / 1.010 | 1.337 / 1.336 / 1.337 | 1.336 / 1.336 |

- On LSSH-99 the median beats the mean for every forecaster, and for AutoETS
  (0.804) it beats rolling selection (0.808); 73 of 98 series improve on the whole
  series (Holm p < 0.001). For ARMA(1,1) it becomes significant (Holm p = 0.028).
- On M4 with all windows the mean is ruined by the short windows on hourly
  series (AutoETS 1.027) while the median is not (0.926, better than the mean
  with the windows restricted to >= 300). The median does not need the minimum
  window to be chosen well.

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
