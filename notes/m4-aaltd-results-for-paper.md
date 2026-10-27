# M4 and aaltd results for the paper (aeon 8585c7bf)

All runs use aeon commit 8585c7bf (worktree `aeon-bench-next`) with no aeon
code modified. Scripts: `scripts/m_comp_experiments.py` (M4),
`scripts/aaltd_experiments.py` and `scripts/aaltd_score.py` (aaltd). Run log:
`notes/unattended-run-2026-10.md`.

## M4: all 100,000 series, OWA relative to Naive2

Non-seasonal and ML/DL forecasters are fitted to M4-deseasonalised series
(classical multiplicative decomposition, 90% ACF test), then reseasonalised;
this reproduces the official Naive2 exactly. Seasonal models use the M4
frequency's period.

Source: `results/aeon-8585c7bf/m4/owa_all_forecasters.csv`.

| Forecaster | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly | **All** |
|---|---|---|---|---|---|---|---|
| SCUM | 0.804 | 0.854 | 0.872 | 0.902 | 0.983 | 0.651 | **0.845** |
| DOTM | 0.806 | 0.887 | 0.916 | 0.957 | 0.997 | 0.990 | **0.870** |
| AutoCES | 0.872 | 0.907 | 0.895 | 1.005 | 1.019 | 0.781 | **0.894** |
| Theta | 0.871 | 0.918 | 0.907 | 0.969 | 0.999 | 1.006 | **0.897** |
| AutoETS | 0.905 | 0.892 | 0.929 | 0.938 | 1.001 | 0.788 | **0.914** |
| AutoARIMA | 0.918 | 0.894 | 0.918 | 0.902 | 1.012 | 0.692 | **0.915** |
| CES | 0.872 | 0.973 | 1.011 | 1.005 | 1.019 | 4.285 | **0.963** |
| ARIMA | 1.019 | 0.986 | 0.941 | 1.082 | 1.059 | 1.208 | **0.981** |
| *Naive2* | 1 | 1 | 1 | 1 | 1 | 1 | **1.000** |
| ETS | 1.001 | 1.019 | 1.039 | 0.974 | 1.000 | 3.591 | **1.027** |
| Naive | 1.000 | 1.066 | 1.095 | 1.000 | 1.000 | 3.593 | **1.058** |
| RandF | 1.136 | 1.140 | 1.013 | 0.958 | 1.134 | **0.474** | **1.081** |
| TVP | 1.163 | 1.209 | 1.338 | 0.943 | 1.406 | 0.639 | **1.235** |
| DeepAR† | 1.107 | 1.224 | 1.567 | 1.182 | 1.047 | 0.925 | **1.303** |
| N-BEATS† | 1.085 | 1.487 | 1.664 | 1.227 | 1.420 | 1.644 | **1.397** |
| DrCIF (Δy) | 1.210 | 1.495 | 1.708 | 1.366 | 1.213 | 0.643 | **1.450** |
| TCN† | 1.172 | 0.975 | 4.706 | 0.862 | 1.064 | 1.073 | **2.066** |

Not usable as shipped (forecasts explode, mean OWA from 1e2 to 1e47): TAR,
AutoTAR, SETAR, SETARTree, SETARForest. Report as failed, with a count of
exploded series, rather than as a score.

### Findings

1. **aeon's statistical forecasters reproduce the published R results.**
   - AutoETS 0.914 against R `ets` 0.908; AutoARIMA 0.915 against
     `auto.arima` 0.903. Both are equivalent overall by TOST (±5% margin on
     the per-series MASE ratio).
   - SCUM 0.845 against the published entry 036's 0.848 (ratio 1.004). It is
     equivalent on every frequency when run with 036's seasonal periods.
   - The one exception: AutoARIMA is significantly worse on hourly, because
     aeon has no seasonal ARIMA.
   - Detail: `equivalence/README.md`, `equivalence/paper/table_m4.tex`.
2. **The combinations rank highest.** SCUM, DOTM, AutoCES and Theta all beat
   the automatic ETS and ARIMA.
3. **ML and DL forecasters all come in behind Naive2 overall**, with
   exceptions on single frequencies:
   - RandF has the best hourly score of any method (0.474).
   - TCN does well on weekly (0.862) and quarterly (0.975). Its overall 2.066
     comes from about 70 exploding monthly/yearly forecasts (error over 10x
     the series level); recursive forecasting of differences compounds errors.
4. **Five threshold/tree forecasters explode** (see above). Likely causes:
   SETAR's default lag of 10 on short series, SETARTree's default of no
   scaling, and TAR's unstable regimes.

† **Deep learners.** As shipped they do not scale their input, and failed a
600-series pilot (OWA DeepAR 26, TCN 3.3, N-BEATS 2.5). The results above use
a wrapper in the experiment script (`StandardisedForecaster`): standardise
first differences of the last 1,000 points, fit once, forecast recursively
(N-BEATS series-to-series). No aeon code was changed.

## LSSH-99 (the aaltd collection): 99 long series, 30 rolling origins, one-step forecasts

### Protocol

- Each problem is TRAIN and TEST joined, up to 10,000 points.
- Each forecaster is refitted at each of the last 30 points and forecasts one
  step ahead: 3,000 fits per forecaster.
- Seasonal periods follow the Monash conventions (half-hourly 48, hourly 24,
  daily 7, 10-minute 144, minutely 1440, weekly 52, monthly 12).
- Non-seasonal and ML/DL forecasters are deseasonalised as on M4 (additive
  decomposition for the 24 series with non-positive values).
- wind_farms T2 is constant, so scores cover 99 problems.
- **Rel. MAE**: geometric mean of MAE relative to Naive (last value).
  **Rank**: mean rank by MAE among the 23 forecasters that completed every
  problem.

Source: `results/aeon-8585c7bf/aaltd/summary.csv` (per problem:
`per_problem.csv`).

| Forecaster | Rel. MAE | Mean MASE | Mean rank | Median fit (s) |
|---|---|---|---|---|
| DrCIF (Δy) | **0.781** | 1.48 | 9.5 | 174 |
| TCN† | 0.804 | **1.36** | 8.7 | 42 |
| Ridge (Δy) | 0.833 | 1.56 | 10.4 | 0.008 |
| SCUM | 0.843 | 1.44 | **8.6** | 1.2 |
| RandF | 0.845 | 1.67 | 11.2 | 24 |
| DeepAR† | 0.896 | 1.52 | 10.1 | 48 |
| SETAR | 0.898 | 1.52 | 10.5 | 0.76 |
| AutoARIMA | 0.918 | 1.51 | 9.8 | 0.34 |
| TAR‡ | 0.924 | 1.55 | – | 0.002 |
| AutoCES | 0.924 | 1.60 | 12.5 | 0.07 |
| AutoTAR | 0.925 | 1.54 | 11.0 | 0.05 |
| AutoETS | 0.927 | 1.58 | 12.1 | 0.52 |
| ARIMA | 0.932 | 1.53 | 9.7 | 0.02 |
| DOTM | 0.968 | 1.60 | 11.6 | 0.04 |
| N-BEATS† | 0.989 | 1.72 | 14.0 | 39 |
| Theta | 0.990 | 1.59 | 11.2 | 0.002 |
| Regression (Ridge) | 0.998 | 3.07 | 13.2 | 0.09 |
| *Naive* | 1.000 | 1.61 | 11.0 | <0.001 |
| ETS | 1.023 | 1.62 | 11.4 | 0.01 |
| CES | 1.028 | 1.69 | 13.3 | 0.005 |
| TVP | 1.060 | 1.84 | 13.7 | 46 |
| SETARForest | 1.894 | 28.9 | 16.4 | 6.0 |
| SETARTree | 2.201 | 12.5 | 16.5 | 0.61 |
| Seasonal Naive | 2.385 | 6.21 | 19.7 | <0.001 |

† Same scaling/differencing wrapper as on M4.
‡ TAR fails on 11 zero-heavy series (solar, weather, wind): aeon raises a
singular-matrix error. Scored on 89 problems and left out of the ranks.

### Findings

1. **The ordering differs from M4.** With long histories and one-step
   forecasts, ML forecasters on differences come out top: DrCIF and TCN have
   the lowest relative MAE; Ridge on differences is close behind and about
   20,000 times faster to fit than DrCIF. SCUM has the best mean rank and is
   the strongest statistical method.
2. **Most statistical forecasters cluster at 0.92–0.99.** One step ahead on
   high-frequency data, Naive is hard to beat.
3. **SETAR does well here (0.898) despite exploding on M4**, consistent with
   its lag-10 default suiting long series but not short ones.
4. **Seasonal Naive is worst (2.39):** one step ahead on high-frequency
   series, the last value is far more informative than the value one period
   back.
5. **Deseasonalisation caveat:** by M4's rule (`lag.max = 10·log10 n`)
   periods above about 40 are never judged seasonal, so the half-hourly,
   10-minute, minutely and weekly series were not deseasonalised.

## Points for the method section

- aeon commit 8585c7bf, no aeon code modified.
- Deep learners: scaling/differencing wrapper and 1,000-point history cap.
- Deseasonalisation: M4-style for non-seasonal and ML/DL models; additive
  for aaltd series with non-positive values.
- Ridge (Δy) ran on aaltd only; there is no M4 result for it.
- CES on M4 is on raw series (0.963); deseasonalised it scores 0.903.

## aeon issues found (worth reporting)

- DL forecasters (TCN, DeepAR, N-BEATS) do not scale their input.
- `iterative_forecast` refits the model.
- SETAR defaults to lag 10, too long for short series; SETARTree defaults to
  no scaling.
- TAR raises a singular-matrix error when a regime is all zeros.
- TensorFlow models leak host memory across fits.
