# M4 and aaltd: investigation notes (not for the paper)

Exploratory analyses behind the paper summary (`m4-aaltd-results-for-paper.md`).
aeon 8585c7bf throughout. Raw results in `/local/ajb2u23/Results/Forecasting/`
(`aeon-8585c7bf`, `aeon-8585c7bf-dl`, `aaltd-8585c7bf`); run log in
`unattended-run-2026-10.md`.

## Why do the ML/DL forecasters do badly on M4?

Three checks, all on the full 100,000 series.

### 1. Series length (`scripts/m4_by_length.py` -> `results/aeon-8585c7bf/m4/owa_by_length.csv`)

OWA and the median per-series MASE ratio to AutoETS, binned by training length,
overall and within each frequency.

- Length helps some models within a frequency (monthly OWA, DrCIF (Δy)
  2.26 / 1.49 / 1.37 / 1.26 and DeepAR 2.38 / 1.26 / 1.02 / 0.98 from 50-99 to
  400-999 points) but not RandF (best on the shortest monthly series, 0.932).
- Even with 1000+ points every ML model trails AutoETS on daily (RandF 1.10,
  DrCIF 1.22, AutoETS 1.00).
- ML wins only on long seasonal series: hourly (RandF 0.474, DrCIF 0.643,
  AutoETS 0.788) and long weekly (TCN 0.742, RandF 0.784, AutoETS 0.955).
- Conclusion: length explains only part; the series' character (trend-
  dominated or near random walk on yearly/quarterly/monthly/daily) matters more.

**Set-up flaw found:** DrCIF (Δy)'s window is `min(100, n - 3)`, so a 60-point
series has about 3 training windows; this drives its 2.26 on short monthly
series. RandF uses `n // 4` below ~100 points and does not suffer. A rerun with
RandF's rule would give DrCIF a fairer M4 score (not done).

### 2. Forecast horizon (`scripts/m4_by_steps.py` -> `owa_by_steps.csv`)

OWA on step 1 only (a true one-step forecast from the end of training), steps
1-3, and all steps.

| | step 1 | all steps |
|---|---|---|
| SCUM | 0.902 | 0.845 |
| AutoETS | 0.965 | 0.914 |
| TCN (Δ) | 1.014 | 2.066 |
| DeepAR (Δ) | 1.078 | 1.303 |
| N-BEATS (Δ) | 1.125 | 1.397 |
| DrCIF (Δy) | 1.154 | 1.450 |
| RandF | 1.157 | 1.081 |
| SETAR | 270.7 | 5.6e46 |

- Recursion accounts for about two-thirds of the gap to AutoETS for the
  deep learners and DrCIF, and all of TCN's (level with AutoETS at step 1).
- None of RandF's: it is as far behind at step 1 (trees on levels cannot
  extrapolate a trend).
- One step ahead, ML wins on hourly (RandF 0.629, DrCIF 0.736 vs AutoETS
  0.805) and weekly (DrCIF 1.039, TCN 1.065 vs stats 1.21-1.30, AutoARIMA
  1.060); all ML models are worse than Naive on yearly.
- SETAR/SETARTree/SETARForest are broken at step 1 (yearly 625): the fit
  fails, not just the recursion.
- The scorer warned "Mean of empty slice": some forecasts contain NaN, which
  the per-series means skip. Not yet traced.

### 3. Equal weight per frequency (`owa_equal_frequency_weights.csv`)

Pooled OWA is dominated by yearly/quarterly/monthly (95% of series); hourly
and weekly are under 1%. Mean of the six per-frequency OWAs (arithmetic;
geometric in the CSV):

- All steps: SCUM 0.844, AutoARIMA 0.889, AutoETS 0.909, AutoCES 0.913,
  DOTM 0.925, Theta 0.945, RandF 0.976 (pooled 1.081), DeepAR 1.175,
  DrCIF 1.273, TCN 1.642. CES, ETS and Naive drop (hourly ~3.6).
- Step 1: AutoARIMA 0.951, SCUM 0.959, TCN 0.988, AutoETS 0.990, DOTM 1.002,
  Theta 1.003, DeepAR 1.028, DrCIF 1.052.

## Differencing on aaltd

Of the ML/DL forecasters first run on aaltd, five predicted changes (TCN,
DeepAR, N-BEATS, Ridge (Δy), DrCIF (Δy)) and five levels (RandF, Ridge,
SETAR, SETARTree, SETARForest). Level models rerun on differences
(2026-10-27; `SETAR*Diff` = aeon model on standardised first differences,
`RandFDeltaY` = level windows of 100, target the next change):

| | levels | differenced |
|---|---|---|
| Ridge | 0.998 | 0.833 |
| RandF | 0.845 | 0.791 |
| SETAR | 0.898 | 0.889 |
| SETARTree | 2.201 | 1.702 |
| SETARForest | 1.894 | 1.713 |

Relative MAE vs Naive (geometric mean, 99 problems). Differencing helps every
model; SETARTree/SETARForest stay far worse than Naive either way.

## aeon issues found

- Deep learners (TCN, DeepAR, N-BEATS) do not scale their input.
- `iterative_forecast` refits the model.
- SETAR defaults to lag 10 (too long for short series); SETARTree to no
  scaling; SETAR-family forecasts explode on M4 even at step 1.
- TAR raises a singular-matrix error when a regime is all zeros.
- TensorFlow models leak host memory across fits.

## LSSH-99 without seasonality (2026-10-27)

`scripts/aaltd_noseason.py`, results in
`/local/ajb2u23/Results/Forecasting/aaltd-8585c7bf-noseason/`. SCUM with period
1 (done); DrCIF (Δy) and TCN (Δ) without deseasonalising (running, slow; the
user chose not to speed them up: the paper uses what is complete).

SCUM relative MAE vs Naive (geometric mean, 98 series), with the Monash period
vs period 1: all 0.843 vs 0.973 (period 1 is 15% worse). By period: 7 (daily)
1.054 vs 1.061 (no difference; weather/river/sunspot have no weekly cycle and
the members decline it), 12: 0.965 vs 1.011, 24: 0.694 vs 0.895, 48: 0.575 vs
0.834, 52 (M4 weekly): 0.699 vs 0.971, 144: 1.026 vs 0.933, 1440: 1.011 vs
0.991. Seasonal SCUM fails on all four traffic_hourly series (1.6-3.5x naive,
vs 1.0-1.1 with period 1); traffic has a 168-hour weekly cycle on top of the
daily one. Not yet checked which member fails.
