# M4 hourly: accuracy over the 48-step horizon (2026-10-29)

Do forecasters on the M4 hourly series (414 series, horizon 48) degrade over the
horizon to random guessing, or at least to no better than Naive2?

The script is `scripts/m4_hourly_horizon.py`, with results in
`results/aeon-8585c7bf/m4-hourly-horizon/`:
- `by_step.csv`: per method and step;
- `by_block.csv`: per method and block of steps;
- `by_step.png`.

## What is compared

- the paper's aeon M4 runs (8585c7bf, including the deep learners);
- the published M4 submissions: Naive2, ETS, ARIMA, Theta and SCUM (entry 036);
- no-skill baselines computed from each training series:
  - `Naive1`: the last value;
  - `SeasonalNaive`: yesterday's 24 values repeated;
  - `Mean28d`: the mean of the last 28 days;
  - `Profile28d`: the mean of each hour of the day over the last 28 days;
  - `Random28d`: a random draw from the last 28 days, the expected error over 50
    draws. This is "random guessing".

## Measures

Per step, or per block of steps:
- OWA against Naive2 at that step;
- mean MASE;
- a robust measure, `gm_rel_mae`: per series, the MAE over the block relative to
  the reference's, then the geometric mean over series;
- the share of series better than the reference.

Mean MASE and OWA are dominated by a few series with an almost exactly repeating
daily cycle. Their seasonal-naive error scale is tiny, so anything that does not
copy yesterday scores 10–20 on them. Read the robust measure first.

## Findings

1. **Nothing sensible degrades to random guessing.** `Random28d` is 4–8× worse
   than Naive2 at every step. `Naive1`, `Mean28d` and `Profile28d` are 1.3–5×
   worse. Every forecaster that models the daily cycle stays far from these.

2. **Errors follow the daily cycle, not the horizon.**
   - Error is lowest at steps 24 and 48 and highest around steps 14–18 and 38–42.
   - Day 2 is only modestly worse than day 1. For example, mean MASE is 1.0 at
     steps 1–6 and 1.2 at steps 37–48 for RandF.
   - At steps 23–24 and 46–48 everything converges on Naive2: the OWA of most
     methods rises to about 1 (see the plot).
   - The reason is that at those steps Naive2 equals seasonal naive. "The same hour
     yesterday" is the best single predictor and few methods beat it there.

3. **Naive2 is a weak benchmark on hourly data mid-horizon.**
   - Naive2 deseasonalises, carries the last level and reseasonalises with
     indices averaged over the whole training series.
   - Except at multiples of 24 it is 1.5–2.1× worse than seasonal naive
     (`gm_rel_mae_snaive`).
   - So "better than Naive2" is easy here: OWA against Naive2 flatters every method.

4. **Against seasonal naive, the stronger reference, by block** (robust relative
   MAE, below 1 is better):

   | | 1–6 | 7–12 | 13–24 | 25–36 | 37–48 |
   |---|---|---|---|---|---|
   | RandF | 0.69 | 0.88 | 0.78 | 0.74 | 0.71 |
   | M4 ARIMA (R) | 0.67 | 0.76 | 0.84 | 0.72 | 0.81 |
   | SCUM (aeon) | 0.89 | 1.17 | 0.97 | 0.89 | 0.85 |
   | AutoCES | 1.00 | 1.07 | 0.96 | 0.93 | 0.91 |
   | AutoETS | 0.94 | 1.20 | 1.08 | 0.94 | 0.94 |
   | DrCIF (Δy) | 0.70 | 0.96 | 1.01 | 0.93 | 1.02 |
   | AutoARIMA (aeon) | 1.06 | 1.42 | 1.43 | 1.19 | 1.16 |
   | Naive2 | 1.48 | 2.09 | 2.06 | 1.47 | 1.51 |
   | Theta, DOTM | ≈ Naive2 | | | | |
   | DeepAR† | 1.07 | 1.63 | 1.74 | 1.36 | 1.66 |
   | TCN† | 0.97 | 1.63 | 1.97 | 1.74 | 2.26 |
   | N-BEATS† | 1.50 | 2.36 | 2.77 | 2.48 | 2.75 |

   - **Only RandF and R's ARIMA beat seasonal naive throughout.** RandF beats it
     on 57–70% of series in every block.
   - **SCUM, AutoCES and AutoETS are roughly level with seasonal naive.** They are
     slightly worse at steps 7–12 and slightly better on day 2.
   - **aeon AutoARIMA is 16–43% worse than seasonal naive,** and clearly worse than
     R's ARIMA submission: 0.67–0.84.
   - **Theta and DOTM are no better than Naive2.** They forecast the deseasonalised
     series flat, which makes them Naive2.
   - **DrCIF (Δy) is the best method at steps 1–6** (equal with RandF) and decays to
     seasonal naive by day 2. Recursive forecasting of changes integrates its
     errors.
   - **The deep learners† degrade fastest.**
     - TCN goes from level with seasonal naive at steps 1–6 to 2.3× worse at
       37–48; N-BEATS is worse than Naive2 from step 7.
     - They are fitted once to standardised differences, so the forecast integrates
       the differences.
     - **This is the one place where a method really does fall to "no better than
       Naive2"**, and it happens on day 2.

5. **Answer to the question.**
   - No method decays to random guessing.
   - The statistical models and the forests stay better than Naive2 at every step
     except the multiples of 24, where Naive2 is seasonal naive.
   - **Against seasonal naive** most methods add little beyond the first 6 hours.
     The exceptions are RandF and R's ARIMA, which keep a 20–30% advantage
     throughout.
   - **The deep learners and DrCIF (Δy)** lose their early advantage by day 2.

## Side observations (paper already fixed; not acted on)

- **The paper's hourly "ETS" and "Naive" rows are not seasonal in effect.**
  - In the M4 league they score 3.59, the same as `Naive1` here.
  - The league legend says they "get the M4 period". The run arguments do say
    `--deseasonalise`, but the hourly forecasts match the last-value naive.
  - The period is evidently not used by aeon's default `NaiveForecaster`
    (strategy "last") or the default ETS (A,N,N).
  - Worth checking before anyone quotes those hourly numbers. They only affect
    the hourly column, which is 414 of 100,000 series.
- **The hourly series with a near-exactly repeating cycle** (H182, H269, H219, H313,
  H316: the lag-1 error is about 10–25 times the lag-24 error) dominate the mean
  MASE.
