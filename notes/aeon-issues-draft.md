# Draft aeon issues from the M4 / aaltd experiments

Drafts only, to be opened after the paper draft (user, 2026-10-27). All found
at aeon commit 8585c7bf (`aeon-bench-next`, branch `bench/arima-ets`); check
each against current `main` before opening. Evidence paths are in this repo or
under `/local/ajb2u23/Results/Forecasting/`.

## 1. Deep learning forecasters do not scale their input

`TCNForecaster`, `DeepARForecaster`, `NBeatsForecaster` train on the raw
series. On series in the hundreds or thousands they fail to train (DeepAR
forecasts about zero). M4 pilot, 600 series, OWA vs Naive2: DeepAR 26, TCN 3.3,
N-BEATS 2.5 as shipped; with per-series standardisation 2.86 / 1.74 / 1.82; on
standardised first differences 1.10 / 1.41 / 1.46.
Suggested fix: standardise (option to difference) inside fit/predict, as most
DL forecasting libraries do. Evidence: `results/aeon-8585c7bf/m4/owa_dl_pilot_600.csv`;
wrapper `StandardisedForecaster` in `scripts/m_comp_experiments.py`.

## 2. `iterative_forecast` refits the model

`iterative_forecast(y, h)` calls `fit` again, so a fitted model cannot be used
to forecast recursively without paying the fit cost twice (expensive for DL).
Suggested fix: forecast from the fitted state, or document and offer a
predict-only recursion.

## 3. TensorFlow forecasters leak host memory across fits

Repeated fit/predict in one process grows memory (about 75 MB per fit even
with `tf.keras.backend.clear_session()`; workers reached 7 GB and the host hit
the OOM killer on 2026-10-23). Workaround here: recycle worker processes.
Suggested: release the Keras model/graph after fit, or document.

## 4. Deep learning regressors write `best_model.keras` to the working directory

Default `file_path="./"` with the default checkpoint callback leaves
`best_model.keras` (and others) in the caller's current directory; parallel
fits in one directory overwrite each other's checkpoints. Found as a stray
`best_model.keras` in the horizon repo root.
Suggested: default to a temporary directory, or only checkpoint when
`save_best_model=True`.

## 5. SETAR forecasts explode on short series (default lag 10)

M4 (deseasonalised), mean OWA 5.6e46; already broken one step ahead (yearly
step-1 OWA 625), so the fit, not the recursion, fails. Yearly series have a
median of ~29 training points; a lag of 10 leaves too few points per regime.
Fine on long series (aaltd relative MAE 0.898).
Suggested: cap the lag by series length / minimum points per regime, fall
back to linear AR when a regime is under-determined.
Evidence: `results/aeon-8585c7bf/m4/owa_by_steps.csv`.

## 6. SETARTree / SETARForest unstable (default `scale=False`)

M4 mean OWA 1e36 / 2e37. aaltd (long series, one step ahead): complete, but
worse than Naive on 73/99 problems, more than 10x Naive's error on 11-12;
SETARForest 467x on solar_10_minutes T5 (MASE 2004), mostly series with long
zero/flat stretches (solar, wind, traffic). Standardised differenced input
reduces but does not remove it (worst 122x).
Suggested: scale by default; guard leaf regressions against near-constant
data (regularise or fall back to the leaf mean).
Evidence: `results/aeon-8585c7bf/aaltd/per_problem.csv`.

## 7. TAR raises LinAlgError when a regime is all zeros

`TAR` fails with a singular-matrix `LinAlgError` on 11 aaltd series with long
zero runs (solar, weather, wind): all 30 origins. Suggested: least squares
with a pseudo-inverse / ridge term, or drop under-determined regimes.

## 8. AutoTAR forecasts explode on M4

M4 mean OWA 1.3e13 (yearly 2.8e13, hourly 1125), step-1 yearly 3.9.
Probably the same causes as 5 and 7. Investigate together.

## Possibly (check before filing)

- `NaiveForecaster` default strategy is `"last"`; `seasonal_period` is ignored
  unless `strategy="seasonal_last"`. Documented, but easy to misuse.
- `IntervalForestRegressor` (50 trees) did not finish one fit on 10,000
  windows of length 100 in 30 minutes, against 2.5 minutes for CIF.
