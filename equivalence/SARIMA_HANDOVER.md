# SARIMA equivalence: handover

Run aeon's `SARIMA` and `AutoSARIMA` through the IJF paper's equivalence protocol on the benchmark
machine (Hydra). **No full run has completed, so there are no results yet.** The local run
was stopped when the work moved to this machine. The only output so far is a
smoke test, which proves the harness runs, not that either is equivalent.

## What to test

- aeon: branch `ajb/sarima`, commit `c5615dd85`, with `SARIMA` and `AutoSARIMA`,
  pushed to origin. The run scripts refuse a dirty checkout.
- Harness: `sarima.py`, `sarima_refs.R`, `run_sarima.sh`, `m4.py` and
  `../scripts/m_comp_experiments.py` on horizon `main`.

## Pass criteria (from the paper)

Taken from `main.tex` Table `tab:other_stats` and `equivalence/README.md`:

| Evidence | Standard |
|---|---|
| Accuracy | Geometric mean of per-series MASE ratio aeon / reference |
| Equivalence | TOST on log ratios, ±5% margin: 90% interval within [0.952, 1.05] |
| Difference | Paired Wilcoxon signed-rank, Holm corrected |
| Simulated | 12 processes x 100 series, lengths 100 and 500 |
| Real data | M4, reported by frequency |
| Speed | Reference time / aeon time at n = 100, 1,000 and 10,000, one thread |

"Inconclusive" means the interval crosses a margin. It is not evidence that aeon
is worse. Per-process results at 100 series are often too few to show equivalence
within 5%, so expect some.

The paper's wording to support would be "equivalent to `stats::arima`, the same
objective, and to statsforecast and statsmodels". The most direct evidence is
`R CSS`, then `objective_summary.csv`.

## Running it

Prerequisites:

- `AEON_M4_DATASET`: the `Dataset` directory of the M4-methods repository, holding
  `Train/` and `Test/` CSVs. This is **not** the `.tsf` layout in
  `/local/ajb2u23/Data/Forecasting/m4` that `m4.py` uses.
- `ENV`: Python environment with aeon's dependencies, `joblib`, `statsmodels`,
  `statsforecast`, and whatever `horizon.sweep` imports. Defaults to
  `/local/ajb2u23/conda/env/aeon-fcbench`.
- `RSCRIPT`: R with the `forecast` package. Defaults to `Rscript`.
- A clean aeon checkout. The script refuses a dirty one.

Smoke test first (a few minutes):

```bash
export PYTHONPATH=$AEON NUMBA_CACHE_DIR=$(mktemp -d)
python equivalence/sarima.py series --out /tmp/s --replicates 2 --m4-limit 3
python equivalence/sarima.py python --out /tmp/s --jobs 8
Rscript equivalence/sarima_refs.R /tmp/s/series.csv /tmp/s/r.csv 8
python equivalence/sarima.py analyse --out /tmp/s
```

Then the full run: 7,228 series with 6 fits each, plus 3 automatic fits on each
of the 2,400 simulated series. Run time is not known yet, and statsmodels on long M4
series will dominate it:

```bash
AEON_M4_DATASET=... equivalence/run_sarima.sh $AEON 60
```

### AutoSARIMA on all of M4

This is the comparison with the paper's M4 table and the published `auto.arima`.
It uses the existing M4 runner, the `.tsf` data and `official/`, not
`AEON_M4_DATASET`. Run it from the benchmark environment with a fresh
`NUMBA_CACHE_DIR` and `PYTHONPATH=$AEON`, without `--deseasonalise`:

```bash
python scripts/m_comp_experiments.py --datasets M4 --limit-per-type 0 \
    --forecasters AutoSARIMA --backends aeon --n-jobs 60 \
    --progress-every 10000 --output-dir $OUT
python equivalence/m4.py --results $OUT --out equivalence/results/aeon-<commit>/m4-sarima
```

`m_comp_experiments.py` and `m4.py` have been changed to accept `AutoSARIMA`.
`_fit_aeon` was checked on a synthetic series. The whole `m4.py` flow has not been
run, because the `.tsf` data is not on the development machine. Fits took about
0.5 s on three hourly series. Monthly (48,000) will dominate the run.

Results go to `equivalence/results/sarima-c5062a53/`:

| File | Contents |
|---|---|
| `summary.txt` | Pooled and per-process verdicts, failures, objective summary |
| `tost.csv` | Every comparison |
| `failures.csv` | Fit failures by toolkit |
| `objective_summary.csv` | aeon's AIC minus aeon's objective at R's CSS estimates |
| `timing_python.csv`, `timing_r.csv` | Speed |

`objective_summary.csv` should show differences at or below about 0. A positive
difference means R found a better optimum of aeon's own objective, which is an
estimation weakness regardless of the forecast accuracy.

## Design

- **Simulated:** `PROCESSES` in `sarima.py`, with m = 4, 7, 12 and 24. Each is
  fitted with its true order. h = max(12, 2m).
- **M4:** all 414 hourly, plus 1,000 monthly and 1,000 quarterly sampled with seed
  0. Each is fitted with (0,1,1)(0,1,1)m and (1,1,1)(0,1,1)m.
- **Constant:** only when d + D = 0. R uses a mean, aeon an intercept. They are
  reparametrised for the objective check in `_at_r_params`.
- **MASE:** scaled by the in-sample seasonal naive error, as in M4.
- **Automatic (simulated only):** `aeon auto` (`AutoSARIMA`) against `R auto.arima`
  and `statsforecast auto`. These are reported in `tost.csv` as bands `auto 100`
  and `auto 500`.

`AutoSARIMA` follows aeon's `AutoARIMA`, not `auto.arima`: limits p <= 3, q <= 2,
d <= 3, the same starts in seasonal form, a constant start only if d + D = 0,
both constant settings tried at each neighbour, first improvement, 200 then 500
iterations, AIC, and the 1.01 root check. It adds P, Q <= 2, D <= 1 and R's default
seasonal strength rule for D, using aeon's STL. With period 1 it gives the same
order, AIC and forecast as `AutoARIMA` on airline. On airline with period 12 it
selects (1,1,0)(0,1,0)12, and R's default selects (2,1,1)(0,1,0)12.

Unlike R it uses AIC rather than AICc, has no limit on p + q + P + Q and allows a
constant with d + D = 2. The README suggests AICc as an untested cause of the
yearly gap, and seasonal models have more parameters, so look here first if
monthly or quarterly are behind R.

## Known issues

1. **`statsforecast CSS` is not a valid reference at present.** On version 2.0.3
   (development machine) it returned constant forecasts for 66 of 66 smoke-test
   series, with MASE up to about 150. The cause is not diagnosed. Check it on the
   benchmark version (paper: 2.1.1). If it persists, drop it and report only
   `statsforecast CSS-ML`. Do not report its "aeon better" verdicts.
2. **Two separate M4 comparisons.** `run_sarima.sh` tests fixed orders on a sample
   of M4. Only the `AutoSARIMA` run above bears on the hourly AutoARIMA gap (OWA
   0.692 against 0.577).
3. **Version differences.** The development machine had statsforecast 2.0.3 and
   statsmodels 0.14.6; the paper cites 2.1.1 and 0.15.0. Record what you run.
4. **R timing resolution.** `system.time` is too coarse for CSS below about n =
   10,000: it printed 0 at n = 100 and 1,000. Raise the repeats in
   `sarima_refs.R` if R's CSS speed-up goes in the paper.
5. **R timing series.** `timing_r.csv` comes from R's own generator, so it uses a
   different series from `timing_python.csv` at the same n, and it is written at
   the end of the R stage, so it is measured under whatever load remains. Rerun
   that part on an idle machine if it goes in the paper.
6. **Harmless noise.** `queue.Full` tracebacks from loky appeared at the end of the
   `python` stage in the smoke test. `python.csv` was written and complete.

## State of the code

Checked on the development machine, with `AutoSARIMA`:

- All of `aeon/forecasting`: 548 passed with a fresh numba cache.
- `check_estimator` on `SARIMA` and `AutoSARIMA`: 19 checks each, no failures.
- ruff check and format: clean.
- Against ARIMA with P=Q=0: identical parameters, AIC and forecast on airline.
- Against statsmodels on airline (0,1,1)(0,1,1)12: forecasts within 1%.

Not fixed, outside this work:

- ARIMA's exog forecasts have the double-counting bug found in review (SARIMA no
  longer accepts exog).
- `ARIMA._predict` undifferences from the training series `self._series`, not from
  the `y` passed in. SARIMA uses `y`. Seen when reading the code, not tested.

## After the run

- If it passes: the paper needs updating. It says `aeon`'s ARIMA is non-seasonal
  (Section 4 and the commented toolkit text), `SARIMA` and `AutoSARIMA` are not in
  the list of statistical forecasters or in `tab:other_stats`, and the M4 text
  gives seasonal ARIMA as the reason for the hourly difference.
- `SCUM` still uses non-seasonal `AutoARIMA`. Whether to switch its member to
  `AutoSARIMA` is a separate decision.
- Add a section to `equivalence/README.md` for SARIMA.
- Consider adding one or two fixed reference forecasts to `test_sarima.py`, as
  `test_dotm.py` does, so equivalence is guarded without statsmodels.
