# Equivalence of aeon's AutoETS and AutoARIMA

Is aeon's automatic ETS and ARIMA as accurate as the reference implementations,
and how fast is it? This directory holds the tests and the evidence for aeon
commit **8585c7bf** (branch `bench/arima-ets`: `ajb/arima`, PR #3931, merged with
`ajb/ets`; 2026-10-21). It uses two kinds of evidence:

1. **M4, against the published benchmarks.** aeon's forecasts for all 100,000
   M4 series, compared with the official M4 forecasts of R's `forecast::ets` and
   `forecast::auto.arima`.
2. **Simulated series, against other Python toolkits.** The horizon benchmark:
   series from known ARIMA and ETS processes, compared with statsforecast,
   sktime, skforecast, pmdarima and statsmodels, with model selection (`auto`)
   and with the true model given (`fixed`).

Two kinds of test are used throughout, on paired per-series errors:

- **Difference:** two-sided Wilcoxon signed-rank test, Holm corrected. It
  asks whether aeon and the reference differ at all. With tens of thousands of
  series it detects differences far too small to matter.
- **Equivalence:** TOST on the per-series log error ratio (aeon / reference)
  with a **±5% margin**. A verdict of "equivalent" means the 90% interval of
  the geometric mean ratio lies within [0.952, 1.05]. "Inconclusive" means
  the interval crosses a margin; it is not evidence that aeon is worse.

## Verdict

| | M4, against R (TOST ±5%) | Simulated, against Python toolkits |
|---|---|---|
| **AutoETS** | **Equivalent** overall and on yearly, quarterly, monthly, weekly and daily. Hourly: aeon 8% more accurate, inconclusive. | No competitor significantly better in any comparison. |
| **AutoARIMA** | **Equivalent** overall and on yearly, quarterly, weekly and daily. Monthly: equivalent on sMAPE, MASE just outside the margin (ratio 1.047, upper limit 1.050). Hourly: **aeon 58-61% worse**, because aeon has no seasonal ARIMA (see caveats). | No competitor significantly better in any comparison. |

The difference tests are stricter, and on M4 they separate aeon from R on
several frequencies, in both directions, with small effects (rank-biserial
|r| <= 0.11, except hourly ARIMA, r = 0.39). Details below.

**Run time:** aeon's AutoETS is faster than every competitor at every length.
aeon's AutoARIMA is the fastest on short series; skforecast's auto-ARIMA is
faster from about 500 observations, and statsforecast's at 10,000. Automatic
fits take 1.8-4.8x as long as in aeon 1.6.0, whose estimation was wrong (see
`reports/aeon-arima-ets-estimation.md`).

## 1. M4 against the published R forecasts

`equivalence/results/aeon-8585c7bf/m4/`. aeon AutoETS gets the M4 seasonal period
(1, 4, 12, 1, 1, 24). aeon's AutoARIMA is non-seasonal, so it is fitted to
series deseasonalised as in the M4 benchmarks: 90% ACF seasonality test, then
classical multiplicative decomposition. This reproduces the official Naive2
forecasts exactly on all 100,000 series. The official submissions score exactly
their published values: ETS sMAPE 12.725, MASE 1.680, OWA 0.908; ARIMA 12.669,
1.666, 0.903.

**OWA** (relative to Naive2, lower is better; `owa.csv`)

| | aeon AutoETS | R ets | aeon AutoARIMA | R auto.arima |
|---|---|---|---|---|
| Yearly (23,000) | 0.905 | 0.903 | 0.918 | 0.892 |
| Quarterly (24,000) | 0.892 | 0.890 | 0.894 | 0.898 |
| Monthly (48,000) | 0.929 | 0.914 | 0.918 | 0.903 |
| Weekly (359) | 0.938 | 0.931 | 0.902 | 0.932 |
| Daily (4,227) | 1.001 | 0.996 | 1.012 | 1.044 |
| Hourly (414) | 0.788 | 0.852 | 0.692 | 0.577 |
| **All** | **0.914** | **0.908** | **0.915** | **0.903** |

**Equivalence** (TOST, ±5%; geometric mean of per-series MASE ratios aeon / R,
with 90% interval; `tost.csv`, which also has sMAPE)

| | AutoETS / ets | verdict | AutoARIMA / auto.arima | verdict |
|---|---|---|---|---|
| Yearly | 0.988 (0.984-0.993) | equivalent | 1.029 (1.025-1.033) | equivalent |
| Quarterly | 1.010 (1.006-1.014) | equivalent | 1.012 (1.008-1.016) | equivalent |
| Monthly | 1.038 (1.035-1.041) | equivalent | 1.047 (1.044-1.050) | inconclusive |
| Weekly | 1.015 (1.001-1.029) | equivalent | 0.980 (0.954-1.006) | equivalent |
| Daily | 1.003 (1.000-1.005) | equivalent | 0.979 (0.973-0.986) | equivalent |
| Hourly | 0.912 (0.866-0.962) | inconclusive | 1.614 (1.494-1.743) | aeon worse |
| **All** | **1.018 (1.015-1.020)** | **equivalent** | **1.033 (1.031-1.035)** | **equivalent** |

**Difference** (Wilcoxon signed-rank on per-series MASE, Holm corrected over 12
tests per method; rank-biserial r < 0 means aeon has lower errors;
`signed_rank.csv`, which also has sMAPE)

| | AutoETS: r | verdict | AutoARIMA: r | verdict |
|---|---|---|---|---|
| Yearly | -0.10 | aeon better | +0.10 | R better |
| Quarterly | +0.04 | R better | +0.03 | R better |
| Monthly | +0.08 | R better | +0.11 | R better |
| Weekly | +0.03 | no difference | +0.04 | no difference |
| Daily | +0.07 | R better | +0.04 | no difference |
| Hourly | -0.10 | no difference | +0.39 | R better |

For scale: r = 0.10 corresponds roughly to R having the lower error on 55%
of the series where the two differ, and aeon on 45%.

**What changed from the previous branch head (439ed8bc).** AutoETS: one series
(Q5018) forecast 4.5e9, because an additive-error, multiplicative-season model
was selected. `AutoETS(restrict=True)` now excludes these models, as R does
(overall MASE 9.29 -> 1.68). AutoARIMA: d was chosen with a KPSS test using
`int(sqrt(n))` lags, about 4x more than R's `trunc(3*sqrt(n)/13)`, so 32% of
yearly series were left undifferenced. It now uses R's rule, and 9.4% are
undifferenced, as in R (OWA 0.966 -> 0.915).

## 2. Simulated series against the Python toolkits

`equivalence/results/aeon-8585c7bf/simulated/`, summarising the runs in
`results/aeon-8585c7bf/{auto,fixed}`.
Six ARIMA and six ETS processes, 100 series per process and length, lengths 100
and 500; see the main README and `reports/aeon-arima-ets-estimation.md`.

**Difference** (Wilcoxon signed-rank on per-series MASE, Holm corrected;
`difference_verdicts.csv`)

| mode | comparisons | competitor better | no difference | aeon better |
|---|---|---|---|---|
| auto | 18 pooled | 0 | 16 | 2 |
| auto | 108 per process | 0 | 105 | 3 |
| fixed | 18 pooled | 0 | 14 | 4 |
| fixed | 108 per process | 0 | 101 | 7 |

No competitor is significantly more accurate than aeon in any of the 252
comparisons. Every "aeon better" verdict is for ETS: against statsforecast,
and in fixed mode also against sktime and statsmodels, which share one engine.

**Equivalence** (TOST, ±5%, on per-series MASE ratios; `tost_verdicts.csv`)

| mode | comparisons | equivalent | inconclusive | aeon better | aeon worse |
|---|---|---|---|---|---|
| auto | 18 pooled | 16 | 1 | 1 | 0 |
| auto | 108 per process | 79 | 26 | 3 | 0 |
| fixed | 18 pooled | 14 | 3 | 1 | 0 |
| fixed | 108 per process | 98 | 5 | 5 | 0 |

The four inconclusive pooled comparisons all lean towards aeon: its errors are
lower, but the interval crosses the lower margin. These are ETS against
statsforecast, sktime and statsmodels. Per process, 100 series is often too few
to show equivalence within 5%. Of the 31 inconclusive per-process comparisons,
15 lean towards aeon being worse, all at length 100. The largest is AutoARIMA
against the statsmodels grid search on ARIMA(1,1,0), with a ratio of 1.084
(1.019-1.153).

Against 439ed8bc (`results/comparisons/8585c7bf-vs-439ed8bc/`), no pooled or
per-process accuracy changed significantly in either mode.

## Run time

Median seconds to fit and forecast one series, single threaded, JIT
compilation excluded, 5 repeats, on an otherwise idle machine
(`equivalence/results/aeon-8585c7bf/simulated/timing.csv`). The competitor columns give
competitor time / aeon time: above 1, aeon is faster.

| mode | process | n | aeon (s) | pmdarima | skforecast | sktime | statsforecast | statsmodels |
|---|---|---|---|---|---|---|---|---|
| auto | ARMA(1,1) | 100 | 0.027 | 17.8 | 3.1 | 10.6 | 4.2 | 7.7 |
| auto | ARMA(1,1) | 1,000 | 0.163 | 9.7 | 0.28 | 6.5 | 1.4 | 4.7 |
| auto | ARMA(1,1) | 10,000 | 0.973 | 12.9 | 0.12 | 9.2 | 0.82 | 4.1 |
| auto | ETS A,A,A m=12 | 100 | 0.016 | | 14.3 | 36.4 | 3.2 | 47.3 |
| auto | ETS A,A,A m=12 | 1,000 | 0.107 | | 13.7 | 10.1 | 4.2 | 11.4 |
| auto | ETS A,A,A m=12 | 10,000 | 1.058 | | 14.4 | 5.6 | 4.2 | 6.8 |
| fixed | ARMA(1,1) | 100 | 0.001 | 48.5 | 4.2 | 52.5 | 10.1 | 11.3 |
| fixed | ARMA(1,1) | 10,000 | 0.034 | 57.0 | 0.72 | 57.0 | 2.0 | 8.4 |
| fixed | ETS A,A,A m=12 | 100 | 0.001 | | 21.2 | 44.0 | 4.3 | 41.5 |
| fixed | ETS A,A,A m=12 | 10,000 | 0.045 | | 37.5 | 10.8 | 6.9 | 10.7 |

- **AutoETS** is faster than every competitor at every length: 3-7x faster
  than statsforecast, 5-47x faster than the others.
- **AutoARIMA** is faster than every competitor at 100-200 observations.
  skforecast is level at 500 and faster from 1,000 (3.6x at 1,000, 8x at
  10,000). statsforecast is level at 5,000 and 1.2x faster at 10,000. With the
  order given (fixed mode), aeon is faster than all but skforecast, which
  overtakes it at about 5,000.
- **Against earlier aeon:** auto-mode fits take 1.8-2.4x (ETS) and 2.3-4.8x
  (ARIMA) as long as in aeon 1.6.0; fixed-mode fits 1.2-1.4x (ETS) and
  1.9-2.4x (ARIMA). The extra time buys the multi-start optimisation that makes
  the estimates correct. Against 439ed8bc, AutoETS is 13-22% faster (fewer
  candidate models with `restrict`), and AutoARIMA is 1.2-1.8x slower except at
  length 500 (0.8x), probably from the new diagonal stepwise moves (not
  checked).
- **On M4**, aeon took a median of 26 ms per series for AutoETS and 17 ms for
  AutoARIMA, about one CPU-hour each for all 100,000 series. These times were
  measured with 40 worker processes running, and R's published forecasts come
  without run times, so they are not a comparison.

## Caveats

- **Hourly AutoARIMA.** R's `auto.arima` fits seasonal ARIMA (m = 24); aeon has
  only non-seasonal ARIMA, so it runs on deseasonalised series. Hourly
  equivalence needs seasonal ARIMA in aeon. The same difference applies, less
  visibly, to monthly and quarterly.
- **Search limits.** aeon's AutoARIMA searches p <= 3, q <= 2, d <= 3, against
  R's p, q <= 5 and d <= 2. Using R's limits barely changes the results (see
  below).
- **One aeon forecast can still go negative where R's does not.** Q5018
  (A,A,A after a level drop) does, with MASE 0.42 against R's 0.20.
- **Simulated series come from the model classes being fitted.** They test
  estimation and selection; M4 tests real data.
- **Run times** come from one machine (64 cores, one thread per fit). The
  timing processes are ARMA(1,1) and ETS(A,A,A) m=12 only.

## Rerunning

Paths here are relative to the repository root. Run from the root with the benchmark environment. `AEON` is a clean
git checkout of the aeon to test, such as a worktree; never your development
checkout. Both scripts refuse a checkout with uncommitted changes, use a fresh
numba cache, and name results after the commit.

```bash
equivalence/run_m4.sh        $AEON   # ~2 CPU-hours: M4 runs, then m4.py
equivalence/run_simulated.sh $AEON   # ~1 hour: both modes in parallel, then simulated.py
```

The summary steps on their own, from existing results:

```bash
python equivalence/m4.py --results /local/ajb2u23/Results/Forecasting/aeon-<commit> \
    --out equivalence/results/aeon-<commit>/m4
python equivalence/simulated.py --run results/aeon-<commit> \
    --out equivalence/results/aeon-<commit>/simulated
```

`--margin` sets the TOST margin (default 0.05). M4 data and the official
forecasts are read from `/local/ajb2u23/Data/Forecasting/m4`
(`m4_<frequency>_dataset.tsf`; `official/` holds the test sets and
`submission-{ETS,ARIMA,Naive2}.csv` from the
[M4-methods repository](https://github.com/Mcompetitions/M4-methods)).

| file | |
|---|---|
| `run_m4.sh`, `run_simulated.sh` | run aeon, then summarise |
| `m4.py` | M4: OWA, signed-rank and TOST against the published forecasts |
| `m4_score.py` | M4 scoring (sMAPE, MASE, OWA as in the M4 evaluation) |
| `simulated.py` | simulated: verdict counts, TOST, timing table |
| `results/aeon-<commit>/m4/` | `owa.csv`, `signed_rank.csv`, `tost.csv`, `run_info.json`; `per_series.csv.gz` is regenerated, not committed |
| `results/aeon-<commit>/simulated/` | `difference_verdicts.csv`, `tost.csv`, `tost_verdicts.csv`, `timing.csv`, `run_info.json` |

## AutoARIMA with R's order limits

`equivalence/results/aeon-8585c7bf-arima525/m4/`: the same AutoARIMA with
`max_p=5, max_d=2, max_q=5` (R's defaults; `--arima-max-order 5 2 5`).

| OWA | default limits (p<=3, q<=2, d<=3) | R's limits (5, 2, 5) | R auto.arima |
|---|---|---|---|
| Yearly | 0.918 | 0.917 | 0.892 |
| Quarterly | 0.894 | 0.897 | 0.898 |
| Monthly | 0.918 | 0.916 | 0.903 |
| Weekly | 0.902 | 0.886 | 0.932 |
| Daily | 1.012 | 1.010 | 1.044 |
| Hourly | 0.692 | 0.667 | 0.577 |
| **All** | **0.915** | **0.914** | **0.903** |

The wider search chooses p > 3 or q > 2 for 14% of series (3% of yearly, about
half of weekly and hourly). It takes 2.7x the CPU time (154 against 58 CPU-minutes) for
almost no overall gain. With it, monthly MASE becomes equivalent (1.041), and
weekly sMAPE becomes inconclusive in aeon's favour (interval 0.952-1.002).
Yearly is unchanged: 97% of yearly series get the same model. So the yearly gap
to R is not caused by the order limits. One untested candidate is the
information criterion: aeon selects by AIC from the CSS likelihood, while R's
`auto.arima` uses AICc (`ic = "aicc"`), which penalises extra parameters much
more on short series such as M4 yearly (median length about 30).

## In progress

- A fresh statsforecast 2.1.1 AutoARIMA on M4, as a seasonal Python reference.
