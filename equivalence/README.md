# Equivalence of aeon's AutoETS and AutoARIMA

Is aeon's automatic ETS and ARIMA as accurate as the reference implementations,
and how fast is it? This directory holds the tests and the evidence for aeon
commit **8585c7bf** (branch `bench/arima-ets`: `ajb/arima`, PR #3931, merged with
`ajb/ets`; 2026-10-21). It uses two kinds of evidence:

1. **M4, against the published benchmarks.** aeon's forecasts for all 100,000
   M4 series, compared with the official M4 forecasts of R's `forecast::ets` and
   `forecast::auto.arima`, and of SCUM (Petropoulos and Svetunkov's entry 036,
   rank 6).
2. **Simulated series, against other Python toolkits.** The horizon benchmark:
   series from known ARIMA and ETS processes, compared with statsforecast,
   sktime, skforecast, pmdarima, statsmodels and smooth (Svetunkov's ADAM
   package), with model selection (`auto`) and with the true model given
   (`fixed`).

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
| **AutoETS** | **Equivalent** overall and on yearly, quarterly, monthly, weekly and daily. Hourly: aeon 8% more accurate, inconclusive. | No other toolkit significantly better in any comparison. |
| **AutoARIMA** | **Equivalent** overall and on yearly, quarterly, weekly and daily. Monthly: equivalent on sMAPE, MASE just outside the margin (ratio 1.047, upper limit 1.050). Hourly: **aeon 58-61% worse**, because aeon has no seasonal ARIMA (see caveats). | No other toolkit significantly better in any comparison. |
| **SCUM** | **Equivalent** to the published SCUM overall (ratio 1.004) and, when run with 036's seasonal periods (weekly 52, daily 7, hourly 168), on every frequency (all ratios within 1%). With the M4 benchmark periods used in the tables below, weekly is 15% worse and hourly 14% better. | Not in the simulated benchmark. |

The difference tests are stricter, and on M4 they separate aeon from R on
several frequencies, in both directions, with small effects (rank-biserial
|r| <= 0.11, except hourly ARIMA, r = 0.39). Details below.

**Run time:** on seasonal ETS(A,A,A) data aeon's AutoETS is faster than
every other toolkit at every length. On non-seasonal ETS(A,A,N) it is level
with statsforecast up to 500 and faster beyond, but statsmodels and sktime
catch up at about 3,000 observations and are 1.2-1.4x faster from 5,000. aeon's AutoARIMA is
the fastest on short series; skforecast's auto-ARIMA is faster from about 500
observations, and statsforecast's at 10,000. Figures in `paper/` (median of 20 series per length,
`results/aeon-8585c7bf/sweep-auto-r20`). Automatic
fits take 1.8-4.8x as long as in aeon 1.6.0, whose estimation was wrong (see
`reports/aeon-arima-ets-estimation.md`).

## 1. M4 against the published R forecasts

`equivalence/results/aeon-8585c7bf/m4/`. aeon AutoETS gets the M4 seasonal period
(1, 4, 12, 1, 1, 24). aeon's AutoARIMA is non-seasonal, so it is fitted to
series deseasonalised as in the M4 benchmarks: 90% ACF seasonality test, then
classical multiplicative decomposition. This reproduces the official Naive2
forecasts exactly on all 100,000 series. The official submissions score exactly
their published values: ETS sMAPE 12.725, MASE 1.680, OWA 0.908; ARIMA 12.669,
1.666, 0.903. SCUM is aeon's `SCUM(season_length=m)` as shipped: the median of
AutoETS, AutoCES, AutoARIMA and DOTM. Its AutoARIMA member sees the raw series.

**OWA** (relative to Naive2, lower is better; `owa.csv`)

| | aeon AutoETS | R ets | aeon AutoARIMA | R auto.arima | statsforecast AutoARIMA | aeon SCUM | SCUM (036) |
|---|---|---|---|---|---|---|---|
| Yearly (23,000) | 0.905 | 0.903 | 0.918 | 0.892 | 0.906 | 0.804 | 0.806 |
| Quarterly (24,000) | 0.892 | 0.890 | 0.894 | 0.898 | 0.900 | 0.854 | 0.852 |
| Monthly (48,000) | 0.929 | 0.914 | 0.918 | 0.903 | 0.904 | 0.872 | 0.876 |
| Weekly (359) | 0.938 | 0.931 | 0.902 | 0.932 | 0.868 | 0.902 | 0.751 |
| Daily (4,227) | 1.001 | 0.996 | 1.012 | 1.044 | 1.006 | 0.983 | 0.984 |
| Hourly (414) | 0.788 | 0.852 | 0.692 | 0.577 | 0.578 | 0.651 | 0.663 |
| **All** | **0.914** | **0.908** | **0.915** | **0.903** | **0.907** | **0.845** | **0.848** |

statsforecast 2.1.1's AutoARIMA (a port of `auto.arima`, seasonal, run here on
all of M4) is a second reference. It matches R on monthly and hourly, but is
also behind R on yearly (0.906 against 0.892). So about half of aeon's yearly
gap (0.918) is shared by the Python port.

**Equivalence** (TOST, ±5%; geometric mean of per-series MASE ratios aeon /
reference, with 90% interval; `tost.csv`, which also has sMAPE)

| | AutoETS / ets | verdict | AutoARIMA / auto.arima | verdict | SCUM / 036 | verdict |
|---|---|---|---|---|---|---|
| Yearly | 0.988 (0.984-0.993) | equivalent | 1.029 (1.025-1.033) | equivalent | 0.998 (0.995-1.000) | equivalent |
| Quarterly | 1.010 (1.006-1.014) | equivalent | 1.012 (1.008-1.016) | equivalent | 1.005 (1.002-1.007) | equivalent |
| Monthly | 1.038 (1.035-1.041) | equivalent | 1.047 (1.044-1.050) | inconclusive | 1.008 (1.006-1.009) | equivalent |
| Weekly | 1.015 (1.001-1.029) | equivalent | 0.980 (0.954-1.006) | equivalent | 1.148 (1.110-1.187) | aeon worse |
| Daily | 1.003 (1.000-1.005) | equivalent | 0.979 (0.973-0.986) | equivalent | 1.000 (0.998-1.001) | equivalent |
| Hourly | 0.912 (0.866-0.962) | inconclusive | 1.614 (1.494-1.743) | aeon worse | 0.856 (0.809-0.907) | aeon better |
| **All** | **1.018 (1.015-1.020)** | **equivalent** | **1.033 (1.031-1.035)** | **equivalent** | **1.004 (1.003-1.005)** | **equivalent** |

**SCUM with 036's seasonal periods.** The weekly and hourly SCUM differences
come from the seasonal period. The published code
([M4-methods/036](https://github.com/Mcompetitions/M4-methods/tree/master/036%20-%20fotpetr))
gives weekly series period 52, daily 7 and hourly 168, where these runs use the
M4 benchmark periods (1, 1, 24). Rerun with 036's periods
(`--m4-period weekly=52 daily=7 hourly=168`, output in
`.../aeon-8585c7bf-scum036periods/`; yearly, quarterly and monthly unchanged):

| SCUM / 036 | OWA aeon | OWA 036 | MASE ratio (90% CI) | TOST | signed-rank r | verdict |
|---|---|---|---|---|---|---|
| Weekly | 0.753 | 0.751 | 1.005 (0.979-1.031) | equivalent | +0.01 | no difference |
| Daily | 0.986 | 0.984 | 1.002 (1.000-1.004) | equivalent | +0.11 | R better |
| Hourly | 0.653 | 0.663 | 0.993 (0.965-1.022) | equivalent | +0.01 | no difference |
| **All** | **0.845** | **0.848** | **1.004 (1.003-1.006)** | **equivalent** | | |

The only remaining structural difference is in the published code's one
switch: for frequency above 24 (weekly and hourly) it uses `smooth::es` for
the ETS member instead of `forecast::ets`; aeon uses `AutoETS` throughout.
It also uses seasonal `auto.arima`, where aeon's ARIMA member is non-seasonal.
Neither shows up in accuracy at these periods. The run took 1.7x the CPU time
of the period-1/24 run on daily (3570 against 2081 s), about 2x on weekly.

**Difference** (Wilcoxon signed-rank on per-series MASE, Holm corrected over 12
tests per method; rank-biserial r < 0 means aeon has lower errors;
`signed_rank.csv`, which also has sMAPE)

| | AutoETS: r | verdict | AutoARIMA: r | verdict | SCUM: r | verdict |
|---|---|---|---|---|---|---|
| Yearly | -0.10 | aeon better | +0.10 | R better | -0.06 | aeon better |
| Quarterly | +0.04 | R better | +0.03 | R better | +0.03 | R better |
| Monthly | +0.08 | R better | +0.11 | R better | +0.02 | R better |
| Weekly | +0.03 | no difference | +0.04 | no difference | +0.36 | R better |
| Daily | +0.07 | R better | +0.04 | no difference | -0.00 | no difference |
| Hourly | -0.10 | no difference | +0.39 | R better | -0.37 | aeon better |

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

| mode | comparisons | other toolkit better | no difference | aeon better |
|---|---|---|---|---|
| auto | 22 pooled | 0 | 20 | 2 |
| auto | 132 per process | 0 | 129 | 3 |
| fixed | 22 pooled | 0 | 18 | 4 |
| fixed | 132 per process | 0 | 127 | 5 |

No other toolkit is significantly more accurate than aeon in any of the 308
comparisons. Every "aeon better" verdict is for ETS: against statsforecast,
and in fixed mode at length 100 also against sktime and statsmodels (which
share one engine) and smooth.

**Equivalence** (TOST, ±5%, on per-series MASE ratios; `tost_verdicts.csv`)

| mode | comparisons | equivalent | inconclusive | aeon better | aeon worse |
|---|---|---|---|---|---|
| auto | 22 pooled | 20 | 1 | 1 | 0 |
| auto | 132 per process | 99 | 30 | 3 | 0 |
| fixed | 22 pooled | 18 | 3 | 1 | 0 |
| fixed | 132 per process | 120 | 7 | 5 | 0 |

The four inconclusive pooled comparisons all lean towards aeon: its errors are
lower, but the interval crosses the lower margin. These are ETS against
statsforecast (both modes, length 500) and, in fixed mode at length 100,
against sktime and statsmodels. Per process, 100 series is often too few to
show equivalence within 5%. Of the 37 inconclusive per-process comparisons, 12
lean towards aeon being worse, all at length 100. The largest is AutoETS
against skforecast on ETS(M,A,M) m=12 in auto mode, ratio 1.048
(1.004-1.094).

**Adapter corrections (2026-10-22).** Three harness errors were found and
fixed, and the affected rows rerun at 8585c7bf (provenance in each
`run_info.json` under `refills`):

- aeon's AutoETS was called with `seasonal_period=None` on non-seasonal
  processes, so aeon estimated the period, often found a spurious one, and
  fitted seasonal candidates as well (seasonal models chosen on 11-25% of
  non-seasonal series at length 100, and fits 3-6x slower). It now gets
  period 1, as every other toolkit does; estimating the period remains the
  separate `aeon (auto sp)` entry.
- The statsmodels ARIMA grid could not fit a drift, used statsmodels' KPSS
  lag rule rather than R's, and now also rejects near unit roots as R does
  (statsmodels sometimes converges to degenerate fits with a spuriously low
  AIC).
- The statsmodels ETS grid allowed additive-error, multiplicative-season
  models, which aeon, R, statsforecast and sktime exclude.

None of these changed a pooled verdict. aeon's AutoETS ratios at length 100
fell slightly (against sktime from 0.986 to 0.981). The 1.6.0 and 439ed8bc
runs used the old adapters and are left as they were; comparisons between them
are unaffected, since both used the same adapters.

Against 439ed8bc (`results/comparisons/8585c7bf-vs-439ed8bc/`), no pooled or
per-process accuracy changed significantly in either mode.

## Run time

Median seconds to fit and forecast one series, single threaded, JIT
compilation excluded, 5 repeats, on an otherwise idle machine
(`equivalence/results/aeon-8585c7bf/simulated/timing.csv`). The toolkit columns give
toolkit time / aeon time: above 1, aeon is faster.

| mode | process | n | aeon (s) | pmdarima | skforecast | sktime | smooth | statsforecast | statsmodels |
|---|---|---|---|---|---|---|---|---|---|
| auto | ARMA(1,1) | 100 | 0.027 | 17.8 | 3.1 | 10.6 | 2.7 | 4.2 | 8.0 |
| auto | ARMA(1,1) | 1,000 | 0.163 | 9.7 | 0.28 | 6.5 | 2.0 | 1.4 | 4.7 |
| auto | ARMA(1,1) | 10,000 | 0.973 | 12.9 | 0.12 | 9.2 | 9.5 | 0.82 | 4.1 |
| auto | ETS A,A,A m=12 | 100 | 0.016 | | 14.3 | 36.4 | 5.1 | 3.2 | 37.5 |
| auto | ETS A,A,A m=12 | 1,000 | 0.107 | | 13.7 | 10.1 | 3.0 | 4.2 | 10.1 |
| auto | ETS A,A,A m=12 | 10,000 | 1.058 | | 14.4 | 5.6 | 6.4 | 4.2 | 5.6 |
| fixed | ARMA(1,1) | 100 | 0.001 | 48.5 | 4.2 | 52.5 | 18.5 | 10.1 | 11.3 |
| fixed | ARMA(1,1) | 10,000 | 0.034 | 57.0 | 0.72 | 57.0 | 29.2 | 2.0 | 8.4 |
| fixed | ETS A,A,A m=12 | 100 | 0.001 | | 21.2 | 44.0 | 12.3 | 4.3 | 41.5 |
| fixed | ETS A,A,A m=12 | 10,000 | 0.045 | | 37.5 | 10.8 | 23.1 | 6.9 | 10.7 |

At length 100, timed on 100 series per process instead of 5
(`results/aeon-8585c7bf-n100/`), aeon's AutoARIMA is 3-59x faster than every
other toolkit on all six ARIMA processes, and its AutoETS 2-38x faster on the
seasonal processes. On the three non-seasonal ETS processes aeon and
statsforecast are level (3-4 ms; statsforecast 0.8-1.0x aeon's time) and the
others are 3-10x slower.

- **AutoETS** on seasonal data (ETS(A,A,A), m=12) is faster than every other
  toolkit at every length: 3-7x faster than statsforecast, 5-47x faster than
  the others. On non-seasonal ETS(A,A,N) (length sweep,
  `results/aeon-8585c7bf/sweep-auto`) the margins are small: statsforecast is
  level with aeon up to 500 and 1.4-1.6x slower beyond; statsmodels and sktime
  are 1.7-10x slower up to 1,000, level at 3,000 and 1.2-1.4x faster from
  5,000; skforecast is 3-8x slower throughout. aeon's cost here is its
  optimiser: about 2,000 Nelder-Mead evaluations per fit at length 100
  (three starts per candidate, two thirds of the evaluations on the damped
  models), against a gradient-based fit per candidate in statsmodels.
- **smooth** is slower than aeon throughout: 2-10x for automatic ARIMA and
  3-6x for automatic ETS, and 9-30x with the model given.
- **AutoARIMA** is faster than every other toolkit at 100-200 observations.
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
  length-sweep processes are ARMA(1,1), ARIMA(1,1,0), ETS(A,A,N) and
  ETS(A,A,A) m=12; the length-100 run covers all twelve.

## Rerunning

Paths here are relative to the repository root. Run from the root with the benchmark environment. `AEON` is a clean
git checkout of the aeon to test, such as a worktree; never your development
checkout. Both scripts refuse a checkout with uncommitted changes, use a fresh
numba cache, and name results after the commit.

```bash
equivalence/run_m4.sh        $AEON   # ~5 CPU-hours: M4 runs, then m4.py
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
`submission-{ETS,ARIMA,036,Naive2}.csv` from the
[M4-methods repository](https://github.com/Mcompetitions/M4-methods)).

| file | |
|---|---|
| `run_m4.sh`, `run_simulated.sh` | run aeon, then summarise |
| `m4.py` | M4: OWA, signed-rank and TOST of AutoETS, AutoARIMA and SCUM against the published forecasts |
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

## Theta, DOTM and AutoCES

aeon's other statistical forecasters with an established counterpart, tested
the same way on a smaller scale (`paper/table_other_stats.md`):

| aeon | reference | simulated ratio n=100 | n=500 | M4 ratio | M4 TOST | M4 freq. aeon/ref better | aeon ms n=100 | ref. ms n=100 | speed-up |
|---|---|---|---|---|---|---|---|---|---|
| Theta | statsforecast Theta | 0.999+ | 0.998+ | 1.000 | equivalent | 4/1 | 0.31 | 0.59 | 1.9 |
| Theta | M4 Theta (published) | - | - | 1.000 | equivalent | 3/1 | - | - | - |
| DOTM | statsforecast DOTM | 1.002 | 1.001 | 0.990 | equivalent | 3/2 | 0.50 | 1.49 | 3.0 |
| AutoCES | statsforecast AutoCES | 1.000 | 1.000 | 1.000 | equivalent | 0/0 | 0.63 | 0.72 | 1.2 |
| AutoCES | smooth AutoCES | 1.001- | 1.004- | 1.000 | equivalent | 1/4 | 0.63 | 19.75 | 31.4 |

- **Simulated** (`results/aeon-8585c7bf/sim-other`, `scripts/sim_other_stats.py`):
  the 12 horizon processes, lengths 100 and 500, 100 series each; ratio of
  MASE aeon / reference, + / - where the signed-rank test (Holm) favours aeon /
  the reference. All 8 pooled comparisons are equivalent within 5%, and 94 of
  96 per-process ones (2 inconclusive).
- **M4**: all 100,000 series. Equivalent overall and on every frequency (TOST
  ±5%), except AutoCES against smooth on hourly series: inconclusive, with
  aeon 4% more accurate. aeon's Theta comes from the 439ed8bc run (its code is
  unchanged at 8585c7bf); the reference runs are statsforecast 2.1.1 and
  smooth 1.2.0 (`/local/.../statsforecast-2.1.1`, `.../smooth-1.2.0`), and
  M4's official Theta forecasts.
- **Run time** (`results/aeon-8585c7bf/timing-other`,
  `scripts/timing_other_stats.py`): milliseconds to fit and forecast one
  series of length 100, single-threaded, geometric mean over the 12 processes
  of the median over 100 series; speed-up is reference time / aeon time.
  aeon's Theta is deseasonalised as on M4, and that cost is included.
- aeon's AutoCES and statsforecast's give practically the same forecasts
  (ratio 1.000 with a zero-width interval). smooth's AutoCES, the original
  authors' implementation, is 0.1-0.4% more accurate on simulated series,
  significantly so, and about 30x slower.

**Run time by length** (`paper/table_other_timing.md`; ms per series,
single-threaded, geometric mean over the 12 processes of the median over 100
series at length 100 and 20 at 1,000 and 10,000):

| aeon vs | n = 100 | n = 1,000 | n = 10,000 |
|---|---|---|---|
| Theta vs statsforecast | 0.31 vs 0.59 (1.9x) | 0.63 vs 13.5 (21x) | 3.9 vs 1,867 (481x) |
| DOTM vs statsforecast | 0.50 vs 1.49 (3.0x) | 4.9 vs 38 (7.8x) | 81 vs 5,598 (69x) |
| AutoCES vs statsforecast | 0.63 vs 0.72 (1.2x) | 1.5 vs 3.4 (2.3x) | 12 vs 37 (3.0x) |
| AutoCES vs smooth | 0.63 vs 19.8 (31x) | 1.5 vs 83 (57x) | 12 vs 801 (66x) |

- statsforecast's Theta and DOTM grow roughly with the square of the length
  (Theta: 9 ms, 111 ms, 1.2 s at 1,000, 3,000, 10,000). The time is all in its
  compiled `_lib.theta.optimize`, and a single likelihood pass
  (`pegels_resid`) takes 63 ms at length 10,000, so each evaluation appears to
  be quadratic in the length (the C++ source was not inspected).
- statsforecast's AutoCES fails on all 20 AR(2) series of length 10,000 ("no
  model able to be fitted"); its 10,000 figure is over the other 11 processes.
  aeon's AutoCES fits them.

## Other aeon forecasters on M4

Same commit, all with the M4 seasonal period. None has a published M4
counterpart, so this is accuracy only. OWA:

| | DOTM | AutoCES | CES (deseasonalised) | CES (raw) | SCUM |
|---|---|---|---|---|---|
| Yearly | 0.806 | 0.872 | 0.872 | 0.872 | 0.804 |
| Quarterly | 0.887 | 0.907 | 0.911 | 0.973 | 0.854 |
| Monthly | 0.916 | 0.895 | 0.914 | 1.011 | 0.872 |
| Weekly | 0.957 | 1.005 | 1.005 | 1.005 | 0.902 |
| Daily | 0.997 | 1.019 | 1.019 | 1.019 | 0.983 |
| Hourly | 0.990 | 0.781 | 1.150 | 4.285 | 0.651 |
| **All** | **0.870** | **0.894** | **0.903** | **0.963** | **0.845** |

`CES` is aeon's default `CES(model="N")`, non-seasonal CES, which ignores the
seasonal period. Like the other non-seasonal forecasters it is now run with
`--deseasonalise` (M4-style), output in `.../aeon-8585c7bf-ces-deseason/`.
Without it, CES fails on hourly (MASE above 10 on 139 of 414 series); with it,
hourly is still worse than Naive2 (OWA 1.150) and well behind AutoCES
(0.781). This is the method, not aeon's code: statsforecast
`AutoCES(model="N")` fitted to the same deseasonalised hourly series gives
the same forecasts (median MASE ratio 1.000000, mean MASE 2.580 for both).
Non-seasonal CES is worse than Naive2 on 79% of hourly series. On about a
quarter of them its forecasts drift downwards over the 48 steps (5th
percentile: the second day's mean is 0.63 times the first's), and the fitted
real smoothing parameter sits at its upper bound (1.8) on 326 of 414 series.
AutoCES selects among the non-seasonal and
seasonal forms (N, S, P, F) itself, so it is run on the raw series, as are
DOTM and SCUM. Yearly, weekly and daily have period 1, so all CES variants are
identical there.
