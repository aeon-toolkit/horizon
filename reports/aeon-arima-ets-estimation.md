# aeon ARIMA and ETS estimation: 1.6.0 against the fix branch

**Status: draft for review.** Branch results from `results/aeon-fix-439ed8bc/`
(head `439ed8bc`, all three ARIMA/ETS fix PRs combined); 1.6.0 results from
`results/aeon-1.6.0/` (release commit `b694b1c03`).

## Summary

On simulated series from known ARIMA and ETS processes, aeon 1.6.0's AutoARIMA
and AutoETS were significantly less accurate than statsforecast, sktime,
skforecast, pmdarima and statsmodels: pooled over processes, every competitor
was significantly better at both series lengths, with automatic model selection
and when every toolkit was given the true model. The causes were bugs and
weaknesses in model estimation, not in the model classes. With the changes on
the branch `fix/forecasting-arima-ets-estimation`, no competitor is
significantly better than aeon in any comparison, though aeon's AutoETS still
selects the true model less often than the competitors and loses more accuracy
than they do by selecting on short series. aeon remains faster than
statsforecast, sktime, pmdarima and statsmodels on most processes and lengths,
but each automatic fit costs about 2.5-3x more than in 1.6.0, skforecast's ARIMA
is now faster on series longer than a few hundred observations, and
statsforecast's AutoETS is faster on short non-seasonal series.

## Experiment

`python -m horizon run --mode {auto,fixed}`; see the README.

- **Processes.** Six ARIMA processes (ARMA(1,1), AR(2), MA(2), white noise,
  ARIMA(1,1,0), ARIMA(0,1,1) with drift) and six ETS processes (A,N,N; A,A,N;
  A,Ad,N; A,N,A m=4; A,A,A m=12; M,A,M m=12). `src/horizon/simulate.py`.
- **Series.** 100 per process and length, lengths 100 and 500, horizon 12 or
  two seasons. Every toolkit and both aeon versions see identical series.
- **Toolkits.** aeon AutoARIMA/AutoETS (ETS also with the period estimated),
  statsforecast, sktime, skforecast, pmdarima (ARIMA only), statsmodels (AIC
  grid search written in the adapter). Same order limits for every auto-ARIMA
  (p <= 3, d <= 2, q <= 2), non-seasonal ARIMA throughout because aeon's
  AutoARIMA is non-seasonal, and the true seasonal period given for ETS.
- **Modes.** `auto`: each toolkit selects the model. `fixed`: each toolkit fits
  the true model, so differences come only from parameter estimation.
- **Accuracy.** MASE per series; paired two-sided Wilcoxon signed-rank tests of
  aeon against each competitor, Holm corrected, pooled over processes and per
  process.
- **Run time.** Fit plus forecast, single threaded, JIT compilation excluded;
  ARMA(1,1) and ETS(A,A,A) m=12 for lengths 100 to 10,000, 5 repeats.

## What changed in aeon

| commit | change | why |
| --- | --- | --- |
| ETS initial states | per-step initial trend, level at the end of the first season, detrended seasonal indices, OLS start for non-seasonal trend | the initial trend was the change per season (m times too large); fits compensated by collapsing phi to 0.3-0.5 |
| ETS damped_trend | `ETS(damped_trend=...)`, AutoETS searches damped and undamped trends | phi was always estimated, so an undamped trend could not be fitted |
| ETS multi-start | three Nelder-Mead runs (default start, R-style start, restart from the best) | a single run stopped >0.1 AIC above the optimum in 25% of candidate fits |
| ARIMA AIC | CSS likelihood over all n observations | over n - max(p,q) it favoured large orders; white noise was identified 7% of the time |
| ARIMA constant start | constant starts at the series mean | from ~0.025 with a 0.005 step, fits ran out of iterations (all AR(2) fits missed the optimum) |
| ARIMA constraints and multi-start | stationary/invertible region only; Hannan-Rissanen start and restart; AutoARIMA rejects roots within 1.01 of the unit circle | local minima, and spurious non-invertible optima that a better optimiser otherwise selects |
| fastmath | no full fastmath where inf/NaN are handled | undefined behaviour of isfinite and comparisons with inf |
| convergence | Nelder-Mead convergence checked every iteration, on values and vertices | the check ran only after shrink steps, 25-40% wasted iterations |

The optimiser studies in `studies/` give the evidence for each design choice.

## Results

Commands: `python -m horizon compare results/aeon-1.6.0/<mode>
results/aeon-fix-439ed8bc/<mode>` (outputs in
`results/comparisons/fix-439ed8bc-vs-1.6.0/<mode>/`) and `python -m horizon
compare results/aeon-fix-439ed8bc/auto results/aeon-fix-439ed8bc/fixed
--auto-fixed`. Both runs used the same seeds, so aeon 1.6.0, the branch and
every competitor forecast identical series. All tables give median MASE over
the 600 (pooled) or 100 (per process) series; "competitors" is the range of the
competitors' medians; the p-value is a paired Wilcoxon test of 1.6.0 against
the branch, Holm corrected within each mode.

### Accuracy, pooled over processes

The branch is significantly more accurate than 1.6.0 in all eight pooled
comparisons (two families, two lengths, two modes), and its median now lies
inside or below the competitors' range.

**auto** (each toolkit selects the model)

| family | n | 1.6.0 | branch | competitors | p (Holm) | change |
| --- | --- | --- | --- | --- | --- | --- |
| ARIMA | 100 | 1.347 | 1.210 | 1.190-1.218 | <0.001 | branch better |
| ARIMA | 500 | 1.376 | 1.172 | 1.169-1.191 | <0.001 | branch better |
| ETS | 100 | 1.025 | 0.909 | 0.894-0.938 | <0.001 | branch better |
| ETS | 500 | 0.917 | 0.819 | 0.813-0.858 | <0.001 | branch better |

**fixed** (each toolkit fits the true model)

| family | n | 1.6.0 | branch | competitors | p (Holm) | change |
| --- | --- | --- | --- | --- | --- | --- |
| ARIMA | 100 | 1.294 | 1.167 | 1.166-1.167 | <0.001 | branch better |
| ARIMA | 500 | 1.342 | 1.193 | 1.189-1.191 | <0.001 | branch better |
| ETS | 100 | 1.028 | 0.869 | 0.881-0.912 | <0.001 | branch better |
| ETS | 500 | 0.917 | 0.822 | 0.813-0.853 | <0.001 | branch better |

Verdicts of aeon against each competitor (paired Wilcoxon, Holm corrected,
from each run's `significance.csv`; 1.6.0 verdicts from
`results/aeon-1.6.0/<mode>/significance.csv`):

| mode | comparisons | aeon 1.6.0 | branch 439ed8bc |
| --- | --- | --- | --- |
| auto | pooled, 18 vs competitors | 18 competitor better | 16 no difference, 2 aeon better |
| auto | per process, 108 vs competitors | 60 competitor better, 48 no difference | 104 no difference, 4 aeon better |
| fixed | pooled, 18 | 18 competitor better | 14 no difference, 4 aeon better |
| fixed | per process, 108 | 30 competitor better, 78 no difference | 101 no difference, 7 aeon better |

(auto also compares aeon with its own estimated-period ETS variant, `aeon (auto
sp)`: no significant difference in 1.6.0 or on the branch, pooled or per
process. Counting those, auto has 134 no difference and 6 aeon better.)

On the branch no competitor is significantly better than aeon in any pooled or
per-process comparison in either mode. Where aeon is significantly better, it
is mostly against statsforecast's ETS (pooled at both lengths in both modes,
and per process on the m=12 seasonal processes); in fixed mode at length 100
aeon's ETS is also better than sktime and statsmodels pooled (both wrap
statsmodels' ETSModel), and in auto mode at length 100 aeon's ARIMA beats the
statsmodels grid search on AR(2).

### Accuracy, per process

No process got significantly worse. The improvements are concentrated where
the 1.6.0 bugs bit: AR(2), MA(2) and white noise for ARIMA (the CSS-AIC and
constant-start bugs), and the trended and seasonal ETS processes (the initial
trend bug). A few medians rose without a significant change, most visibly ETS
A,A,N at length 100 in auto mode (1.191 to 1.420; the competitors' range is
1.360-1.447, so 1.6.0's median there was below every competitor's).

**auto**

| family | n | process | 1.6.0 | branch | competitors | p (Holm) | change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARIMA | 100 | ARMA(1,1) | 1.693 | 1.478 | 1.440-1.473 | 0.110 | none |
| ARIMA | 100 | AR(2) | 0.992 | 0.928 | 0.925-0.956 | <0.001 | branch better |
| ARIMA | 100 | MA(2) | 1.228 | 1.021 | 1.012-1.026 | <0.001 | branch better |
| ARIMA | 100 | ARIMA(1,1,0) | 3.767 | 3.867 | 3.740-4.120 | 0.631 | none |
| ARIMA | 100 | ARIMA(0,1,1)+drift | 2.496 | 2.585 | 2.334-2.660 | 0.974 | none |
| ARIMA | 100 | white noise | 0.761 | 0.710 | 0.712-0.723 | <0.001 | branch better |
| ARIMA | 500 | ARMA(1,1) | 1.630 | 1.565 | 1.564-1.594 | 0.006 | branch better |
| ARIMA | 500 | AR(2) | 0.901 | 0.820 | 0.813-0.829 | 0.001 | branch better |
| ARIMA | 500 | MA(2) | 1.219 | 0.990 | 0.992-1.003 | <0.001 | branch better |
| ARIMA | 500 | ARIMA(1,1,0) | 3.793 | 3.596 | 3.251-3.527 | 0.958 | none |
| ARIMA | 500 | ARIMA(0,1,1)+drift | 2.639 | 2.770 | 2.746-2.757 | 0.958 | none |
| ARIMA | 500 | white noise | 0.752 | 0.712 | 0.709-0.716 | 0.001 | branch better |
| ETS | 100 | A,N,N | 0.964 | 1.008 | 0.941-0.960 | 0.166 | none |
| ETS | 100 | A,A,N | 1.191 | 1.420 | 1.360-1.447 | 1.000 | none |
| ETS | 100 | A,Ad,N | 1.053 | 1.122 | 1.087-1.093 | 1.000 | none |
| ETS | 100 | A,N,A m=4 | 0.878 | 0.892 | 0.895-0.902 | 1.000 | none |
| ETS | 100 | A,A,A m=12 | 1.232 | 0.620 | 0.590-0.797 | <0.001 | branch better |
| ETS | 100 | M,A,M m=12 | 1.115 | 0.544 | 0.537-0.721 | <0.001 | branch better |
| ETS | 500 | A,N,N | 0.898 | 0.898 | 0.898-0.904 | 1.000 | none |
| ETS | 500 | A,A,N | 0.972 | 0.879 | 0.895-0.895 | 0.015 | branch better |
| ETS | 500 | A,Ad,N | 0.975 | 0.968 | 0.965-0.970 | 0.255 | none |
| ETS | 500 | A,N,A m=4 | 0.910 | 0.881 | 0.886-0.919 | 1.000 | none |
| ETS | 500 | A,A,A m=12 | 0.676 | 0.317 | 0.328-0.333 | <0.001 | branch better |
| ETS | 500 | M,A,M m=12 | 1.175 | 0.757 | 0.728-0.943 | <0.001 | branch better |

**fixed**

| family | n | process | 1.6.0 | branch | competitors | p (Holm) | change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARIMA | 100 | ARMA(1,1) | 1.588 | 1.430 | 1.444-1.444 | 0.260 | none |
| ARIMA | 100 | AR(2) | 1.135 | 0.938 | 0.936-0.937 | <0.001 | branch better |
| ARIMA | 100 | MA(2) | 1.039 | 1.021 | 1.025-1.025 | 1.000 | none |
| ARIMA | 100 | ARIMA(1,1,0) | 3.351 | 3.351 | 3.351-3.351 | 1.000 | none |
| ARIMA | 100 | ARIMA(0,1,1)+drift | 2.357 | 2.357 | 2.378-2.401 | 1.000 | none |
| ARIMA | 100 | white noise | 0.713 | 0.713 | 0.713-0.713 | 1.000 | none |
| ARIMA | 500 | ARMA(1,1) | 1.556 | 1.565 | 1.571-1.575 | 0.399 | none |
| ARIMA | 500 | AR(2) | 1.077 | 0.820 | 0.818-0.818 | <0.001 | branch better |
| ARIMA | 500 | MA(2) | 1.010 | 0.988 | 0.989-0.989 | 0.067 | none |
| ARIMA | 500 | ARIMA(1,1,0) | 3.519 | 3.519 | 3.517-3.519 | 0.010 | branch better |
| ARIMA | 500 | ARIMA(0,1,1)+drift | 2.394 | 2.394 | 2.409-2.415 | 0.399 | none |
| ARIMA | 500 | white noise | 0.709 | 0.709 | 0.709-0.709 | 0.556 | none |
| ETS | 100 | A,N,N | 0.927 | 0.927 | 0.940-0.941 | 1.000 | none |
| ETS | 100 | A,A,N | 1.348 | 1.247 | 1.170-1.258 | 0.458 | none |
| ETS | 100 | A,Ad,N | 1.024 | 1.089 | 1.007-1.013 | 1.000 | none |
| ETS | 100 | A,N,A m=4 | 0.859 | 0.859 | 0.884-0.884 | 1.000 | none |
| ETS | 100 | A,A,A m=12 | 1.176 | 0.614 | 0.608-0.776 | <0.001 | branch better |
| ETS | 100 | M,A,M m=12 | 1.115 | 0.558 | 0.545-0.664 | <0.001 | branch better |
| ETS | 500 | A,N,N | 0.898 | 0.898 | 0.898-0.898 | 0.283 | none |
| ETS | 500 | A,A,N | 0.972 | 0.897 | 0.895-0.895 | 0.016 | branch better |
| ETS | 500 | A,Ad,N | 0.977 | 0.968 | 0.965-0.970 | 0.283 | none |
| ETS | 500 | A,N,A m=4 | 0.883 | 0.883 | 0.886-0.886 | 0.527 | none |
| ETS | 500 | A,A,A m=12 | 0.681 | 0.326 | 0.326-0.333 | <0.001 | branch better |
| ETS | 500 | M,A,M m=12 | 1.175 | 0.700 | 0.674-0.953 | <0.001 | branch better |

Boxplots: `results/aeon-fix-439ed8bc/<mode>/plots/mase_boxplot_<family>_n<n>.png`
(and the same paths under `results/aeon-1.6.0/`); critical difference diagrams
`cd_<family>_n<n>.png`.

Auto against fixed (the cost of model selection): at length 500 no toolkit,
aeon included, is significantly more accurate given the true model. At length
100, for ARIMA every toolkit is significantly more accurate in fixed mode, with
similar gaps (aeon 1.210 vs 1.167, competitors 1.190-1.218 vs 1.166-1.167). For
ETS at length 100 aeon's gap is the largest (0.909 vs 0.869, Holm p <0.001);
the competitors' gaps are 0.001-0.027 and none is significant after
correction. So aeon's ETS selection costs more accuracy on short series than
the competitors' does, consistent with its lower true-model rates below, even
though its auto-mode accuracy is not significantly different from theirs.

### Model selection

Rate at which auto mode selects the true model, averaged over processes
(`accuracy_summary.csv`, column `true_model_rate`):

| family | n | aeon 1.6.0 | branch | competitors |
| --- | --- | --- | --- | --- |
| ARIMA | 100 | 0.12 | 0.38 | 0.36-0.46 |
| ARIMA | 500 | 0.15 | 0.51 | 0.49-0.61 |
| ETS | 100 | 0.37 | 0.43 | 0.46-0.54 |
| ETS | 500 | 0.72 | 0.76 | 0.78-0.83 |

ARIMA: 1.6.0 rarely found the true orders (AR(2) 1%, ARMA(1,1) 3-10%, white
noise 4-15%) because the AIC over n - max(p,q) observations favoured large
orders. The branch is within or above the competitors' range on every process
and length except ARIMA(0,1,1) with drift at length 100 (0.35 against
0.38-0.61) and ARIMA(1,1,0) at length 500 (0.44 against 0.50-0.69); accuracy
on both is not significantly different from the competitors'.

ETS: the 1.6.0 rates are not comparable process by process. 1.6.0 always
estimated phi but labelled the trend "A", so it scored 100% on A,A,N and 0% on
A,Ad,N whatever it fitted. The branch chooses between damped and undamped
trends, and its misses on trended processes are mostly a damped trend chosen for
an undamped one (at length 500, A,A,N: 85 A,A,N and 15 A,Ad,N; A,A,A m=12: 81
A,A,A and 18 A,Ad,A). Even so, the branch's ETS selection is weaker than the
competitors'. Its rate is below their range on A,N,N (0.21 and 0.43 against
0.41-0.45 and 0.49-0.65; the commonest wrong choice is A,Ad,N), A,A,N (0.51 and
0.85 against 0.58-0.59 and 0.88-0.89), A,N,A m=4 at 100 (0.51 against
0.56-0.69), M,A,M m=12 at 100 (0.55 against 0.60-0.83) and A,A,A m=12 at 500
(0.81 against 0.87-0.96). It is within or above their range on A,Ad,N, A,A,A
m=12 at 100, and A,N,A m=4 and M,A,M m=12 at 500. None of these per-process
differences is a significant difference in accuracy, but the auto-fixed gap
above shows the weaker selection does cost accuracy on short series. The cause
has not been investigated.

### Run time

Median seconds per fit and forecast, single threaded, 5 repeats, JIT
compilation excluded (`timing_change.csv`). The branch against 1.6.0:

| mode | process | n=100 | n=1000 | n=10,000 | branch / 1.6.0 |
| --- | --- | --- | --- | --- | --- |
| auto | ARMA(1,1) | 0.008 → 0.021 | 0.034 → 0.092 | 0.27 → 0.81 | 2.5-3.2x |
| auto | ETS A,A,A m=12 | 0.007 → 0.020 | 0.044 → 0.124 | 0.44 → 1.23 | 2.3-2.8x |
| fixed | ARMA(1,1) | 0.0005 → 0.0010 | 0.0018 → 0.0040 | 0.014 → 0.035 | 1.8-2.5x |
| fixed | ETS A,A,A m=12 | 0.0008 → 0.0009 | 0.0035 → 0.0051 | 0.032 → 0.045 | 1.2-1.4x |

The cost comes from the multi-start optimisation (three Nelder-Mead runs for
ETS, Hannan-Rissanen start and restart for ARIMA) and, in auto mode, the larger
ETS search (damped and undamped trends).

Speed of the branch relative to each competitor (competitor time / aeon time,
from `timing_summary.csv`; above 1 means aeon is faster):

| mode | process | n | pmdarima | skforecast | sktime | statsforecast | statsmodels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| auto | ARMA(1,1) | 100 | 22.7 | 4.0 | 13.8 | 5.2 | 10.0 |
| auto | ARMA(1,1) | 1000 | 17.2 | 0.50 | 11.3 | 2.4 | 8.3 |
| auto | ARMA(1,1) | 10,000 | 15.5 | 0.14 | 11.1 | 0.98 | 4.9 |
| auto | ETS A,A,A m=12 | 100 | | 11.2 | 28.5 | 2.5 | 36.9 |
| auto | ETS A,A,A m=12 | 10,000 | | 12.1 | 4.8 | 3.6 | 5.8 |
| fixed | ARMA(1,1) | 100 | 49.4 | 4.4 | 53.3 | 10.1 | 11.3 |
| fixed | ARMA(1,1) | 10,000 | 54.7 | 0.66 | 54.6 | 1.9 | 8.0 |
| fixed | ETS A,A,A m=12 | 100 | | 21.1 | 43.7 | 4.3 | 40.8 |
| fixed | ETS A,A,A m=12 | 10,000 | | 38.6 | 10.9 | 6.8 | 10.9 |

aeon is faster than every competitor on these processes with two exceptions:
skforecast's auto-ARIMA is faster from about length 500 (7x at 10,000), and in
fixed mode from about length 5,000; and statsforecast's AutoARIMA is level with
aeon at length 10,000. The length sweep (`results/aeon-fix-439ed8bc/sweep-auto`,
timing on four processes) adds non-seasonal ETS A,A,N, where statsforecast's
AutoETS is about 4x faster than aeon at length 100 in auto mode (1.15x in
fixed mode) and aeon about 1.5x faster from length 1,000, and sktime and
statsmodels are 1.2-1.3x faster than aeon at length 10,000 in auto mode. Plots: `results/comparisons/fix-439ed8bc-vs-1.6.0/<mode>/plots/`.

## Caveats

- Simulated data from the model classes being fitted. This tests estimation
  and selection, not robustness to misspecification; a benchmark on real data
  (e.g. M4) is a natural next step.
- The ETS comparison gives each toolkit the true seasonal period; aeon's
  estimated-period variant is reported separately.
- Run times are single threaded on one machine.
