# Optimiser studies

Targeted experiments behind the ARIMA/ETS estimation changes proposed to aeon.
Unlike the main benchmark they look inside the fitting: how close aeon gets to
the optimum of its own loss, and which starting points and stopping rules get
there. Scripts that use helpers only present on the aeon branch say so.

Run with the aeon to be studied importable, and a fresh `NUMBA_CACHE_DIR` per
run (numba does not invalidate cached callers when a dependency in another file
changes, which silently mixes old and new code).

| script | aeon | question |
| --- | --- | --- |
| `selection_rates.py` | any | how often AutoARIMA selects the true order |
| `optimiser_gap.py` | any | how far fits are from the optimum of aeon's loss |
| `arima_start_strategies.py` | any | ARIMA starting simplices and restarts |
| `ets_start_strategies.py` | branch | ETS starting simplices and restarts |
| `stopping_rules.py` | branch | Nelder-Mead stopping rules, single runs |
| `fit_loss_and_time.py` | branch | loss and time of the full multi-start fits |

## Findings (aeon 1.6.0 unless stated)

**ARIMA, fixed order (`optimiser_gap.py`, 100 series of length 500).** Fits miss
the minimum of their own loss on 61% of ARMA(1,1) and MA(2) series and all AR(2)
series (median AIC gap 227). The constant starts at ~0.025 with a 0.005 step,
so Nelder-Mead runs out of iterations before reaching the series mean. Starting
the constant at the mean removes these failures.

**AutoARIMA (`selection_rates.py`).** The CSS likelihood was evaluated over
n - max(p, q) observations, so larger orders gained about as much per lag as
the AIC penalty takes away. True-order rates for ARMA(1,1), AR(2), MA(2), white
noise and ARIMA(1,1,0): 7%, 0%, 10%, 7%, 25% in 1.6.0; 48%, 45%, 66%, 69%, 55%
on the branch (statsforecast 54%, 53%, 61%, 56%, 69%).

**ARIMA starting strategies (`arima_start_strategies.py`).** Against an
unconstrained reference, failure rates (gap > 0.1): 1.6.0 default 67%, mean
start 27%, Hannan-Rissanen 19.5%, best of both plus restart 16%. Much of the
remainder is the reference reaching non-invertible optima (e.g. airline
ARIMA(1,1,1), theta = -1.11). With a better optimiser but no constraint,
AutoARIMA selected more overparameterised, near non-invertible models, which is
why the branch constrains fits to the stationary, invertible region and, as R's
auto.arima, rejects candidates with roots within 1.01 of the unit circle.

**ETS (`optimiser_gap.py`, `ets_start_strategies.py`).** In 1.6.0 Nelder-Mead
usually does reach the optimum of its loss, but the loss is wrong: the initial
trend was the change per season rather than per step, so fits collapsed phi to
0.3-0.5. With correct initial states, a single run still stops in a poor local
minimum often: gap > 0.1 for 25% of candidate fits (95th percentile 110). The
branch's three-run fit (default start, R-style start, restart from the best)
reduces this to 2.5% (95th percentile 0.005).

**Stopping rule (`stopping_rules.py`, `fit_loss_and_time.py`).** The 1.6.0
convergence test runs only after a shrink step, so runs continue 25-40% past
convergence. Checking value spread every iteration stops in false minima (7.6%
of ARIMA fits worse); also requiring the vertices to have closed up keeps
results within 0.01 AIC for 99% of fits, with ARIMA fits 8% and ETS fits 22%
faster.
