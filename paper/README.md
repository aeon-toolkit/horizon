# Reproducing Section 5 of the aeon paper

Section 5 (Benchmarking and evaluation) of the aeon forecasting paper. Every
number in it comes from aeon commit `8585c7bf` (branch `bench/arima-ets`, a
clean worktree; no aeon code changed). The paper calls this aeon 1.7: the
forecasting code at 8585c7bf ships unchanged in that release. The results are fixed: runs started
after 2026-10-27 are not part of the paper.

League tables, every forecaster run:

- [M4 league table](m4_league.md) (Section 5.2, 26 entries)
- [LSSH-99 league table](lssh99_league.md) (Section 5.3, 33 forecasters)

## Rebuilding the tables from stored results

```bash
paper/make_tables.sh      # a few minutes; no forecasting
```

This rebuilds every table, figure and league page below from the stored
results, and reproduces the committed files exactly (figure PDFs differ only
in their creation date). Some steps read the per-series output under
`/local/ajb2u23/Results/Forecasting` (`$RESULTS`) and the M4 and LSSH-99
data under `/local/ajb2u23/Data/Forecasting`. Those files are too large for the
repository. The LSSH-99 league (`scripts/lssh99_league.py`) and the 5.1 summaries
read only committed files.

`paper/env.sh` sets the environment: `AEON` (the worktree, checked against
the commit before every run), the conda env, `PYTHONPATH`, a fresh numba
cache, and TensorFlow's CUDA libraries.

## Rerunning the experiments

| Section | Script | Cost |
|---|---|---|
| 5.1 Simulated series | `paper/run_simulated.sh` | about a day; timing steps need an idle machine |
| 5.2 M4 | `paper/run_m4.sh cpu`, `paper/run_m4.sh gpu K F` | ~100 CPU-hours (TVP ~70 of them); several GPU-days per deep learner |
| 5.3 LSSH-99 | `paper/run_lssh99.sh cpu`, `paper/run_lssh99.sh gpu K` (K = 0-3) | ~600 CPU-hours (about a day on 32 cores; DrCIF 190, CIF 100); ~110 GPU-worker-hours for the deep learners |

All runs resume where they stopped. Then run `paper/make_tables.sh`.

## 5.1 Forecasting simulated series

| In the paper | File | Built by | From |
|---|---|---|---|
| Table `tab:toolkits` (toolkits and versions) | in `main.tex` | by hand | versions in `results/aeon-8585c7bf/sweep-auto-r20/run_info.json` |
| Figures `fig:timing`, `fig:timing-more` | `equivalence/paper/fig_timing_{arima,ets}.pdf` | `equivalence/paper.py` | `results/aeon-8585c7bf/sweep-auto-r20/` (`python -m horizon sweep --mode auto --only timing --repeats 20`) |
| Run-time statements in the text (speed-ups at length 100) | `equivalence/README.md`, `results/aeon-8585c7bf/{auto,fixed}/timing_summary.csv` | `python -m horizon run` | `results/aeon-8585c7bf/{auto,fixed}`; smooth in `results/aeon-8585c7bf-smooth/` |
| Table `tab:other_stats` (Theta, DOTM, AutoCES) | `equivalence/paper/table_theta_dotm_ces.tex` | by hand from `table_other_stats.tex` and `table_other_timing.tex` (`equivalence/paper.py`) | `results/aeon-8585c7bf/sim-other/` (`scripts/sim_other_stats.py`), `results/aeon-8585c7bf/timing-other/` (`scripts/timing_other_stats.py`), M4 column from `equivalence/results/aeon-8585c7bf/m4/` |
| statsforecast main (PR #1232) speed-ups in the text | `results/aeon-8585c7bf/timing-sf-main/` | `scripts/timing_other_stats.py --entries ...` with statsforecast 2.1.1 and with main | |

Detail and verdicts: [`equivalence/README.md`](../equivalence/README.md).

**Gap.** The sentence "no significant difference in forecast quality at each
series length" has no accuracy sweep at 8585c7bf behind it. The 8585c7bf
sweeps (`sweep-auto`, `sweep-auto-r20`) are timing only. The per-length
accuracy tests were run at 439ed8bc with the older adapters
(`results/aeon-fix-439ed8bc/sweep-{auto,fixed}/equivalence_clean.csv`). At
8585c7bf the accuracy tests cover lengths 100 and 500 only
(`results/aeon-8585c7bf/{auto,fixed}/significance.csv`).

## 5.2 M4

| In the paper | File | Built by |
|---|---|---|
| Table `tab:m4` | `paper/tab_m4.tex` (tabular body) | `scripts/m4_league.py` |
| Every forecaster | [`paper/m4_league.md`](m4_league.md), `results/aeon-8585c7bf/m4/league.csv` | `scripts/m4_league.py` |
| Equivalence with R `ets`, `auto.arima` and SCUM (text) | `equivalence/results/aeon-8585c7bf/m4/`, `equivalence/paper/table_m4.tex` | `equivalence/m4.py`, `equivalence/paper.py` |
| SCUM with entry 036's periods (weekly 0.753 vs 0.751) | `equivalence/README.md` | `$RESULTS/aeon-8585c7bf-scum036periods/` |
| First-step gap of the deep learners (text) | `results/aeon-8585c7bf/m4/owa_by_steps.csv` | `scripts/m4_by_steps.py` |

Forecasts: `scripts/m_comp_experiments.py`, one result file per forecaster in
`$RESULTS/aeon-8585c7bf/aeon/<F>/` (deep learners in `aeon-8585c7bf-dl`). Each
file records its arguments. Non-seasonal, machine learning and deep learning
forecasters use `--deseasonalise`. SCUM, DOTM, AutoCES, AutoETS and CES get
the raw series and the M4 period. The published forecasts (`submission-{Naive2,ETS,ARIMA,Theta,036}.csv`,
[M4-methods](https://github.com/Mcompetitions/M4-methods)) and test sets are in
`/local/ajb2u23/Data/Forecasting/m4/official`.

The deep learners (`TCNDiff`, `DeepARDiff`, `NBeatsDiff`) are aeon's TCN,
DeepAR and N-BEATS inside `StandardisedForecaster` (in `m_comp_experiments.py`).
It standardises the first differences of the last 1,000 observations and fits
once. As shipped they failed a 600-series pilot
(`results/aeon-8585c7bf/m4/owa_dl_pilot_600.csv`).

## 5.3 LSSH-99

| In the paper | File | Built by |
|---|---|---|
| Table `tab:lssh` | `paper/tab_lssh.tex` (tabular body) | `scripts/lssh99_league.py` |
| Every forecaster | [`paper/lssh99_league.md`](lssh99_league.md), `results/aeon-8585c7bf/aaltd/league.csv` | `scripts/lssh99_league.py` |
| Scores per series | `results/aeon-8585c7bf/aaltd/{per_problem,summary}.csv` | `scripts/aaltd_score.py --forecasters paper/lssh99_forecasters.txt` |

Forecasts: `scripts/aaltd_experiments.py` (30 rolling one-step refits per
series), rows in `$RESULTS/aaltd-8585c7bf/{cpu,dl-gpu0..3}/<F>.csv`. The
forecaster set is fixed in `paper/lssh99_forecasters.txt`. The scorer refuses
to run if any listed forecaster has fits missing, and ignores everything else
in the results directory: runs started later (the deep-learning (Δy)
regressors, RandomInterval, KNN, RDST, RIST and the Rockets) cannot change the
scores. QUANT (Δy) is not included: it needs torch, which the environment
lacks.

Data: LSSH-99 (the aaltd collection) in `/local/ajb2u23/Data/Forecasting/aaltd`,
one directory per series with `_TRAIN.csv` and `_TEST.csv`, joined into the
full series. 100 series are stored; wind_farms T2 is constant and is not part
of the collection.

## Not in the paper

Exploratory analyses (by series length, forecast step, equal frequency
weights, LSSH-99 without seasonality) are in
[`notes/m4-aaltd-investigation.md`](../notes/m4-aaltd-investigation.md), with
scripts `scripts/m4_by_length.py`, `scripts/m4_by_steps.py` and
`scripts/aaltd_noseason.py`.
