# horizon

Experiments comparing aeon's statistical forecasters with other forecasting
toolkits, and the evaluation behind changes proposed to aeon.

The current experiments compare automatic ARIMA and ETS model selection in
[aeon](https://github.com/aeon-toolkit/aeon) with
[statsforecast](https://github.com/Nixtla/statsforecast),
[sktime](https://github.com/sktime/sktime),
[skforecast](https://github.com/skforecast/skforecast),
[pmdarima](https://github.com/alkaline-ml/pmdarima) and
[statsmodels](https://github.com/statsmodels/statsmodels), on series simulated
from known processes. They ask two questions:

1. **Accuracy.** Are aeon's forecasts significantly different from the other
   toolkits'? Paired Wilcoxon signed-rank tests on per-series MASE, Holm
   corrected, plus critical difference diagrams.
2. **Speed.** How does the time to fit and forecast grow with series length?

## Install

```bash
pip install -e ".[toolkits]"
```

aeon is a plain dependency. To evaluate a development version of aeon, install
it in the same environment (e.g. `pip install -e path/to/aeon`); every run
records which aeon it used (version, path, git commit and branch) in
`run_info.json`.

## Running

```bash
python -m horizon run --mode auto  --out results/<name>/auto    # model selection
python -m horizon run --mode fixed --out results/<name>/fixed   # true model given
python -m horizon run --mode auto  --out scratch/try --quick    # minutes, smoke test
python -m horizon analyse results/<name>/auto                   # rebuild tables/plots
python -m horizon compare results/<a>/auto results/<b>/auto --out results/comparisons/<b>-vs-<a>
```

A full run takes about an hour on one core per mode. Runs append to their
output directory and skip finished rows, so an interrupted run can be resumed by
rerunning the same command.

`--mode auto` gives each toolkit only the series (and, for ETS, the true
seasonal period). `--mode fixed` gives each toolkit the true generating model,
so differences come only from parameter estimation.

Notes for comparable numbers:

- All toolkits are pinned to one thread. JIT compilation (aeon, statsforecast)
  is paid before any timed call.
- Every toolkit sees exactly the same series; seeds are fixed, so two runs with
  different software versions pair up series by series.
- numba caches compiled code on disk and does not always notice when a function
  it depends on in another file has changed. When comparing aeon versions from
  the same source tree, set `NUMBA_CACHE_DIR` to a fresh directory per run.

## Layout

```
src/horizon/
  simulate.py      generating processes (ARIMA and ETS) and their true models
  adapters.py      one adapter per toolkit and mode, with the fairness notes
  experiments.py   accuracy and timing experiments
  analyse.py       summaries, significance tests, plots for one run
  compare.py       comparisons between runs
  cli.py           python -m horizon
tests/             smoke tests (pytest)
results/<run>/     one directory per run and mode, see below
reports/           written comparisons
paper/             Section 5 of the aeon paper: how to reproduce it, M4 and LSSH-99 league tables
```

Each result directory holds `run_info.json`, the raw per-series
`accuracy.csv.gz` and `timing.csv`, the derived `accuracy_summary.csv`,
`significance.csv`, `timing_summary.csv`, and `plots/`. Raw point forecasts
(`forecasts.csv`) are not committed; rerun to regenerate them.

## Results

| run | aeon |
| --- | --- |
| `results/aeon-1.6.0/` | release 1.6.0 |
| `results/aeon-fix-439ed8bc/` | branch `fix/forecasting-arima-ets-estimation` at 439ed8bc (ARIMA/ETS estimation fixes); comparison with 1.6.0 in `results/comparisons/fix-439ed8bc-vs-1.6.0/` |
| `results/aeon-8585c7bf/` | `bench/arima-ets` at 8585c7bf (`ajb/arima` PR #3931 with R's KPSS lag rule, merged with `ajb/ets` with `AutoETS(restrict=True)`); comparison with 439ed8bc in `results/comparisons/8585c7bf-vs-439ed8bc/` |

See `reports/` for the written comparisons, and `equivalence/` for the evidence
that aeon's AutoETS and AutoARIMA match the published M4 forecasts of R's
`ets` and `auto.arima` and the other Python toolkits, with run times.
