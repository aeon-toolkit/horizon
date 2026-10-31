# Length-100 timing on 100 series per process (aeon 8585c7bf)

- `auto/timing.csv`: fit-and-forecast time of every toolkit (auto mode) on 100
  series of length 100 for each of the 12 processes, made with
  `horizon.experiments.run_timing(processes=<all 12>, lengths=[100],
  n_repeats=100)` (the CLI times only ARMA(1,1) and ETS(A,A,A) m=12). Rows for
  aeon on non-seasonal ETS and for the statsmodels grids come from the
  corrected adapters (2026-10-22; see `equivalence/README.md`).
- `ets_diagnostics.csv.gz`: for the same ETS series, every candidate model
  aeon's AutoETS fits, with the loss evaluations, iterations and iteration-limit
  hits of each of its three Nelder-Mead starts
  (`scripts/ets_n100_diag.py`). Run before the adapter fix: on the
  non-seasonal processes `total_s` is AutoETS(seasonal_period=None), which
  also fitted seasonal candidates whenever it estimated a period above 1,
  while the per-candidate rows cover only the six non-seasonal candidates.
  That mismatch (time unrelated to evaluations) is how the adapter error was
  found.
- `auto/run_info.json` records the first (two-process) CLI call that started
  this directory; the other processes were added with `run_timing` as above.
