# Handover: training windows for automatic ARIMA/ETS on long series

Written 2026-10-29 for a new session that takes over this line of work. The
horizon session keeps the aeon paper, the LSSH-99/M4 benchmark runs and the
background GPU jobs; this note is everything needed to carry on the window work
without them.

**Moved (2026-10-29, commit ccd3290):** `papers/windowed-ets-arima/` is no longer in
horizon. The paper is its own Overleaf project (`~/Papers/arxiv-arima`); its tools and
the window work are in `~/Code/aeon-forecasting/local/windows`. Paths below under
`papers/` refer to the copy at commit 21b2daa.

## 1. The question

Statistical forecasters are fitted to the whole history. On long series (LSSH-99:
99 series, up to 10,000 points) that assumes one model with fixed parameters
throughout. Two questions (user, 2026-10-27):

1. Is it generally better **not** to fit all the data, i.e. fit to the last `w`
   points?
2. Can `w` be chosen, or windows combined, **automatically** from data before the
   forecast origin?

Scope agreed for the write-up: experiments (1) and an algorithm (2) up to a
`Windower` wrapper. The SCUM-like windowed ensemble is **parked**.

## 2. Setup (do not skip)

- aeon code: the worktree `/home/ajb2u23/Code/aeon-bench-next` at 8585c7bf (= the
  aeon 1.7 forecasting code). Never edit files there. Results are named
  `*-8585c7bf`.
- `source paper/env.sh` (from `~/Code/horizon`) sets `$PY`, `$RESULTS`
  (`/local/ajb2u23/Results/Forecasting`), PYTHONPATH, LD_LIBRARY_PATH and a fresh
  NUMBA_CACHE_DIR; `check_worktree` verifies the commit.
- Do **not** edit `scripts/m_comp_experiments.py` or `scripts/aaltd_experiments.py`:
  the GPU jobs still running (InceptionTime, DisjointCNN then ResNet) re-import
  them. Add forecasters through wrapper scripts that monkeypatch `mc`
  (examples: `scripts/levels_regressors.py`, `scripts/drcif4.py`); spawned workers
  re-import `__main__`, so patch at module top level.
- **Stopping a run:** kill the process group, not just the parent. Pool workers
  otherwise survive as orphans (PPID 1) and keep burning CPU; this happened twice.
  Check afterwards with `ps -u ajb2u23 -o pid,ppid,cmd | awk '$2==1'`.
- Waiters using `pgrep -f` match their own command line: use a bracket pattern
  (`"[l]ssh99_window"`).
- Big outputs go under `/local/ajb2u23`. Commit locally, never push without
  asking. Say "other toolkits", never "competitors".

## 3. What was run

All on aeon 8585c7bf: AutoARIMA, AutoETS, ARIMA (defaults = ARMA(1,1)) and ETS
(defaults = SES). AutoARIMA, ARIMA and ETS are M4-deseasonalised; AutoETS is
seasonal with the Monash period. These are the paper's settings.

**LSSH-99** (`scripts/lssh99_window.py`, scorer `scripts/lssh99_window_score.py`):
- Last 60 origins, refitted at each, one step ahead. Origins 0-29 are validation;
  30-59 are the paper's 30 test origins.
- At the whole series, the test forecasts are identical to the paper's runs.
- Windows: 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500 and the
  whole series.
- Raw rows in `$RESULTS/lssh99-window-8585c7bf/`.
- Scores in `results/aeon-8585c7bf/lssh99-window/`.

**M4, the 5,070 series with at least 500 training points**
(`scripts/m4_window.py`, `scripts/m4_window_score.py`):
- Standard M4 protocol, OWA against the official Naive2.
- Raw rows in `$RESULTS/m4-window-8585c7bf{,-min300}/`.
- Scores in `results/aeon-8585c7bf/m4-window-min300/`.

**Reruns with in-sample fit statistics** (`scratch/run_window_fitstats.sh`):
- Output in `$RESULTS/{lssh99,m4}-window-fitstats-8585c7bf/`.
- The forecasts are identical to the first runs.

**Forecast store** (`scripts/window_store.py build|evaluate`):
- Every window's forecast and fit statistics in one place, at
  `$RESULTS/window-store-8585c7bf`.
- Any combination scheme can be scored from it without refitting.
- `window_combine.py` holds the schemes.

**Rebuild everything from stored forecasts in minutes:** the commands are in
`papers/windowed-ets-arima/README.md`.

## 4. Results

### LSSH-99

Relative MAE against Naive on the 30 test origins, geometric mean over 98 series:

| | whole series | best fixed w | rolling selection | mean over windows | oracle |
|---|---|---|---|---|---|
| AutoARIMA | 0.918 | 0.846 (w=500) | 0.800 | 0.834 (66/32 better/worse, Holm p = 0.002) | 0.745 |
| AutoETS | 0.927 | 0.859 (w=1000) | 0.808 | 0.818 (68/30, Holm p < 0.001) | 0.734 |
| ARIMA (1,1) | 0.932 | 0.862 (w=300) | 0.823 | 0.881 | 0.787 |
| ETS (SES) | 1.023 | 1.005 | 1.006 | 1.014 | 0.990 |

- **Fixed windows** of 300-1000 points beat the whole series by 6-8%. No single
  window is significant after Holm correction.
- **Combining beats choosing.**
  - The median over windows gains 11-13%.
  - Weighting windows by recent out-of-sample (holdout) error gains 12-15%, with
    about 70% of series better.
  - Best schemes for AutoETS: inverse holdout MSE 0.789 and inverse in-sample MSE
    0.790. The median gives 0.804; the whole series gives 0.927.
- **Where the gain comes from:** almost all of it is on sub-daily series.
  - Hourly (21 series): AutoARIMA goes from 0.774 to 0.549 with rolling selection.
    The likely reason is that the daily profile is estimated locally.
  - Minutely wind (5 series): the whole series is worse than Naive and windows give
    about 0.63. This is surprising and needs inspecting.
  - Daily, weekly and monthly series gain little.
- **Cost:** a 500-1000 point window is 2-4x cheaper per fit than the whole series.
  As run, though, combining fits all 13 windows.
- **Not a fair league comparison:** windowed AutoARIMA and AutoETS would be level
  with DrCIF (Δy) at 0.781, but the other forecasters were not given windows.

### M4 long series

OWA (`papers/windowed-ets-arima/tables/m4_main.tex`):

| | Whole | Median | Median of better half | Inv. holdout MSE, w≥300 | Inv. in-sample MSE |
|---|---|---|---|---|---|
| AutoARIMA | 0.920 | 0.905 | 0.898 | **0.894** | 0.912 |
| AutoETS | 0.945 | 0.926 | 0.924 | **0.922** | 0.943 |

- The gains are smaller than on LSSH-99.
- They are concentrated on hourly, monthly and weekly series; daily series don't
  change.
- **The plain mean is bad on hourly** (AutoARIMA 1.221, AutoETS 1.697) because some
  windows blow up. Use the median, or a minimum window. A minimum of w≥300 matters
  on M4.
- **In-sample weights don't transfer to M4.** Neither do Akaike-style weights or
  picking the best-fitting window. This is consistent with the forecast combination
  puzzle.
- **Robust choice:** the median, and the median of the better half, are within 2%
  of the best scheme on both benchmarks.

### Caveats

- **Short hourly windows:** below 300 points hourly series are not deseasonalised,
  because the M4 test only looks at lags up to 10·log10 n. Results for windows under
  300 on hourly series therefore mix two effects.
- **AutoETS model choice:** a short window can change which model AutoETS selects,
  not just its parameters. For example, solar is AAA at 1000 points and AAN at 500.
  The selected model is recorded per row but has not been analysed.
- **Noisy selection:** choosing a window from 30 one-step errors is noisy.
- **Timings:** these were measured with other jobs on the machine, so only relative
  times are meaningful.

## 5. Write-ups

**`papers/windowed-ets-arima/`: first LaTeX draft** (commit 21b2daa, the user's to
take up):
- `main.tex` and `main.pdf`.
- `refs.bib`: 30 entries, each verified against Crossref by DOI. See
  `refs_verification.md` and `check_refs.py`.
- `make_tables.py` generates `tables/*.tex`; don't edit the tables by hand.
- `evaluate_schemes.py` scores the combination schemes.
- Open `\todo{}`s: authors, URL, implementing the `Windower`, simulations, and
  inspecting the wind result.

**`reports/windowed-arima-ets.md`: the earlier technical report:**
- motivation;
- related work (the literature search of 2026-10-27);
- Part 1 experiments;
- Part 2, the `Windower` design (not built);
- open questions.

**`notes/lssh99-window-experiment.md`: experiment notes and the full plan.**

**Related work found** (all in `refs.bib`):
- selecting the estimation window: Pesaran and Timmermann (2007); Inoue, Jin and
  Rossi (2017);
- combining across windows: Pesaran and Pick (2011);
- down-weighting older data;
- the forecast combination puzzle;
- robust means;
- Akaike weights;
- bagging.

## 6. Not done / next steps (none started; check with the user first)

1. **Build the `Windower`.** Design in the report, §4.1.
   - It is a wrapper forecaster that fits the base forecaster on a geometric grid of
     windows (a minimum of about 300) and combines the forecasts, by median or by
     inverse holdout MSE.
   - `EnsembleForecaster` over a `TrainWindowForecaster(f, w)` would give mean or
     median combination with no new ensemble class.
   - Rolling selection needs stored past errors, so it would be stateful.
2. **Simulated controls.** Report §3.3; also a paper `\todo`.
3. **Inspect the minutely wind result** and the AutoETS model switching.
4. **Plan part A: choose the window without full refits.**
   - Fit once and roll the state forward. aeon `ARIMA._predict(y)` can do this
     already; `ETS._predict` ignores `y`.
   - Use one filter pass with suffix sums to score every window for fixed ETS
     parameters.
   - Update parameters incrementally: RLS for AR terms, Hannan-Rissanen for MA
     terms, warm starts for ETS.
   - Try a forgetting factor instead of a hard window. Note that TVP was poor on
     LSSH-99.
5. **Plan part B:** try the cheap schemes first on simple fixed models: SES by an
   alpha grid, ARMA(1,1), and AR(1) and AR(2) with exact sliding RLS.
6. **Plan part C: long AR(k) as a substitute for ARMA**, fitted in closed form, with
   the order chosen by Levinson-Durbin and AIC.
   - aeon's `ARIMA(p,0,0)` fits a pure AR by Nelder-Mead on CSS. That is slow and
     probably badly converged. OLS is exact for CSS when q = 0, so a closed-form
     path for that case is a candidate aeon change.
   - **ARForecaster:** the user may build it in a separate conversation. Its pieces
     and pitfalls are in the notes' "Plan and ideas" section, part C. In
     particular, `AutoCorrelationSeriesTransformer` is not the standard ACF, so
     don't use it for Yule-Walker.
7. **Parked:** the SCUM-like ensemble of windowed members. The design sketch and an
   offline test with AutoCES and DOTM are in the notes, part 0.

## 7. Other horizon state (for context only; owned by the horizon session)

- **aeon paper:** results fixed. Table 4 last changed 2026-10-29, Overleaf at
  1515d07. Rebuild horizon's copies with `paper/make_tables.sh`.
- **GPU jobs:** the aaltd DL (Δy) jobs are still running: InceptionTime on GPU 0,
  and DisjointCNN then ResNet on GPU 2. They will be reviewed when they finish.
- **DrCIF deseasonalising study:** done, `notes/drcif-deseasonalising.md`. The
  scripts and results for it are not yet committed.
