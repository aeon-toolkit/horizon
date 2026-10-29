# Unattended runs, 2026-10-23 onwards (user away ~4-5 days)

All at aeon 8585c7bf (`/home/ajb2u23/Code/aeon-bench-next`); aeon code never edited.

## Queue
1. M4, aeon stats not yet run at 8585c7bf: Naive, ETS, ARIMA, Theta, TAR, AutoTAR, TVP
   (`scratch/run_m4_stats_rest.sh`), then ML as shipped: SETAR, SETARTree,
   SETARForest, RandF, DrCIFDeltaY (`scratch/run_m4_ml.sh`). Deseasonalised as before.
2. DL re-pilot (600 M4 series) through `StandardisedForecaster` in
   `scripts/m_comp_experiments.py`: series standardised by training mean/sd,
   fitted once (aeon's `iterative_forecast` refits, so the first pilot trained
   TCN/DeepAR twice). `scratch/run_dl_scaled_pilot.sh`.
3. `scratch/run_queue_dl.sh`: DL forecasters with pilot OWA < 1.5 get full M4
   (GPU 0-2), then each GPU takes a quarter of the aaltd DL fits; GPU 3 starts its
   aaltd quarter straight away.
4. `scratch/run_queue_aaltd_cpu.sh`: after the M4 CPU runs, aaltd rolling origin
   (`scripts/aaltd_experiments.py`: 100 problems, last 30 points, refit at each,
   one-step forecast), Naive, SeasonalNaive, then the stats, then the ML
   forecasters, 32 workers. Output `/local/ajb2u23/Results/Forecasting/aaltd-8585c7bf/`.
   Score with `scripts/aaltd_score.py`.

## Pilot findings before the queue (as shipped, 600 M4 series, OWA vs Naive2)
RandF 1.01, DrCIFDeltaY 1.53, NBeats 2.5, TCN 3.3, DeepAR 26 (forecasts ~0:
no input scaling), Ridge RegressionForecaster 106, SETAR/SETARTree/SETARForest
explode (up to 1e22): SETAR default lag=10 on 19-point yearly series, SETARTree
scale=False with ill-conditioned solves. These are aeon defaults, not adapter bugs.

## Log
- 2026-10-23 08:55: queue started. M4 Naive, ETS, ARIMA, Theta, TAR, AutoTAR done; TVP running; scaled DL pilot running on GPUs 0-2.
- 2026-10-23 09:51: M4 stats finished (Naive, ETS, ARIMA, Theta, TAR, AutoTAR, TVP). Overall OWA: SCUM 0.845, DOTM 0.870, AutoCES 0.894, Theta 0.897, AutoETS 0.914, AutoARIMA 0.915, ARIMA 0.981, ETS 1.027, Naive 1.058, TVP 1.235. TAR and AutoTAR forecasts explode on some yearly/quarterly/monthly/hourly series (OWA up to 1e13): aeon behaviour, not fixed. Scored in scratch/m4_owa_latest.csv. M4 ML forecasters started.
- 2026-10-23 10:30: DeepAR scaled pilot OWA 2.86 (26 as shipped): trend leaves recent standardised values out of range. Added StandardisedForecaster(difference=True) (TCNDiff, DeepARDiff, NBeatsDiff), piloting on GPU 3; the DL gate now takes the better variant per model if its OWA < 1.5. M4 SETAR and SETARTree done.
- 2026-10-23 10:35: level-scaled DL pilot done: TCN 1.74, N-BEATS 1.82, DeepAR 2.86 (all fail the 1.5 gate). Differenced pilot now parallel: DeepARDiff GPU 3, TCNDiff GPU 0, NBeatsDiff GPU 2. M4 SETARForest done.
- 2026-10-23 12:00: DeepARDiff pilot OWA 1.10 (passes). Full M4 for a DL model is ~1400 worker-hours (~7 days on one GPU at pilot speed), so each passing DL model now runs M4 on the first 1000 series per frequency (6000), then its aaltd quarter, then the rest of M4 with --resume, into aeon-8585c7bf-dl. M4 RandF done; DrCIFDeltaY running.
- 2026-10-23 12:22: DL pilot verdict, differenced variants all pass and were chosen: DeepARDiff 1.10 (levels 2.86), TCNDiff 1.41 (1.74), NBeatsDiff 1.46 (1.82). Started M4 6000-series sample on GPUs 0-2 and aaltd DL quarter on GPU 3.
- 2026-10-23 14:40: DL runs far too slow on long series (N-BEATS ~50 M4 series in 2 h; aaltd ~5.5 min/fit), so stopped them and capped the differenced DL models' training history at the last 1000 observations (DL_MAX_TRAIN in m_comp_experiments.py; capped fit on 9919-point D4099 ~50 s). Uncapped partial output moved to *-dl-uncapped-partial. Pilots (OWA gate) were uncapped. Restarted run_queue_dl.sh.
- 2026-10-23 16:30: aaltd DL shard 3 crashed: host out of memory (OOM killer, 16:07). TensorFlow workers leaked memory (TCN M4 workers 7 GB each, growing). Fix: StandardisedForecaster now clears the Keras session after each forecast. Also found 24/100 aaltd series contain zeros, so M4 multiplicative deseasonalisation divided by zero: these are now decomposed additively (R decompose type='additive'); M4 unaffected (all positive). Dropped 180 aaltd TCNDiff rows on those series. Note: by M4's rule (lag.max = 10 log10 n) periods above ~40 (48, 52, 144, 1440) are never judged seasonal, so those aaltd series are not deseasonalised. DL queue stopped while testing; M4 DL samples keep their 400 rows each (resume).
- 2026-10-23 16:25: clear_session alone did not stop the leak (~75 MB/fit in a test), so TensorFlow workers are now recycled (max_tasks_per_child: after each 50-series chunk on M4, every 20 fits on aaltd). DL queue restarted; M4 samples resume from 400 rows each, aaltd shard 3 from 270 fits.
- 2026-10-23 17:55: aaltd DL shard 3 hung for ~1.5 h: with max_tasks_per_child the 6 workers exited after 20 fits each and the Python 3.11 pool never replaced them. aaltd_experiments.py now runs TensorFlow fits in rounds of 10 per worker, each round a fresh pool; shard 3 rerun (scratch/run_aaltd_dl_gpu3.sh), resuming at 390 fits. M4 DL recycling (one 50-series chunk per worker) is working: samples at 800 series each.
- 2026-10-23 18:30: all M4 CPU runs done (ML: SETAR, SETARTree, SETARForest, RandF, DrCIFDeltaY). Overall OWA: RandF 1.081 (hourly 0.474, best of all methods on hourly), DrCIFDeltaY 1.450 (hourly 0.643); SETAR/SETARTree/SETARForest explode as shipped (OWA up to 1e47). scratch/m4_owa_latest.csv updated. aaltd CPU queue started: Naive and SeasonalNaive done (3000 fits each), stats running.
- 2026-10-23 18:40: aaltd stats so far: Naive, SeasonalNaive, ETS, AutoETS, ARIMA, AutoARIMA, Theta, DOTM, CES, AutoCES all 3000/3000 ok. TAR: 330 fits failed (all 30 origins on 11 solar/weather/wind series), aeon TAR raises LinAlgError (singular matrix) when a regime is all zeros; aeon behaviour, left as failures.
- 2026-10-23 19:10: aaltd interim scores (99 problems; wind_farms T2 is constant, left out): relative MAE vs Naive (geometric mean) AutoARIMA 0.918, AutoCES 0.924, AutoTAR 0.925, AutoETS 0.927, ARIMA 0.932, DOTM 0.968, Theta 0.990, Naive 1, ETS 1.023, CES 1.028, SeasonalNaive 2.39. aaltd runner now records each fit's error message and retries failed fits on a rerun (89 TCNDiff errors on GPU 3 with no message recorded; TCNDiff refits sunspot fine now). Scorer fixed for zero-error Naive and constant series.
- 2026-10-23 19:05: aaltd DL shard 3 crashed reading its CSV: my edit to aaltd_experiments.py reached the running job's fresh worker rounds (spawn re-imports the script), so 60 rows had an extra error column. Truncated those rows to the 11-column header and resumed; it is retrying the 89 failed TCNDiff fits. Lesson: do not edit scripts that running jobs import.
- 2026-10-23 21:10: aaltd stats stage done (all 12 stats + Naive/SeasonalNaive; only TAR had failures). Relative MAE vs Naive: SCUM 0.843 (best, mean rank 4.8), AutoARIMA 0.918, AutoCES 0.924, AutoTAR 0.925, AutoETS 0.927, ARIMA 0.932, DOTM 0.968, Theta 0.990, ETS 1.023, CES 1.028, TVP 1.060, SeasonalNaive 2.39. ML stage started (RegressionForecaster first). Scores in results/aeon-8585c7bf/aaltd/.
- 2026-10-23 22:30: DeepARDiff M4 sample done (first 1000 per frequency; all 359 weekly and 414 hourly): OWA 1.168 (pilot 1.10), worse than Naive2; on the same 6000 series SCUM 0.855, AutoARIMA 0.924, AutoETS 0.947, Theta 0.954, RandF 1.062. Table scratch/m4_dl_sample_owa.csv. GPU 1 now on its aaltd quarter, then the rest of M4.
- 2026-10-23 22:45: GPU 3 finished its aaltd DL quarter (TCN, DeepAR, N-BEATS). It now runs the rest of M4 for DeepARDiff (scratch/run_m4_dl_rest_gpu3.sh); GPU 1's queue step for that was removed (killed its subshell; its aaltd quarter, pid 222233, continues; watch aaltd_logs/dl-gpu1.log for completion, after which GPU 1 is free).
- 2026-10-23 23:35: TCNDiff M4 sample done (the sample is 4773 series: 1000 per frequency but only 359 weekly and 414 hourly exist). OWA 1.659: yearly 2.96 drags it down; quarterly 0.86, monthly 0.96, weekly 0.86, daily 1.05, hourly 1.07. GPU 0 now on its aaltd quarter, then the rest of M4.
- 2026-10-24 00:20: NBeatsDiff M4 sample done: OWA 1.405 (yearly 1.25, quarterly 1.25, monthly 1.64, weekly 1.23, daily 1.48, hourly 1.64). All three DL M4 samples done; scratch/m4_dl_sample_owa.csv has them with SCUM/AutoARIMA/AutoETS/RandF on the same 4773 series. GPU 2 now on its aaltd quarter, then rest of M4.
- 2026-10-24 02:50: GPU 1 finished its aaltd DL quarter; it now runs the rest of M4 for TCNDiff (scratch/run_m4_dl_rest_gpu1.sh). Removed that step from GPU 0's queue subshell (its aaltd quarter, pid 3911212, continues; GPU 0 is free afterwards). GPU 3 runs rest of M4 DeepARDiff (6775/100000 rows at 02:30), GPU 2 runs aaltd quarter then rest of M4 NBeatsDiff.
- 2026-10-24 03:55: GPU 0 finished its aaltd DL quarter; it now runs the rest of M4 for NBeatsDiff (scratch/run_m4_dl_rest_gpu0.sh), removed from GPU 2's queue subshell (its aaltd quarter, pid 1365696, continues; GPU 2 is free afterwards).
- 2026-10-24 05:00: all aaltd DL quarters done. aaltd interim (99 problems, DrCIFDeltaY still running): TCNDiff 0.793 rel MAE (95 problems), DrCIFDeltaY 0.794 (82 so far), RidgeDiff 0.833, SCUM 0.843, RandF 0.845, DeepARDiff 0.896, SETAR 0.898, AutoARIMA 0.918 ... NBeatsDiff 0.989, Naive 1, SETARForest 1.89, SETARTree 2.20. 114 TCNDiff fits failed with GPU out-of-memory (traffic/solar, long windows, 6 workers per GPU); retrying on free GPU 2 with 3 workers (scratch/run_aaltd_tcn_retry.sh).
- 2026-10-24 05:15: aaltd COMPLETE (24 forecasters x 100 problems x 30 one-step refits; wind_farms T2 constant, scored on 99). Only TAR has failures (11 problems). Final, relative MAE vs Naive (geometric mean) / mean rank of 23:
  DrCIFDeltaY 0.781 / 9.5, TCNDiff 0.804 / 8.7, RidgeDiff 0.833 / 10.4, SCUM 0.843 / 8.6 (best rank), RandF 0.845, DeepARDiff 0.896, SETAR 0.898, AutoARIMA 0.918, AutoCES 0.924, AutoTAR 0.925, AutoETS 0.927, ARIMA 0.932, DOTM 0.968, NBeatsDiff 0.989, Theta 0.990, RegressionForecaster 0.998, Naive 1, ETS 1.023, CES 1.028, TVP 1.060, SETARForest 1.89, SETARTree 2.20, SeasonalNaive 2.39. TCN's 114 GPU-OOM fits all succeeded on retry with 3 workers. Scores: results/aeon-8585c7bf/aaltd/{summary,per_problem}.csv.
- 2026-10-24 05:50: M4 DL full runs at ~800-1000 series/h each (DeepARDiff 9975, TCNDiff 6775, NBeatsDiff 6375 rows), ~4-5 days to finish. CPU idle after aaltd, so restarted all three with 16 workers instead of 8 (resume).
- 2026-10-24 06:15: memory fell to 9 GB free with 16 workers each (TF workers 2-6 GB; ~190 GB total). Cut TCNDiff and NBeatsDiff to 10 workers (DeepARDiff, ~2 GB/worker, stays at 16). 157 GB free after restart.
- 2026-10-25 12:35: DeepARDiff full M4 done (100000 series, GPU 3). OWA 1.303 (yearly 1.107, quarterly 1.224, monthly 1.567, weekly 1.182, daily 1.047, hourly 0.925): between TVP (1.235) and DrCIFDeltaY (1.450), behind RandF 1.081. scratch/m4_owa_latest.csv now includes it. GPU 3 idle; TCNDiff (GPU 1) and NBeatsDiff (GPU 0) continue.
- 2026-10-25 12:50: DeepARDiff done freed ~45 GB; raised TCNDiff and NBeatsDiff from 10 to 14 workers (resume).
- 2026-10-26 02:10: TCNDiff full M4 done. OWA 2.066 (yearly 1.172, quarterly 0.975, monthly 4.706, weekly 0.862, daily 1.064, hourly 1.073). The monthly figure is driven by a few explosions: 70 series have an error over 10x their mean level, one (M44129) 245000x; 1729 series (1.7%) exceed 2x. Recursive forecasting of differences compounds errors. N-BEATS (GPU 0) continues.
- 2026-10-26 08:10: NBeatsDiff full M4 done (OWA 1.397). ALL RUNS COMPLETE. Nothing running.

## Final summary (2026-10-26)

### M4, all 100,000 series, aeon 8585c7bf, OWA vs Naive2 (results/aeon-8585c7bf/m4/owa_all_forecasters.csv)
SCUM 0.845, DOTM 0.870, AutoCES 0.894, Theta 0.897, AutoETS 0.914, AutoARIMA 0.915, CES 0.963,
ARIMA 0.981, [Naive2 1.000], ETS 1.027, Naive 1.058, RandF 1.081 (hourly 0.474, best of all),
TVP 1.235, DeepARDiff 1.303, NBeatsDiff 1.397, DrCIFDeltaY 1.450, TCNDiff 2.066 (70 exploding
monthly/yearly forecasts; quarterly 0.975, weekly 0.862). Exploding as shipped: TAR, AutoTAR,
SETAR, SETARTree, SETARForest (mean OWA meaningless; count of exploded series is the better report).

### Deep learners: what was run
As shipped they fail on M4 (pilot OWA DeepAR 26, TCN 3.3, N-BEATS 2.5: no input scaling).
Run through the local StandardisedForecaster wrapper (scripts/m_comp_experiments.py): standardised
first differences of at most the last 1000 points (DL_MAX_TRAIN), fitted once, recursive for
TCN/DeepAR, series-to-series for N-BEATS. Pilot/gate used full histories; full runs use the cap.

### aaltd (100 long series, last 30 points refitted one step ahead; results/aeon-8585c7bf/aaltd)
Rel. MAE vs Naive: DrCIFDeltaY 0.781, TCNDiff 0.804, RidgeDiff 0.833, SCUM 0.843 (best mean rank),
RandF 0.845, DeepARDiff 0.896, SETAR 0.898, AutoARIMA 0.918 ... SETARTree 2.20, SeasonalNaive 2.39.
Only TAR fails (singular matrix on 11 zero-heavy series).

### Horizon-side fixes made unattended (no aeon code changed)
StandardisedForecaster (scaling, differencing, history cap, fit once), additive deseasonalising for
non-positive series, TF worker recycling (memory leak), aaltd TF fits in fresh-pool rounds
(max_tasks_per_child hang on Python 3.11), aaltd fit errors recorded and retried, scorer fixes.
aeon issues worth reporting: DL forecasters don't scale input; iterative_forecast refits;
SETAR lag=10 default on short series; SETARTree scale=False; TAR singular on all-zero regimes;
TF models leak host memory across fits.
