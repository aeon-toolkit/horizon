#!/bin/bash
# Section 5.2, M4: every aeon forecaster on all 100,000 M4 series, aeon 8585c7bf.
# Per-series forecasts go to $RESULTS/aeon-8585c7bf{,-dl}/aeon/<forecaster>/; then
# paper/make_tables.sh scores them. CPU runs take about 1-70 CPU-hours each (TVP
# the longest); each deep learner several GPU-days. Runs resume (--resume).
#
#   paper/run_m4.sh cpu          # statistical and machine learning forecasters
#   paper/run_m4.sh gpu GPU F    # one deep learner F (TCNDiff, DeepARDiff, NBeatsDiff)
set -euo pipefail
cd "$(dirname "$0")/.."
source paper/env.sh
check_worktree
m4() {  # m4 OUT_DIR N_JOBS [extra args] -- forecaster
    $PY scripts/m_comp_experiments.py --datasets M4 --limit-per-type 0 --backends aeon \
        --resume --progress-every 10000 --output-dir "$@"
}
if [ "${1:-}" = cpu ]; then
    export CUDA_VISIBLE_DEVICES=""
    # seasonal models get the M4 period and the raw series (CES: raw, as in the paper)
    for F in SCUM DOTM AutoCES AutoETS CES; do
        m4 $RESULTS/aeon-8585c7bf --n-jobs 60 --forecasters $F
    done
    # non-seasonal and machine learning models: M4-style deseasonalised
    for F in Theta AutoARIMA ARIMA ETS NaiveForecaster TVP TAR AutoTAR \
             RandF DrCIFDeltaY SETAR SETARTree SETARForest; do
        m4 $RESULTS/aeon-8585c7bf --n-jobs 60 --deseasonalise --forecasters $F
    done
elif [ "${1:-}" = gpu ]; then
    # deep learners through StandardisedForecaster (differenced, last 1000 points);
    # ~2-6 GB host memory per worker
    CUDA_VISIBLE_DEVICES=$2 m4 $RESULTS/aeon-8585c7bf-dl --n-jobs 14 --deseasonalise --forecasters $3
else
    echo "usage: run_m4.sh cpu | gpu GPU FORECASTER" >&2; exit 2
fi
