#!/bin/bash
# Section 5.3, LSSH-99: every forecaster in paper/lssh99_forecasters.txt, 30
# rolling one-step refits per series, aeon 8585c7bf. Raw rows go to
# $RESULTS/aaltd-8585c7bf/{cpu,dl-gpu<k>}/<forecaster>.csv (DrCIF on values:
# $RESULTS/aaltd-8585c7bf-levels/cpu); then
# paper/make_tables.sh scores them. Runs resume and retry failed fits on a rerun.
#
#   paper/run_lssh99.sh cpu      # everything but the deep learners
#   paper/run_lssh99.sh gpu K    # quarter K (0-3) of the deep learner fits on GPU K
set -euo pipefail
cd "$(dirname "$0")/.."
source paper/env.sh
check_worktree
OUT=$RESULTS/aaltd-8585c7bf
DL="TCNDiff DeepARDiff NBeatsDiff"
run() {  # run N_JOBS OUT_SUBDIR FORECASTER... [extra args]
    local nj=$1 sub=$2; shift 2
    $PY scripts/aaltd_experiments.py --deseasonalise --n-jobs $nj --out $OUT/$sub --forecasters "$@"
}
if [ "${1:-}" = cpu ]; then
    export CUDA_VISIBLE_DEVICES=""
    for F in $(grep -v -w -e TCNDiff -e DeepARDiff -e NBeatsDiff -e DrCIFLevels paper/lssh99_forecasters.txt); do
        run 32 cpu $F   # the regressors take 2-4 GB per worker on 10,000-point series
    done
    # DrCIF on values is registered by its own wrapper (scripts/levels_regressors.py)
    $PY scripts/levels_regressors.py --deseasonalise --n-jobs 32 \
        --out $RESULTS/aaltd-8585c7bf-levels/cpu --forecasters DrCIFLevels
elif [ "${1:-}" = gpu ]; then
    # TensorFlow fits run in rounds of fresh worker pools (host memory leak);
    # a second pass retries fits that ran out of GPU memory
    for pass in 1 2; do
        CUDA_VISIBLE_DEVICES=$2 run 6 dl-gpu$2 $DL --shard $2/4
    done
else
    echo "usage: run_lssh99.sh cpu | gpu K" >&2; exit 2
fi
