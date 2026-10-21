#!/bin/bash
# Run aeon AutoETS and AutoARIMA on all 100,000 M4 series, then score them
# against the published M4 forecasts (equivalence/m4.py).
#
#   equivalence/run_m4.sh AEON_CHECKOUT [OUT_ROOT] [N_JOBS]
#
# AEON_CHECKOUT is a clean git checkout of aeon (a worktree, not the checkout you
# develop in). Results go to OUT_ROOT/aeon-<short commit>/aeon/<forecaster>/
# (default OUT_ROOT /local/ajb2u23/Results/Forecasting), the summary to
# equivalence/results/aeon-<short commit>/m4/. AutoARIMA is non-seasonal, so it
# is fitted to M4-style deseasonalised series (--deseasonalise), as the M4
# benchmarks were; AutoETS gets the seasonal period. About 1 CPU-hour each.
set -euo pipefail
AEON=$(realpath "${1:?usage: run_m4.sh AEON_CHECKOUT [OUT_ROOT] [N_JOBS]}")
OUT_ROOT=${2:-/local/ajb2u23/Results/Forecasting}
N_JOBS=${3:-60}
ENV=/local/ajb2u23/conda/env/aeon-fcbench
cd "$(dirname "$0")/.."
if [ -n "$(git -C "$AEON" status --porcelain)" ]; then
    echo "$AEON has uncommitted changes; benchmark a clean checkout" >&2; exit 1
fi
SHORT=$(git -C "$AEON" rev-parse --short=8 HEAD)
OUT=$OUT_ROOT/aeon-$SHORT
export PYTHONPATH=$AEON
export LD_LIBRARY_PATH=$ENV/lib
export NUMBA_CACHE_DIR=$(mktemp -d)  # fresh: numba does not see changes across files
echo "aeon $(git -C "$AEON" rev-parse HEAD) from $AEON -> $OUT"
for F in AutoETS AutoARIMA; do
    EXTRA=; [ $F = AutoARIMA ] && EXTRA=--deseasonalise
    $ENV/bin/python scripts/m_comp_experiments.py --datasets M4 --limit-per-type 0 \
        --forecasters $F --backends aeon $EXTRA --n-jobs "$N_JOBS" \
        --progress-every 10000 --output-dir "$OUT"
done
$ENV/bin/python equivalence/m4.py --results "$OUT" \
    --out equivalence/results/aeon-$SHORT/m4
