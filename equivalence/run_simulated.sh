#!/bin/bash
# Run the horizon benchmark on simulated ARIMA/ETS series in both modes with the
# given aeon, then summarise equivalence and run time (equivalence/simulated.py).
#
#   equivalence/run_simulated.sh AEON_CHECKOUT
#
# AEON_CHECKOUT is a clean git checkout of aeon. Results go to
# results/aeon-<short commit>/{auto,fixed}, the summary to
# equivalence/results/aeon-<short commit>/simulated/. The two modes run in
# parallel, one process each, for about an hour; the timing phase needs an
# otherwise idle machine.
set -euo pipefail
AEON=$(realpath "${1:?usage: run_simulated.sh AEON_CHECKOUT}")
ENV=/local/ajb2u23/conda/env/aeon-fcbench
cd "$(dirname "$0")/.."
if [ -n "$(git -C "$AEON" status --porcelain)" ]; then
    echo "$AEON has uncommitted changes; benchmark a clean checkout" >&2; exit 1
fi
SHORT=$(git -C "$AEON" rev-parse --short=8 HEAD)
export PYTHONPATH=$AEON
export LD_LIBRARY_PATH=$ENV/lib
export NUMBA_CACHE_DIR=$(mktemp -d)
echo "aeon $(git -C "$AEON" rev-parse HEAD) from $AEON -> results/aeon-$SHORT"
PIDS=()
for M in auto fixed; do
    $ENV/bin/python -m horizon run --mode $M --out results/aeon-$SHORT/$M &
    PIDS+=($!)
done
for P in "${PIDS[@]}"; do wait "$P"; done  # fails if either mode failed
$ENV/bin/python equivalence/simulated.py --run results/aeon-$SHORT \
    --out equivalence/results/aeon-$SHORT/simulated
