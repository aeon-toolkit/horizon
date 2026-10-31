#!/bin/bash
# SARIMA equivalence with the order given (equivalence/sarima.py): simulated
# series and M4, against R, statsforecast and statsmodels, then run time.
#
#   equivalence/run_sarima.sh AEON_CHECKOUT [N_JOBS]
#
# AEON_CHECKOUT is a clean git checkout of aeon. AEON_M4_DATASET is the Dataset
# directory of the M4-methods repository. ENV is the Python environment and
# RSCRIPT the Rscript binary; R needs the forecast package. Results go to
# equivalence/results/sarima-<short commit>/. The timing phase needs an
# otherwise idle machine.
set -euo pipefail
AEON=$(realpath "${1:?usage: run_sarima.sh AEON_CHECKOUT [N_JOBS]}")
N_JOBS=${2:-60}
ENV=${ENV:-/local/ajb2u23/conda/env/aeon-fcbench}
RSCRIPT=${RSCRIPT:-Rscript}
cd "$(dirname "$0")/.."
if [ -n "$(git -C "$AEON" status --porcelain)" ]; then
    echo "$AEON has uncommitted changes; benchmark a clean checkout" >&2; exit 1
fi
OUT=equivalence/results/sarima-$(git -C "$AEON" rev-parse --short=8 HEAD)
export PYTHONPATH=$AEON
export LD_LIBRARY_PATH=$ENV/lib
export NUMBA_CACHE_DIR=$(mktemp -d)  # fresh: numba does not see changes across files
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
echo "aeon $(git -C "$AEON" rev-parse HEAD) from $AEON -> $OUT"
$ENV/bin/python equivalence/sarima.py series --out "$OUT"
$ENV/bin/python equivalence/sarima.py python --out "$OUT" --jobs "$N_JOBS"
"$RSCRIPT" equivalence/sarima_refs.R "$OUT/series.csv" "$OUT/r.csv" "$N_JOBS"
$ENV/bin/python equivalence/sarima.py analyse --out "$OUT" | tee "$OUT/summary.txt"
$ENV/bin/python equivalence/sarima.py timing --out "$OUT"
