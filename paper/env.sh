# Common environment for the Section 5 scripts: aeon 8585c7bf from a clean
# worktree, the aeon-fcbench conda env, single-threaded numeric libraries.
# Source it from the repository root: `source paper/env.sh`.
AEON=${AEON:-/home/ajb2u23/Code/aeon-bench-next}   # worktree checked out at $COMMIT
COMMIT=8585c7bf9796a52b50e90f472c7c8718248eceea
ENV=${ENV:-/local/ajb2u23/conda/env/aeon-fcbench}
PY=$ENV/bin/python
RESULTS=${RESULTS:-/local/ajb2u23/Results/Forecasting}   # raw per-series output
SP=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export PYTHONPATH=$AEON
# pip's cuSOLVER first, or TensorFlow silently runs on the CPU
export LD_LIBRARY_PATH=$SP/nvidia/cusolver/lib:$ENV/lib
export TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true
# numba does not notice changes across files: always a fresh cache
export NUMBA_CACHE_DIR=$(mktemp -d)

check_worktree() {
    if [ "$(git -C "$AEON" rev-parse HEAD)" != "$COMMIT" ] || [ -n "$(git -C "$AEON" status --porcelain)" ]; then
        echo "ABORT: $AEON is not a clean checkout of $COMMIT" >&2; exit 1
    fi
}
