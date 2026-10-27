#!/bin/bash
# Section 5.1, Forecasting Simulated Series: every run behind Table tab:other_stats
# and Figures fig:timing / fig:timing-more, aeon 8585c7bf. Results go to
# results/aeon-8585c7bf*/ and equivalence/results/aeon-8585c7bf/; then
# paper/make_tables.sh builds the tables and figures. The timing steps need an
# otherwise idle machine; about a day in all.
set -euo pipefail
cd "$(dirname "$0")/.."
source paper/env.sh
check_worktree
R=results/aeon-8585c7bf

# accuracy and timing at n = 100 and 500, both modes, all toolkits except smooth
for M in auto fixed; do $PY -m horizon run --mode $M --out $R/$M & done; wait
# smooth 1.2.0 separately (slow), then merged into $R/{auto,fixed} by hand
for M in auto fixed; do
    $PY -m horizon run --mode $M --toolkits smooth --out results/aeon-8585c7bf-smooth/$M & done; wait
$PY equivalence/simulated.py --run $R --out equivalence/results/aeon-8585c7bf/simulated

# fig:timing, fig:timing-more: AutoARIMA / AutoETS time against length, 20 series per length
$PY -m horizon sweep --mode auto --only timing --repeats 20 --out $R/sweep-auto-r20

# tab:other_stats: Theta, DOTM, AutoCES against statsforecast 2.1.1 and smooth 1.2.0
$PY scripts/sim_other_stats.py --out $R/sim-other --n-jobs 30          # accuracy, n = 100, 500
PYTHONPATH=$AEON:src $PY scripts/timing_other_stats.py --out $R/timing-other --repeats 100 --lengths 100
for n in 1000 10000; do
    PYTHONPATH=$AEON:src $PY scripts/timing_other_stats.py --out $R/timing-other \
        --all-processes --lengths $n --repeats 20
done
# their M4 accuracy: the reference implementations on all of M4 (aeon's runs: run_m4.sh)
$PY scripts/m_comp_experiments.py --datasets M4 --limit-per-type 0 --backends statsforecast \
    --forecasters Theta DOTM AutoCES --n-jobs 30 --output-dir $RESULTS/statsforecast-2.1.1
$PY scripts/m_comp_experiments.py --datasets M4 --limit-per-type 0 --backends smooth \
    --forecasters AutoCES --n-jobs 30 --output-dir $RESULTS/smooth-1.2.0

# the statsforecast main comparison in the text (PR #1232): needs a venv with
# statsforecast installed from GitHub main, here /local/ajb2u23/venvs/sf-main
SF_MAIN=${SF_MAIN:-/local/ajb2u23/venvs/sf-main/bin/python}
E="statsforecast:Theta statsforecast:DOTM statsforecast:AutoCES aeon:Theta aeon:DOTM aeon:AutoCES"
for v in sf-2.1.1:$PY sf-main:$SF_MAIN; do
    for n in 1000 10000; do
        ${v#*:} scripts/timing_other_stats.py --all-processes --lengths $n --repeats 20 \
            --entries $E --out $R/timing-sf-main/${v%%:*}-n$n
    done
done
