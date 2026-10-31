#!/bin/bash
# Rebuild every Section 5 table, figure and league page from the stored results
# (minutes; no forecasting). Steps marked "raw" read the per-series output under
# $RESULTS and the M4/LSSH-99 data under /local; the others read only files
# committed in this repository.
set -euo pipefail
cd "$(dirname "$0")/.."
source paper/env.sh
R=results/aeon-8585c7bf
E=equivalence/results/aeon-8585c7bf

# 5.1 simulated series
$PY equivalence/simulated.py --run $R --out $E/simulated
# 5.1 and 5.2: aeon AutoETS/AutoARIMA/SCUM against the published M4 forecasts (raw)
# and Theta/DOTM/AutoCES against statsforecast 2.1.1 and smooth 1.2.0 (tab:other_stats)
SF=$RESULTS/statsforecast-2.1.1/statsforecast
$PY equivalence/m4.py --results $RESULTS/aeon-8585c7bf --out $E/m4 --reference \
    "statsforecast Theta=$SF/Theta/M4_Theta_Test.csv" \
    "statsforecast DOTM=$SF/DOTM/M4_DOTM_Test.csv" \
    "statsforecast AutoCES=$SF/AutoCES/M4_AutoCES_Test.csv" \
    "smooth AutoCES=$RESULTS/smooth-1.2.0/smooth/AutoCES/M4_AutoCES_Test.csv"
# tab:other_stats inputs, the timing figures and the equivalence tables
$PY equivalence/paper.py --simulated $E/simulated --significance $R \
    --sweep $R/sweep-auto-r20 --m4 $E/m4 --sim-other $R/sim-other \
    --timing-other $R/timing-other --out equivalence/paper

# 5.2 M4 league and tab:m4 (raw)
$PY -W ignore scripts/m4_league.py --out $R/m4 --paper paper

# 5.3 LSSH-99: score the fixed forecaster set (raw), then the league and tab:lssh
(cd scripts && $PY aaltd_score.py --results $RESULTS/aaltd-8585c7bf \
    --forecasters ../paper/lssh99_forecasters.txt --out ../$R/aaltd)
$PY scripts/lssh99_league.py $R/aaltd --paper paper
