"""aaltd rolling-origin runs with no seasonal treatment at all.

Wraps ``aaltd_experiments.py`` (same arguments; do not pass ``--deseasonalise``)
and adds ``SCUMNoSeason``: SCUM with seasonal period 1, so none of its members
models seasonality. Other forecasters run as named, so without
``--deseasonalise`` the ML/DL forecasters see the raw series (their period is
still used to size windows, e.g. TCN).

    PYTHONPATH=<aeon worktree> python scripts/aaltd_noseason.py \\
        --forecasters SCUMNoSeason DrCIFDeltaY --n-jobs 16 --out <dir>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import aaltd_experiments  # noqa: E402

# top level, not under __main__, so spawned workers (which re-import this
# script) see the alias too
aaltd_experiments.ALIASES["SCUMNoSeason"] = ("aeon", "SCUM", False)

if __name__ == "__main__":
    if "--deseasonalise" in sys.argv:
        sys.exit("aaltd_noseason.py runs without --deseasonalise")
    aaltd_experiments.main()
