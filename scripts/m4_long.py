"""``m_comp_experiments.py`` on the long M4 series only.

Same arguments as ``m_comp_experiments.py`` plus ``--min-length N`` (default
500): M4 series whose training part has fewer than N points are skipped. The
default is the subset of the window experiments (5,070 series).

    PYTHONPATH=<aeon worktree> python scripts/m4_long.py --min-length 500 \\
        --datasets M4 --limit-per-type 0 --backends aeon --forecasters DrCIFDeltaY \\
        --types monthly quarterly hourly --n-jobs 40 --output-dir <dir>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import m_comp_experiments  # noqa: E402


def _min_length() -> int:
    if "--min-length" not in sys.argv:
        return 500
    i = sys.argv.index("--min-length")
    value = int(sys.argv[i + 1])
    del sys.argv[i : i + 2]
    return value


MIN_LENGTH = _min_length()
_select = m_comp_experiments._select_m4_series


def _select_long(args):
    return [s for s in _select(args) if s.y_train.size >= MIN_LENGTH]


m_comp_experiments._select_m4_series = _select_long

if __name__ == "__main__":
    raise SystemExit(m_comp_experiments.main())
