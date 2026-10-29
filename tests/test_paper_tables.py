"""The paper's LSSH-99 table and league page rebuild exactly from committed scores."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import lssh99_league  # noqa: E402


def test_lssh99_league_reproduces(tmp_path):
    """scripts/lssh99_league.py on results/aeon-8585c7bf/aaltd gives the committed files."""
    df, counts = lssh99_league.league(ROOT / "results" / "aeon-8585c7bf" / "aaltd")
    n_rank = int(df.mean_rank.notna().sum())
    lssh99_league.write_page(df, counts, n_rank, tmp_path / "lssh99_league.md")
    lssh99_league.write_tex(df, counts, tmp_path / "tab_lssh.tex")
    for name in ["lssh99_league.md", "tab_lssh.tex"]:
        assert (tmp_path / name).read_text() == (ROOT / "paper" / name).read_text(), name
    assert counts == {"sub_daily": 36, "daily_or_slower": 62, "all": 98}
