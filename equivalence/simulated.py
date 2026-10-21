"""Equivalence and speed of aeon against the other toolkits on simulated series.

Summarises a horizon benchmark run (``run_simulated.sh``, i.e.
``python -m horizon run`` in both modes) of one aeon version:

- difference tests: counts of the paired Wilcoxon signed-rank verdicts (Holm
  corrected) in each mode's ``significance.csv``, pooled over processes and per
  process;
- equivalence tests: TOST on per-series log MASE ratios with a 5% margin
  (``horizon.sweep.equivalence``), pooled and per process;
- run time: aeon's median fit-and-forecast time and each competitor's time
  relative to it, from ``timing_summary.csv``.

aeon variants (e.g. ``aeon (auto sp)``) are reported separately from the
competitors.

    python equivalence/simulated.py --run results/aeon-8585c7bf \\
        --out equivalence/results/aeon-8585c7bf/simulated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from horizon.analyse import _find
from horizon.sweep import equivalence

MODES = ["auto", "fixed"]


def _scope(df: pd.DataFrame) -> pd.Series:
    return df["scope"].eq("all").map({True: "pooled", False: "per process"})


def difference_verdicts(sig: pd.DataFrame) -> pd.DataFrame:
    """Signed-rank verdict counts, competitor better / no difference / aeon better."""
    sig = sig[~sig["competitor"].str.startswith("aeon")]
    verdict = sig["verdict"].where(
        sig["verdict"].isin(["no significant difference", "aeon better"]),
        "competitor better",
    )
    return (sig.assign(scope=_scope(sig), verdict=verdict)
            .groupby(["scope", "verdict"]).size().unstack(fill_value=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True,
                        help="run directory holding auto/ and fixed/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    info, diff_rows, eq_rows, timing_rows = {}, [], [], []
    for mode in MODES:
        run = args.run / mode
        if not run.exists():
            continue
        info[mode] = json.loads((run / "run_info.json").read_text())["aeon"]

        sig = pd.read_csv(run / "significance.csv")
        diff_rows.append(difference_verdicts(sig).assign(mode=mode))

        acc = pd.read_csv(_find(run, "accuracy.csv"))
        eq = equivalence(acc, margin=args.margin, alpha=args.alpha).assign(mode=mode)
        eq_rows.append(eq)

        tim = pd.read_csv(run / "timing_summary.csv")
        aeon = tim[tim["toolkit"] == "aeon"].set_index(["family", "process", "n"])
        wide = tim.pivot_table(index=["family", "process", "n"], columns="toolkit",
                               values="aeon_speedup")
        wide = wide.drop(columns=[c for c in wide.columns if c.startswith("aeon")])
        wide.insert(0, "aeon_median_s", aeon["median_s"])
        timing_rows.append(wide.reset_index().assign(mode=mode))

    if not info:
        raise SystemExit(f"no auto/ or fixed/ run under {args.run}")
    diff = pd.concat(diff_rows).reset_index()
    eq = pd.concat(eq_rows, ignore_index=True)
    timing = pd.concat(timing_rows, ignore_index=True)
    diff.to_csv(args.out / "difference_verdicts.csv", index=False)
    eq.to_csv(args.out / "tost.csv", index=False)
    timing.to_csv(args.out / "timing.csv", index=False)
    (args.out / "run_info.json").write_text(json.dumps(
        {"run": str(args.run), "margin": args.margin, "alpha": args.alpha,
         "aeon": info}, indent=2))

    comps = eq[~eq["competitor"].str.startswith("aeon")]
    eq_counts = (comps.assign(scope=_scope(comps))
                 .groupby(["mode", "scope", "verdict"]).size().unstack(fill_value=0))
    eq_counts.to_csv(args.out / "tost_verdicts.csv")
    with pd.option_context("display.width", 200,
                           "display.float_format", "{:.3g}".format):
        print("signed-rank verdicts against competitors\n", diff.to_string(index=False))
        print(f"\nTOST verdicts (margin {args.margin:.0%})\n", eq_counts.to_string())
        print("\ncompetitor time / aeon time\n", timing.to_string(index=False))


if __name__ == "__main__":
    main()
