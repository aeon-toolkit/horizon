"""Equivalence of aeon's AutoETS and AutoARIMA with the published M4 benchmarks.

Scores aeon's M4 forecasts (written by ``run_m4.sh``) and the official M4
submissions on the official test sets, then compares each aeon forecaster with
its published counterpart series by series:

- AutoETS against ``ETS`` (R ``forecast::ets``),
- AutoARIMA against ``ARIMA`` (R ``forecast::auto.arima``),
- SCUM against submission ``036`` (Petropoulos and Svetunkov's SCUM, in R),
- Theta against the M4 ``Theta`` benchmark,

and, given ``--reference NAME=path`` result files of other toolkits, aeon's
Theta, DOTM and AutoCES against those runs (e.g. statsforecast's Theta, DOTM
and AutoCES, smooth's AutoCES). Official forecasts are read from
``submission-<name>.csv``; ``--aeon NAME=path`` adds or replaces an aeon run
(e.g. Theta from an earlier commit whose Theta code is unchanged).

For each frequency and error measure (sMAPE, MASE) it reports

- OWA relative to Naive2, as in the M4 evaluation;
- a paired two-sided Wilcoxon signed-rank test of the per-series errors (ties
  dropped as in R's ``wilcox.test``), Holm corrected over the 12 tests of each
  pair, with the rank-biserial correlation as effect size (negative: aeon has
  the lower errors);
- a TOST equivalence test on per-series log error ratios, the same test and
  default 5% margin as the length sweep (``horizon.sweep.equivalence``).

    python equivalence/m4.py --results /local/.../aeon-8585c7bf \\
        --out equivalence/results/aeon-8585c7bf/m4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from horizon.sweep import _tost
from m4_score import FREQS, M4_HOME, owa_table, read_results, score_runs

# aeon forecaster -> published M4 counterpart (official submission name)
PAIRS = {"AutoETS": "ETS", "AutoARIMA": "ARIMA", "SCUM": "036", "Theta": "Theta"}
REFERENCE_LABELS = {"ETS": "R ets", "ARIMA": "R auto.arima", "036": "SCUM (M4 036)",
                    "Theta": "M4 Theta"}
# aeon forecaster tested against each --reference run named "<toolkit> <forecaster>"
RUN_PAIRS = {"Theta": ["statsforecast Theta"], "DOTM": ["statsforecast DOTM"],
             "AutoCES": ["statsforecast AutoCES", "smooth AutoCES"]}
TIE_TOL = 1e-9


def _provenance(path: Path) -> dict:
    """aeon commit and script options recorded in a result file header."""
    info = {}
    with path.open() as fh:
        for line in fh:
            if line.strip() == "@results":
                break
            m = re.match(r"# @(aeon|script_args|package_version) (.*)", line)
            if m:
                info[m.group(1)] = json.loads(m.group(2))
    return info


def signed_rank(ours: pd.Series, ref: pd.Series) -> dict:
    """Paired two-sided Wilcoxon signed-rank test of ``ours`` against ``ref``."""
    d = (ours - ref).to_numpy()
    nz = d[np.abs(d) > TIE_TOL]
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    r_pos, r_neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    res = wilcoxon(nz) if nz.size else None
    return {
        "n_series": d.size,
        "ties": d.size - nz.size,
        "aeon_better": float(np.mean(d < -TIE_TOL)),
        "ref_better": float(np.mean(d > TIE_TOL)),
        "median_diff": float(np.median(d)),
        "mean_aeon": float(ours.mean()),
        "mean_ref": float(ref.mean()),
        "W": float(res.statistic) if res else np.nan,
        "p_value": float(res.pvalue) if res else 1.0,
        "rank_biserial": float((r_pos - r_neg) / (r_pos + r_neg)) if nz.size else 0.0,
    }


def tost(ours: pd.Series, ref: pd.Series, margin: float, alpha: float) -> dict:
    """TOST on per-series log(aeon / reference) error ratios."""
    ok = (ours > 0) & (ref > 0)
    r = _tost(np.log(ours[ok] / ref[ok]).to_numpy(), margin, alpha)
    if r["p_tost"] < alpha:
        verdict = "equivalent"
    elif r["ratio_ci_high"] < 1 / (1 + margin):
        verdict = "aeon better"
    elif r["ratio_ci_low"] > 1 + margin:
        verdict = "aeon worse"
    else:
        verdict = "inconclusive"
    return {**r, "excluded_zero": int((~ok).sum()), "margin": margin,
            "verdict": verdict}


def _pairs(methods):
    """(aeon method, reference method, reference label) for every pair present."""
    out = [(a, f"M4 {r}", REFERENCE_LABELS[r]) for a, r in PAIRS.items()]
    out += [(a, r, r) for a, refs in RUN_PAIRS.items() for r in refs]
    return [p for p in out if p[0] in methods and p[1] in methods]


def compare(per_series: pd.DataFrame, margin: float, alpha: float):
    """Signed-rank and TOST tables for every aeon/reference pair present."""
    sr_rows, tost_rows = [], []
    methods = set(per_series.method)
    for aeon_name, ref, label in _pairs(methods):
        pair_rows = []
        for metric in ["smape", "mase"]:
            wide = per_series.pivot_table(
                index=["frequency", "id"], columns="method", values=metric
            )
            for freq in FREQS + ["all"]:
                w = wide if freq == "all" else wide.loc[freq]
                ours, theirs = w[aeon_name], w[ref]
                key = {"aeon": aeon_name, "reference": label,
                       "metric": metric,
                       "frequency": freq}
                if freq != "all":
                    pair_rows.append({**key, **signed_rank(ours, theirs)})
                tost_rows.append({**key, **tost(ours, theirs, margin, alpha)})
        sr = pd.DataFrame(pair_rows)
        sr["p_holm"] = multipletests(sr["p_value"], method="holm")[1]
        sr["verdict"] = np.where(
            sr["p_holm"] >= alpha, "no difference",
            np.where(sr["rank_biserial"] < 0, "aeon better", "R better"),
        )
        sr_rows.append(sr)
    return pd.concat(sr_rows, ignore_index=True), pd.DataFrame(tost_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True,
                        help="output directory of run_m4.sh (holds aeon/<F>/)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, default=M4_HOME / "official")
    parser.add_argument("--aeon", nargs="*", default=[], metavar="NAME=PATH",
                        help="aeon result files to add or use instead")
    parser.add_argument("--reference", nargs="*", default=[], metavar="NAME=PATH",
                        help="other toolkits' result files, NAME e.g. 'statsforecast DOTM'")
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    runs, provenance = {}, {}
    paths = {}
    for aeon_name in set(PAIRS) | set(RUN_PAIRS):
        path = args.results / "aeon" / aeon_name / f"M4_{aeon_name}_Test.csv"
        if path.exists():
            paths[aeon_name] = path
    for item in args.aeon + args.reference:
        name, path = item.split("=", 1)
        paths[name] = Path(path)
    for name, path in paths.items():
        runs[name] = read_results(path)
        provenance[name] = _provenance(path)
    if not runs:
        raise SystemExit(f"no aeon results under {args.results}/aeon")
    official = [PAIRS[name] for name in runs if name in PAIRS] + ["Naive2"]
    per_series = score_runs(runs, official, args.official_dir)

    args.out.mkdir(parents=True, exist_ok=True)
    per_series.to_csv(args.out / "per_series.csv.gz", index=False)
    owa = owa_table(per_series)
    owa.to_csv(args.out / "owa.csv")
    sr, eq = compare(per_series, args.margin, args.alpha)
    sr.to_csv(args.out / "signed_rank.csv", index=False)
    eq.to_csv(args.out / "tost.csv", index=False)
    (args.out / "run_info.json").write_text(json.dumps(
        {"results": str(args.results), "margin": args.margin, "alpha": args.alpha,
         "runs": provenance}, indent=2))

    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 200):
        print(owa["owa"].unstack("method").reindex(FREQS + ["all"]).to_string())
        print(sr[["aeon", "metric", "frequency", "aeon_better", "ref_better",
                  "rank_biserial", "p_holm", "verdict"]].to_string(index=False))
        print(eq[["aeon", "metric", "frequency", "ratio", "ratio_ci_low",
                  "ratio_ci_high", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
