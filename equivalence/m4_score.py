"""Score M4 result files against the official M4 benchmarks.

Reads the test-split result files written by ``m_comp_experiments.py`` and the
official point forecasts (``submission-<method>.csv`` from the M4-methods
repository), checks that our test values match the official test sets, and
prints sMAPE, MASE and OWA by frequency as in the M4 evaluation script
(``Benchmarks and Evaluation.R``): sMAPE and MASE averaged over all forecast
points, OWA relative to Naive2.

    python equivalence/m4_score.py NAME=path/to/M4_X_Test.csv ... \
        --official ETS Naive2 --official-dir /local/ajb2u23/Data/Forecasting/m4/official
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FREQS = ["yearly", "quarterly", "monthly", "weekly", "daily", "hourly"]
PREFIX = dict(zip(FREQS, "YQMWDH"))
# first global index of each frequency, as in m_comp_experiments.M4_FILES
START = dict(zip(FREQS, [1, 23001, 47001, 95001, 95360, 99587]))
PERIOD = dict(zip(FREQS, [1, 4, 12, 1, 1, 24]))
M4_HOME = Path("/local/ajb2u23/Data/Forecasting/m4")


def read_results(path: Path) -> pd.DataFrame:
    """Read a result file (header lines, then ``@results``, then CSV)."""
    with path.open() as fh:
        skip = next(i for i, line in enumerate(fh) if line.strip() == "@results") + 1
    df = pd.read_csv(path, skiprows=skip)
    index = df.SeriesID.str.split(":").str[1].astype(int)
    df["id"] = [
        PREFIX[f] + str(i - START[f] + 1) for f, i in zip(df.frequency, index)
    ]
    return df.set_index("id")


def read_tsf_train(freq: str) -> dict[str, np.ndarray]:
    """Return the training part of every series of one frequency."""
    path = M4_HOME / f"m4_{freq}_dataset.tsf"
    out, h, in_data = {}, None, False
    with path.open(encoding="cp1252") as fh:
        for line in fh:
            line = line.strip()
            if not in_data:
                if line.lower().startswith("@horizon"):
                    h = int(line.split()[1])
                in_data = line.lower() == "@data"
                continue
            if line:
                values = np.array(line.split(":", 2)[2].split(","), dtype=float)
                out[PREFIX[freq] + str(len(out) + 1)] = values[:-h]
    return out


def read_wide(path: Path) -> pd.DataFrame:
    """Read an official wide file (id, then one column per horizon step)."""
    return pd.read_csv(path, index_col=0)


def scores(pred: np.ndarray, actual: np.ndarray, scale: np.ndarray):
    """Per-series mean sMAPE and MASE over the horizon (rows are series)."""
    smape = 200 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred))
    mase = np.abs(actual - pred) / scale[:, None]
    return np.nanmean(smape, axis=1), np.nanmean(mase, axis=1)


def score_runs(
    runs: dict[str, pd.DataFrame],
    official: list[str],
    official_dir: Path = M4_HOME / "official",
) -> pd.DataFrame:
    """Per-series sMAPE and MASE of each run and each official submission.

    ``runs`` maps a name to a result table from :func:`read_results`; official
    submissions are named ``"M4 <name>"``. Raises if a run's test values differ
    from the official test sets.
    """
    official_fc = {
        name: read_wide(official_dir / f"submission-{name}.csv") for name in official
    }
    per_series = []
    for freq in FREQS:
        train = read_tsf_train(freq)
        ids = list(train)
        test = read_wide(official_dir / f"{freq.capitalize()}-test.csv")
        h = test.shape[1]
        actual = test.loc[ids].to_numpy(float)
        m = PERIOD[freq]
        scale = np.array([np.mean(np.abs(y[m:] - y[:-m])) for y in train.values()])

        forecasts = {}
        for name, df in runs.items():
            sub = df.loc[ids]
            ours = sub[[f"actual_{i + 1}" for i in range(h)]].to_numpy(float)
            if not np.allclose(ours, actual, rtol=1e-9, atol=1e-6):
                bad = np.flatnonzero(~np.isclose(ours, actual).all(axis=1))
                raise ValueError(f"{name} {freq}: test values differ, e.g. {ids[bad[0]]}")
            forecasts[name] = sub[[f"pred_{i + 1}" for i in range(h)]].to_numpy(float)
        for name, df in official_fc.items():
            forecasts[f"M4 {name}"] = df.loc[ids].iloc[:, :h].to_numpy(float)

        for name, pred in forecasts.items():
            s, q = scores(pred, actual, scale)
            per_series.append(
                pd.DataFrame({"id": ids, "frequency": freq, "method": name,
                              "smape": s, "mase": q})
            )
    return pd.concat(per_series, ignore_index=True)


def owa_table(per_series: pd.DataFrame) -> pd.DataFrame:
    """Mean sMAPE and MASE by frequency and method, and OWA against M4 Naive2."""
    rows = []
    for freq in FREQS + ["all"]:
        df = per_series if freq == "all" else per_series[per_series.frequency == freq]
        out = df.groupby("method")[["smape", "mase"]].mean()
        if "M4 Naive2" in out.index:
            base = out.loc["M4 Naive2"]
            out["owa"] = (out.smape / base.smape + out.mase / base.mase) / 2
        rows.append(out.assign(frequency=freq).reset_index())
    return pd.concat(rows).set_index(["frequency", "method"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="NAME=path to a *_Test.csv file")
    parser.add_argument("--official", nargs="*", default=["ETS", "Naive2"])
    parser.add_argument(
        "--official-dir", type=Path, default=M4_HOME / "official"
    )
    parser.add_argument("--per-series", type=Path, help="write per-series scores")
    args = parser.parse_args()

    runs = {}
    for token in args.runs:
        name, path = token.split("=", 1)
        runs[name] = read_results(Path(path))
    per_series = score_runs(runs, args.official, args.official_dir)
    if args.per_series:
        per_series.to_csv(args.per_series, index=False)
    table = owa_table(per_series)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table.unstack("method").swaplevel(axis=1).sort_index(axis=1).to_string())


if __name__ == "__main__":
    main()
