"""Store of window forecasts for post-processing combinations, and their evaluation.

The window experiments (``lssh99_window.py``, ``m4_window.py``) fit each
forecaster to many training windows. ``build`` collects their rows into a store,
so any way of combining the window forecasts can be scored without refitting:

- ``lssh99.parquet``: one row per (forecaster, problem, origin, window): the
  one-step forecast ``pred``, ``actual``, ``naive`` (last value), in-sample fit
  scores (``insample_mse``, ``insample_mae``, ``insample_rel_mae``, ``aic``,
  ``n_resid``), ``n_train``, ``period``, ``model``, ``fit_seconds``, and
  ``validation`` (origin < 30) / test (origin >= 30).
- ``m4.parquet``: one row per (forecaster, id, split, window), split ``val``
  (fitted without the last h training points) or ``test``; ``pred`` is the
  h-step forecast (a list); same fit scores.
- ``m4_truth.parquet``: per M4 id, the test values, the holdout values, the
  published Naive2 forecast and the MASE scale.

A combination scheme is a function ``scheme(ctx) -> forecast`` of a context
built per forecast origin (see ``lssh99_contexts`` / ``m4_contexts``), e.g.::

    from window_store import Store
    s = Store("/local/ajb2u23/Results/Forecasting/window-store-8585c7bf")
    def median_long(ctx):                     # median of windows >= 300
        keep = (ctx["windows"] == 0) | (ctx["windows"] >= 300)
        return np.median(ctx["preds"][keep], axis=0)
    s.evaluate_lssh99({"median_long": median_long})
    s.evaluate_m4({"median_long": median_long})

Every scheme of ``window_combine.combine`` is available as
``window_store.BUILT_IN``.

    python scripts/window_store.py build \\
        --lssh99 /local/ajb2u23/Results/Forecasting/lssh99-window-fitstats-8585c7bf \\
        --m4 /local/ajb2u23/Results/Forecasting/m4-window-fitstats-8585c7bf \\
        --out /local/ajb2u23/Results/Forecasting/window-store-8585c7bf
    python scripts/window_store.py evaluate --store <out> --out results/aeon-8585c7bf/window-combine
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "equivalence"))

VALIDATION = 30  # LSSH-99 origins 0..29 are validation, 30..59 test
ROLL = 30  # previous one-step errors kept in the LSSH-99 context
FIT_COLS = ["insample_mse", "insample_mae", "insample_rel_mae", "aic", "n_resid"]


# ----------------------------------------------------------------- build
def build(lssh99: Path | None, m4: Path | None, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if lssh99:
        from lssh99_league import sub_daily

        d = pd.concat([pd.read_csv(p) for p in sorted(lssh99.glob("*.csv"))], ignore_index=True)
        d = d.drop_duplicates(["forecaster", "problem", "origin", "window"], keep="last")
        d["split"] = np.where(d.origin < VALIDATION, "val", "test")
        d["sub_daily"] = d.problem.map(sub_daily)
        d.to_parquet(out / "lssh99.parquet", index=False)
        print(f"lssh99: {len(d)} rows, {d.status.ne('ok').sum()} not ok")
    if m4:
        from m4_window_score import load_truth

        d = pd.concat([pd.read_csv(p) for p in sorted(m4.glob("*.csv"))], ignore_index=True)
        d = d.drop_duplicates(["forecaster", "id", "split", "window"], keep="last")
        d["pred"] = d.pred.map(lambda s: json.loads(s) if isinstance(s, str) else None)
        d.to_parquet(out / "m4.parquet", index=False)
        ids = d[["id", "frequency"]].drop_duplicates()
        truth = load_truth(ids.groupby("frequency").id.apply(list).to_dict())
        t = pd.DataFrame([dict(id=i, frequency=v["freq"], n=v["n"], test=v["test"].tolist(),
                               holdout=v["holdout"].tolist(), naive2=v["naive2"].tolist(),
                               scale=v["scale"]) for i, v in truth.items()])
        t.to_parquet(out / "m4_truth.parquet", index=False)
        print(f"m4: {len(d)} rows, {d.status.ne('ok').sum()} not ok; {len(t)} series")
    (out / "README.txt").write_text(__doc__)


# ----------------------------------------------------------------- built-in schemes
def _builtin(name: str) -> Callable:
    from window_combine import combine

    def scheme(ctx):
        return combine(ctx["preds"], ctx["insample_mse"], ctx["insample_rel_mae"],
                       ctx.get("oos_mse"))[name]
    scheme.__name__ = name
    return scheme


BUILT_IN_NAMES = ["mean", "median", "trimmed", "inv_mse", "inv_rel_mae", "inv_rel_mae2",
                  "akaike_n100", "best_insample", "inv_oos_mse", "median_best_half"]


def built_in() -> dict[str, Callable]:
    return {n: _builtin(n) for n in BUILT_IN_NAMES}


def whole_series(ctx):
    return ctx["preds"][list(ctx["windows"]).index(0)]


# ----------------------------------------------------------------- store
class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lssh = self._m4 = self._truth = None

    @property
    def lssh99(self) -> pd.DataFrame:
        if self._lssh is None:
            self._lssh = pd.read_parquet(self.path / "lssh99.parquet")
        return self._lssh

    @property
    def m4(self) -> pd.DataFrame:
        if self._m4 is None:
            self._m4 = pd.read_parquet(self.path / "m4.parquet")
        return self._m4

    @property
    def m4_truth(self) -> pd.DataFrame:
        if self._truth is None:
            self._truth = pd.read_parquet(self.path / "m4_truth.parquet").set_index("id")
        return self._truth

    # ------------------------------------------------------------- LSSH-99
    def lssh99_contexts(self, forecaster: str):
        """Yield (problem, origin, ctx) for every test origin.

        ctx: ``windows`` (0 = whole series) and, aligned with them, ``preds``
        (shape (k, 1)), ``n_train``, the fit scores (``FIT_COLS``), ``past_err``
        ((k, 30): the window's one-step errors at the previous 30 origins, NaN
        where it was not fitted), ``oos_mse`` (their mean square); plus
        ``period``, ``sub_daily``, ``actual``, ``naive``, ``origin``.
        """
        d = self.lssh99
        d = d[(d.forecaster == forecaster) & (d.status == "ok") & d.pred.notna()]
        for p, g in d.groupby("problem"):
            e = (g.pred - g.actual).abs()
            err = g.assign(e=e).pivot_table(index="origin", columns="window", values="e")
            for o, r in g[g.origin >= VALIDATION].groupby("origin"):
                r = r.sort_values("window")
                ws = r.window.to_numpy()
                past = err.reindex(columns=ws).loc[o - ROLL:o - 1].to_numpy().T
                ctx = dict(windows=ws, preds=r.pred.to_numpy()[:, None],
                           n_train=r.n_train.to_numpy(), past_err=past,
                           oos_mse=np.nanmean(past ** 2, axis=1) if past.size else None,
                           period=int(r.period.iloc[0]), sub_daily=bool(r.sub_daily.iloc[0]),
                           actual=float(r.actual.iloc[0]), naive=float(r.naive.iloc[0]),
                           origin=int(o), problem=p,
                           **{c: r[c].to_numpy(float) for c in FIT_COLS if c in r})
                yield p, o, ctx

    def evaluate_lssh99(self, schemes: dict[str, Callable], forecasters=None,
                        include_whole=True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Relative MAE per problem (per scheme and forecaster) and a summary."""
        from lssh99_window_score import summarise

        schemes = ({"full": whole_series} if include_whole else {}) | dict(schemes)
        forecasters = forecasters or sorted(self.lssh99.forecaster.unique())
        per, summ = [], []
        for f in forecasters:
            errs, naive = {k: {} for k in schemes}, {}
            for p, o, ctx in self.lssh99_contexts(f):
                naive.setdefault(p, []).append(abs(ctx["naive"] - ctx["actual"]))
                for k, fn in schemes.items():
                    errs[k].setdefault(p, []).append(abs(float(np.ravel(fn(ctx))[0]) - ctx["actual"]))
            nv = pd.Series({p: np.mean(v) for p, v in naive.items()})
            rel = pd.DataFrame({k: pd.Series({p: np.mean(v) for p, v in e.items()})
                                for k, e in errs.items()}).div(nv.where(nv > 0), axis=0)
            rel = rel.rename_axis("problem")
            per.append(rel.assign(forecaster=f).reset_index())
            fit = self.lssh99[self.lssh99.forecaster == f].groupby("window").fit_seconds.median()
            summ.append(summarise(rel, fit).assign(forecaster=f).reset_index())
        return pd.concat(per, ignore_index=True), pd.concat(summ, ignore_index=True)

    # ------------------------------------------------------------- M4
    def m4_contexts(self, forecaster: str):
        """Yield (id, ctx) per series.

        ctx: ``windows`` (test fits) and, aligned with them, ``preds`` (k, h),
        ``n_train``, the fit scores; ``val_windows``/``val_preds`` (the holdout
        fits) and ``oos_mse`` (holdout MSE of each test window, NaN where it had
        no holdout fit); ``holdout``, ``frequency``, ``h``, ``n``.
        """
        d = self.m4
        d = d[(d.forecaster == forecaster) & (d.status == "ok") & d.pred.notna()]
        truth = self.m4_truth
        for i, g in d.groupby("id"):
            t = truth.loc[i]
            te = g[g.split == "test"].sort_values("window")
            va = g[g.split == "val"].set_index("window")
            if 0 not in set(te.window):
                continue
            ws = te.window.to_numpy()
            hold = np.asarray(t.holdout, float)
            oos = np.array([np.mean((np.asarray(va.pred[w]) - hold) ** 2) if w in va.index
                            else np.nan for w in ws])
            ctx = dict(windows=ws, preds=np.stack([np.asarray(p, float) for p in te.pred]),
                       n_train=te.n_train.to_numpy(), oos_mse=oos,
                       val_windows=va.index.to_numpy(),
                       val_preds=np.stack([np.asarray(p, float) for p in va.pred]) if len(va) else None,
                       holdout=hold, frequency=t.frequency, h=len(t.test), n=int(t.n), id=i,
                       **{c: te[c].to_numpy(float) for c in FIT_COLS if c in te})
            yield i, ctx

    def evaluate_m4(self, schemes: dict[str, Callable], forecasters=None,
                    include_whole=True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Per-series sMAPE/MASE and OWA (vs Naive2) by frequency, band and overall."""
        from m4_window_score import band, owa_table, smape_mase, versus_full

        schemes = ({"full": whole_series} if include_whole else {}) | dict(schemes)
        forecasters = forecasters or sorted(self.m4.forecaster.unique())
        truth = self.m4_truth
        per_all, tabs = [], []
        for f in forecasters:
            rows = []
            for i, ctx in self.m4_contexts(f):
                t = truth.loc[i]
                actual = np.asarray(t.test, float)
                for k, fn in schemes.items():
                    s, q = smape_mase(np.ravel(fn(ctx))[: ctx["h"]], actual, t.scale)
                    rows.append(dict(id=i, frequency=t.frequency, n=int(t.n), method=k,
                                     smape=s, mase=q))
                s, q = smape_mase(np.asarray(t.naive2, float), actual, t.scale)
                rows.append(dict(id=i, frequency=t.frequency, n=int(t.n), method="Naive2",
                                 smape=s, mase=q))
            per = pd.DataFrame(rows).assign(forecaster=f)
            per["band"] = per.n.map(band)
            per_all.append(per)
            from m4_score import FREQS
            fr = owa_table(per, "frequency")
            fr = fr.reindex(columns=[q for q in FREQS if q in fr.columns] + ["all"])
            bd = owa_table(per, "band").drop(columns="all")
            tabs.append(fr.join(bd).join(versus_full(per)).assign(forecaster=f)
                        .rename_axis("method").reset_index())
        return pd.concat(per_all, ignore_index=True), pd.concat(tabs, ignore_index=True)


# ----------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--lssh99", type=Path)
    b.add_argument("--m4", type=Path)
    b.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate", help="score every built-in scheme")
    e.add_argument("--store", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "build":
        build(args.lssh99, args.m4, args.out)
        return
    s = Store(args.store)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 220)
    if (args.store / "lssh99.parquet").exists():
        per, summ = s.evaluate_lssh99(built_in())
        per.to_csv(args.out / "lssh99_per_problem.csv", index=False)
        summ.to_csv(args.out / "lssh99_summary.csv", index=False)
        for f, d in summ.groupby("forecaster"):
            print(f"\n== LSSH-99 {f}\n{d.drop(columns='forecaster').set_index('method').to_string()}")
    if (args.store / "m4.parquet").exists():
        per, tab = s.evaluate_m4(built_in())
        per.to_csv(args.out / "m4_per_series.csv.gz", index=False)
        tab.to_csv(args.out / "m4_owa.csv", index=False)
        for f, d in tab.groupby("forecaster"):
            print(f"\n== M4 {f}\n{d.drop(columns='forecaster').set_index('method').to_string()}")


if __name__ == "__main__":
    main()
