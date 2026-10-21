"""Command line interface: ``python -m horizon <command>``.

Commands
--------
run
    Run the accuracy and/or timing experiment and analyse the results::

        python -m horizon run --mode auto --out results/my-run
        python -m horizon run --mode fixed --out results/my-run-fixed --quick

analyse
    Rebuild the tables and plots of a finished run::

        python -m horizon analyse results/my-run

sweep
    Accuracy equivalence and run time over a fine grid of lengths (100 to
    1000 in steps of 100, then to 10000 in steps of 1000), in parallel::

        python -m horizon sweep --mode auto --out results/my-run/sweep-auto --workers 56
        python -m horizon sweep --analyse-only --out results/my-run/sweep-auto

compare
    Compare two runs of the same experiment (e.g. two aeon versions)::

        python -m horizon compare results/aeon-1.6.0/auto results/branch/auto \\
            --out results/comparisons/branch-vs-1.6.0-auto

    or a run's automatic forecasts with its true model forecasts::

        python -m horizon compare results/x/auto results/x/fixed --auto-fixed

Runs append to their output directory and skip finished rows, so an interrupted
run resumes where it stopped.
"""

import os

# Pin every toolkit to one thread before numpy is imported anywhere, so timings
# compare algorithms rather than how many cores each library uses.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

TIMING_PROCESSES = {"arima": ["arma_11"], "ets": ["ets_aaa_m12"]}


def _git(path: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


def run_info(argv) -> dict:
    """Environment of a run: aeon source and commit, package versions, host."""
    import aeon

    from horizon.adapters import availability_table

    aeon_dir = Path(aeon.__file__).resolve().parent
    status = _git(aeon_dir, "status", "--porcelain", "--untracked-files=no")
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(["python -m horizon", *argv]),
        "aeon": {
            "version": aeon.__version__,
            "path": str(aeon_dir),
            "git_commit": _git(aeon_dir, "rev-parse", "HEAD"),
            "git_branch": _git(aeon_dir, "rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(status),
        },
        "packages": {
            r["package"]: r["version"]
            for r in availability_table().to_dict("records")
            if r["available"]
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR", ""),
    }


def _cmd_run(args, argv):
    from horizon.analyse import analyse_all
    from horizon.experiments import run_accuracy, run_timing

    if args.quick:
        args.replicates = min(args.replicates, 5)
        args.lengths = [100]
        args.timing_lengths = [100, 500, 1000]
        args.repeats = min(args.repeats, 2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = run_info(argv)
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"aeon {info['aeon']['version']} from {info['aeon']['path']}")
    print(f"  commit {info['aeon']['git_commit'][:12]} on {info['aeon']['git_branch']}"
          + (" (uncommitted changes)" if info["aeon"]["git_dirty"] else ""))

    if args.only in (None, "accuracy"):
        run_accuracy(
            families=args.families,
            processes=args.processes,
            lengths=args.lengths,
            n_replicates=args.replicates,
            toolkits=args.toolkits,
            results_dir=out,
            mode=args.mode,
        )
    if args.only in (None, "timing"):
        run_timing(
            processes={f: p for f, p in TIMING_PROCESSES.items() if f in args.families},
            lengths=args.timing_lengths,
            n_repeats=args.repeats,
            toolkits=args.toolkits,
            time_budget=args.time_budget,
            results_dir=out,
            mode=args.mode,
        )
    analyse_all(out)


def _cmd_sweep(args, argv):
    from horizon.sweep import analyse_sweep, run_sweep_accuracy, run_sweep_timing

    out = Path(args.out)
    if not args.analyse_only:
        out.mkdir(parents=True, exist_ok=True)
        info = run_info(argv)
        (out / "run_info.json").write_text(json.dumps(info, indent=2))
        print(f"aeon {info['aeon']['version']} commit "
              f"{info['aeon']['git_commit'][:12]} from {info['aeon']['path']}"
              + (" (uncommitted changes)" if info["aeon"]["git_dirty"] else ""))
        if args.only in (None, "accuracy"):
            run_sweep_accuracy(out, mode=args.mode, lengths=args.lengths,
                               replicates=args.replicates, workers=args.workers,
                               families=args.families, toolkits=args.toolkits)
        if args.only in (None, "timing"):
            run_sweep_timing(out, mode=args.mode, lengths=args.lengths,
                             repeats=args.repeats, toolkits=args.toolkits,
                             time_budget=args.time_budget)
    analyse_sweep(out, margin=args.margin)


def _cmd_analyse(args, argv):
    from horizon.analyse import analyse_all

    analyse_all(Path(args.run))


def _cmd_compare(args, argv):
    from horizon.compare import compare_auto_fixed, compare_runs

    if args.auto_fixed:
        out = compare_auto_fixed(Path(args.run_a), Path(args.run_b))
        pooled = out[out["scope"] == "all"]
        print(pooled.to_string(index=False, float_format="%.4g"))
        return
    if args.out is None:
        raise SystemExit("compare needs --out unless --auto-fixed is given")
    res = compare_runs(Path(args.run_a), Path(args.run_b), Path(args.out))
    acc = res["accuracy_change"]
    cols = [
        "family", "n", "scope", "aeon_a_median_mase", "aeon_b_median_mase",
        "competitors_min_median_mase", "competitors_max_median_mase", "p_holm",
        "change",
    ]
    print(acc[cols].to_string(index=False, float_format="%.4g"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="python -m horizon", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run experiments")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--mode", choices=["auto", "fixed"], default="auto",
                   help="auto: model selection; fixed: fit the true generating model")
    p.add_argument("--only", choices=["accuracy", "timing"], default=None)
    p.add_argument("--quick", action="store_true", help="small smoke run")
    p.add_argument("--families", nargs="+", default=["arima", "ets"])
    p.add_argument("--processes", nargs="+", default=None)
    p.add_argument("--toolkits", nargs="+", default=None)
    p.add_argument("--replicates", type=int, default=100)
    p.add_argument("--lengths", type=int, nargs="+", default=[100, 500])
    p.add_argument("--timing-lengths", type=int, nargs="+",
                   default=[100, 200, 500, 1000, 2000, 5000, 10000])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--time-budget", type=float, default=30.0)
    p.set_defaults(func=_cmd_run)

    from horizon.sweep import SWEEP_LENGTHS

    p = sub.add_parser("sweep", help="accuracy equivalence and run time over many lengths")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--mode", choices=["auto", "fixed"], default="auto")
    p.add_argument("--only", choices=["accuracy", "timing"], default=None)
    p.add_argument("--analyse-only", action="store_true")
    p.add_argument("--families", nargs="+", default=["arima", "ets"])
    p.add_argument("--toolkits", nargs="+", default=None)
    p.add_argument("--lengths", type=int, nargs="+", default=SWEEP_LENGTHS)
    p.add_argument("--replicates", type=int, default=50)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--time-budget", type=float, default=30.0)
    p.add_argument("--margin", type=float, default=0.05,
                   help="equivalence margin on the MASE ratio (0.05: within 5%%)")
    p.set_defaults(func=_cmd_sweep)

    p = sub.add_parser("analyse", help="rebuild tables and plots of a run")
    p.add_argument("run")
    p.set_defaults(func=_cmd_analyse)

    p = sub.add_parser("compare", help="compare two runs")
    p.add_argument("run_a", help="baseline run")
    p.add_argument("run_b", help="changed run")
    p.add_argument("--out", default=None)
    p.add_argument("--auto-fixed", action="store_true",
                   help="run_a is an auto run and run_b the fixed run of the same aeon")
    p.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    args.func(args, argv)


if __name__ == "__main__":
    main()
