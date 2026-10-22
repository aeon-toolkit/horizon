"""Smoke tests: the pipeline runs end to end on a tiny configuration."""

import json

import numpy as np
import pytest

from horizon.adapters import get_adapters
from horizon.simulate import ALL_PROCESSES, TRUE_MODELS, TRUE_SPECS


@pytest.mark.parametrize("name", sorted(ALL_PROCESSES))
def test_processes_sample(name):
    """Every process gives finite, positive series of the requested length."""
    y = ALL_PROCESSES[name].sample(120, random_state=0)
    assert y.shape == (120,)
    assert np.all(np.isfinite(y)) and np.all(y > 0)
    # seeded: the same seed reproduces the series
    assert np.array_equal(y, ALL_PROCESSES[name].sample(120, random_state=0))
    assert name in TRUE_MODELS and name in TRUE_SPECS


@pytest.mark.parametrize("mode", ["auto", "fixed"])
@pytest.mark.parametrize("family, process", [("arima", "arma_11"), ("ets", "ets_aan")])
def test_aeon_adapters(mode, family, process):
    """aeon adapters return a forecast of the right shape and name the model."""
    proc = ALL_PROCESSES[process]
    y = proc.sample(100, random_state=1)
    for a in get_adapters(family, include=["aeon"], mode=mode):
        a.spec = TRUE_SPECS[process]
        pred = a(y, 6, proc.seasonal_period)
        assert pred.shape == (6,) and np.all(np.isfinite(pred))
        assert a.selected_model.startswith(family.upper())


def test_run_analyse_compare(tmp_path):
    """A tiny run, its analysis, and a self-comparison all complete."""
    from horizon.analyse import analyse_all
    from horizon.compare import compare_runs
    from horizon.experiments import run_accuracy, run_timing

    out = tmp_path / "run"
    run_accuracy(
        families=["arima"],
        processes=["arma_11"],
        lengths=[60],
        n_replicates=6,
        toolkits=["aeon", "statsmodels (grid)"],
        results_dir=out,
        verbose=False,
    )
    run_timing(
        processes={"arima": ["arma_11"]},
        lengths=[60, 120],
        n_repeats=2,
        toolkits=["aeon"],
        results_dir=out,
        verbose=False,
    )
    analyse_all(out)
    for name in ["accuracy_summary.csv", "significance.csv", "timing_summary.csv"]:
        assert (out / name).exists()
    assert (out / "plots" / "timing_arima_arma_11.png").exists()

    res = compare_runs(out, out, tmp_path / "cmp")
    acc = res["accuracy_change"]
    # identical runs: no change for aeon or the other toolkits
    assert (acc["aeon_a_median_mase"] == acc["aeon_b_median_mase"]).all()
    assert (acc["toolkits_max_abs_change"] == 0).all()
    assert (acc["change"] == "no significant change").all()


def test_resume_skips_finished_rows(tmp_path):
    """Rerunning an experiment does not duplicate rows."""
    import pandas as pd

    from horizon.experiments import run_accuracy

    kwargs = dict(
        families=["arima"],
        processes=["white_noise"],
        lengths=[50],
        n_replicates=3,
        toolkits=["aeon"],
        results_dir=tmp_path,
        verbose=False,
    )
    run_accuracy(**kwargs)
    run_accuracy(**kwargs)
    assert len(pd.read_csv(tmp_path / "accuracy.csv")) == 3


def test_run_info_records_aeon():
    """run_info names the aeon that was used."""
    from horizon.cli import run_info

    info = run_info(["run"])
    assert info["aeon"]["version"]
    assert json.dumps(info)


def test_sweep_quick(tmp_path):
    """A tiny sweep runs in parallel, merges its parts and is analysed."""
    import pandas as pd

    from horizon.sweep import analyse_sweep, run_sweep_accuracy, run_sweep_timing

    acc = run_sweep_accuracy(
        tmp_path, lengths=[60, 120], replicates=3, workers=2,
        families=["arima"], toolkits=["aeon", "statsmodels (grid)"],
    )
    # 6 ARIMA processes x 2 lengths x 3 replicates x 2 toolkits
    assert len(acc) == 6 * 2 * 3 * 2
    # lengths get independent seeds
    assert acc.groupby("n")["seed"].apply(set).pipe(lambda s: not (s[60] & s[120]))
    run_sweep_timing(
        tmp_path, lengths=[60, 120], repeats=2,
        processes={"arima": ["arma_11"]}, toolkits=["aeon"],
    )
    analyse_sweep(tmp_path)
    eq = pd.read_csv(tmp_path / "equivalence.csv")
    assert set(eq["verdict"]) <= {"equivalent", "aeon better", "aeon worse", "inconclusive"}
    assert (eq["ratio_ci_low"] <= eq["ratio"]).all()
    assert (eq["ratio"] <= eq["ratio_ci_high"]).all()
    assert (tmp_path / "plots" / "equivalence_arima.png").exists()
    assert (tmp_path / "plots" / "speedup_arima_arma_11.png").exists()


def test_tost():
    """TOST: identical forecasts are equivalent, a 20% gap is not."""
    import numpy as np

    from horizon.sweep import _tost

    rng = np.random.default_rng(0)
    assert _tost(rng.normal(0, 0.05, 400), 0.05, 0.05)["p_tost"] < 0.05
    r = _tost(rng.normal(np.log(1.2), 0.05, 400), 0.05, 0.05)
    assert r["p_tost"] > 0.5 and r["ratio_ci_low"] > 1.05
    assert _tost(np.zeros(10), 0.05, 0.05)["p_tost"] == 0.0
