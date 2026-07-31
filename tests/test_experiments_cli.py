"""CLI smoke tests: full train -> predict -> compare lifecycle per experiment.

Uses Typer's in-process CliRunner (no subprocesses), tiny epoch counts, and
temp directories — this codifies the manual smoke sequence used during
development.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import experiments.common as common
from experiments.burgers.train import app as burgers_app
from experiments.cylinder_wake.train import app as cylinder_app
from experiments.harmonic_oscillator.train import app as harmonic_app
from experiments.lid_driven_cavity.train import app as cavity_app
from experiments.navier_stokes_inverse.train import app as ns_inverse_app
from experiments.parametric_burgers.train import app as parametric_burgers_app
from experiments.parametric_harmonic.train import app as parametric_app
from experiments.parametric_schrodinger.train import app as parametric_nls_app
from experiments.schrodinger.train import app as schrodinger_app
from experiments.taylor_green.train import app as taylor_green_app

runner = CliRunner()

CASES = [
    pytest.param("harmonic_oscillator", harmonic_app, ["prediction.png"], id="harmonic"),
    pytest.param("burgers", burgers_app,
                 ["prediction_contour.png", "prediction_snapshots.png"], id="burgers"),
    pytest.param("schrodinger", schrodinger_app,
                 ["prediction_contour.png", "prediction_snapshots.png"], id="schrodinger"),
]

TRAIN_ARTIFACTS = ["checkpoint.pt", "metrics.json", "loss_history.png"]


def invoke(app, args):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, f"exit {result.exit_code}:\n{result.output}"
    return result


@pytest.mark.parametrize(("experiment", "app", "predict_artifacts"), CASES)
def test_train_predict_compare_lifecycle(experiment, app, predict_artifacts, tmp_path, monkeypatch):
    run_dir = tmp_path / experiment / "run1"

    # --- train (tiny) ---
    invoke(app, ["train", "-e", "3", "--seed", "0", "--no-show", "-o", str(run_dir)])
    for artifact in TRAIN_ARTIFACTS:
        assert (run_dir / artifact).exists(), f"missing {artifact}"
    assert list((run_dir / "logs").glob("run_*.log")), "missing log file"

    # --- predict (loads the self-describing checkpoint) ---
    invoke(app, ["predict", "--run", str(run_dir), "--no-show"])
    assert (run_dir / "predictions.npz").exists()
    for artifact in predict_artifacts:
        assert (run_dir / artifact).exists(), f"missing {artifact}"

    # --- compare (discovers the run under the outputs root) ---
    monkeypatch.setattr(common, "OUTPUTS_ROOT", tmp_path)
    result = invoke(app, ["compare"])
    assert "No runs" not in result.output


def test_predict_latest_run_default(tmp_path, monkeypatch):
    """predict without --run resolves the newest completed run."""
    monkeypatch.setattr(common, "OUTPUTS_ROOT", tmp_path)
    older = tmp_path / "harmonic_oscillator" / "20200101-000000"
    newer = tmp_path / "harmonic_oscillator" / "20300101-000000"
    invoke(harmonic_app, ["train", "-e", "3", "--seed", "0", "--no-show", "-o", str(older)])
    invoke(harmonic_app, ["train", "-e", "3", "--seed", "1", "--no-show", "-o", str(newer)])

    invoke(harmonic_app, ["predict", "--no-show"])
    assert (newer / "predictions.npz").exists()
    assert not (older / "predictions.npz").exists()


def test_predict_without_any_run_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "OUTPUTS_ROOT", tmp_path / "empty")
    result = runner.invoke(harmonic_app, ["predict", "--no-show"])
    assert result.exit_code != 0


def test_parametric_ensemble_lifecycle(tmp_path, monkeypatch):
    """Parametric experiment: ensemble train -> predict with sigma band -> compare."""
    import numpy as np

    run_dir = tmp_path / "parametric_harmonic" / "run1"

    # --- train a tiny 2-member ensemble ---
    invoke(parametric_app, [
        "train", "-e", "3", "--n-physics", "200", "--ensemble", "2",
        "--seed", "0", "--no-show", "-o", str(run_dir),
    ])
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "checkpoint_1.pt").exists()
    assert (run_dir / "metrics.json").exists()

    # --- predict a never-trained instance: mean + std saved ---
    invoke(parametric_app, [
        "predict", "--w0", "40", "-d", "1.5", "--run", str(run_dir), "--no-show",
    ])
    data = np.load(run_dir / "predictions.npz")
    assert data["u_mean"].shape == data["u_std"].shape
    assert (data["u_std"] >= 0).all()
    assert float(data["u_std"].max()) > 0  # two members must disagree somewhere
    assert (run_dir / "prediction.png").exists()

    # --- out-of-box instance still runs but warns ---
    result = invoke(parametric_app, [
        "predict", "--w0", "150", "-d", "1.0", "--run", str(run_dir), "--no-show",
    ])
    assert "OUTSIDE the trained box" in result.output

    # --- compare discovers the run ---
    monkeypatch.setattr(common, "OUTPUTS_ROOT", tmp_path)
    result = invoke(parametric_app, ["compare"])
    assert "No runs" not in result.output


def test_parametric_burgers_ensemble_lifecycle(tmp_path, monkeypatch):
    """Parametric Burgers: ensemble train -> predict at new nu -> compare."""
    import numpy as np

    run_dir = tmp_path / "parametric_burgers" / "run1"

    # --- train a tiny 2-member ensemble ---
    invoke(parametric_burgers_app, [
        "train", "-e", "3", "--n-physics", "200", "--ensemble", "2",
        "--seed", "0", "--no-show", "-o", str(run_dir),
    ])
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "checkpoint_1.pt").exists()
    assert (run_dir / "metrics.json").exists()

    # --- predict a never-trained viscosity: mean + std saved ---
    invoke(parametric_burgers_app, [
        "predict", "--nu", "0.05", "--run", str(run_dir), "--no-show",
    ])
    data = np.load(run_dir / "predictions.npz")
    assert data["u_mean"].shape == data["u_std"].shape
    assert float(data["u_std"].max()) > 0  # two members must disagree somewhere
    assert (run_dir / "prediction_contour.png").exists()
    assert (run_dir / "prediction_snapshots.png").exists()

    # --- out-of-range viscosity still runs but warns ---
    result = invoke(parametric_burgers_app, [
        "predict", "--nu", "0.5", "--run", str(run_dir), "--no-show",
    ])
    assert "OUTSIDE the trained box" in result.output

    # --- compare discovers the run ---
    monkeypatch.setattr(common, "OUTPUTS_ROOT", tmp_path)
    result = invoke(parametric_burgers_app, ["compare"])
    assert "No runs" not in result.output


def test_parametric_schrodinger_ensemble_lifecycle(tmp_path, monkeypatch):
    """Parametric NLS soliton: ensemble train -> predict at new A -> compare."""
    import numpy as np

    run_dir = tmp_path / "parametric_schrodinger" / "run1"

    # --- train a tiny 2-member ensemble ---
    invoke(parametric_nls_app, [
        "train", "-e", "3", "--n-physics", "200", "--ensemble", "2",
        "--seed", "0", "--no-show", "-o", str(run_dir),
    ])
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "checkpoint_1.pt").exists()
    assert (run_dir / "metrics.json").exists()

    # --- predict a never-trained amplitude: complex mean + std saved ---
    invoke(parametric_nls_app, [
        "predict", "-a", "1.3", "--run", str(run_dir), "--no-show",
    ])
    data = np.load(run_dir / "predictions.npz")
    assert data["u_mean"].shape == data["v_mean"].shape == data["h_mag_mean"].shape
    assert float(data["h_mag_std"].max()) > 0  # two members must disagree somewhere
    assert (run_dir / "prediction_contour.png").exists()
    assert (run_dir / "prediction_snapshots.png").exists()

    # --- out-of-range amplitude still runs but warns ---
    result = invoke(parametric_nls_app, [
        "predict", "-a", "3.0", "--run", str(run_dir), "--no-show",
    ])
    assert "OUTSIDE the trained box" in result.output

    # --- compare discovers the run ---
    monkeypatch.setattr(common, "OUTPUTS_ROOT", tmp_path)
    result = invoke(parametric_nls_app, ["compare"])
    assert "No runs" not in result.output


def test_taylor_green_lifecycle(tmp_path, monkeypatch):
    """Taylor-Green vortex: train -> predict -> compare."""
    run_dir = tmp_path / "taylor_green" / "run1"

    invoke(taylor_green_app, [
        "train", "-e", "3", "--n-physics", "200",
        "--seed", "0", "--no-show", "-o", str(run_dir),
    ])
    for artifact in TRAIN_ARTIFACTS:
        assert (run_dir / artifact).exists(), f"missing {artifact}"

    invoke(taylor_green_app, ["predict", "--run", str(run_dir), "--no-show"])
    assert (run_dir / "predictions.npz").exists()
    assert (run_dir / "prediction_comparison.png").exists()

    monkeypatch.setattr(common, "OUTPUTS_ROOT", tmp_path)
    result = invoke(taylor_green_app, ["compare"])
    assert "No runs" not in result.output


def test_lid_driven_cavity_lifecycle(tmp_path, monkeypatch):
    """Lid-driven cavity: train -> predict -> compare."""
    run_dir = tmp_path / "lid_driven_cavity" / "run1"

    invoke(cavity_app, [
        "train", "-e", "3", "--n-physics", "200",
        "--seed", "0", "--no-show", "-o", str(run_dir),
    ])
    for artifact in TRAIN_ARTIFACTS:
        assert (run_dir / artifact).exists(), f"missing {artifact}"

    invoke(cavity_app, ["predict", "--run", str(run_dir), "--no-show"])
    assert (run_dir / "predictions.npz").exists()
    assert (run_dir / "prediction_cavity.png").exists()

    monkeypatch.setattr(common, "OUTPUTS_ROOT", tmp_path)
    result = invoke(cavity_app, ["compare"])
    assert "No runs" not in result.output


def test_navier_stokes_inverse_lifecycle(tmp_path, monkeypatch):
    """Inverse NS (Kovasznay): train -> predict -> compare, check Re is saved."""
    import numpy as np

    run_dir = tmp_path / "navier_stokes_inverse" / "run1"

    invoke(ns_inverse_app, [
        "train", "-e", "3", "--n-physics", "200", "--n-obs", "50",
        "--seed", "0", "--no-show", "-o", str(run_dir),
    ])
    for artifact in TRAIN_ARTIFACTS:
        assert (run_dir / artifact).exists(), f"missing {artifact}"
    assert (run_dir / "observations.npz").exists()

    invoke(ns_inverse_app, ["predict", "--run", str(run_dir), "--no-show"])
    data = np.load(run_dir / "predictions.npz")
    assert "re_inferred" in data
    assert (run_dir / "prediction_inverse.png").exists()

    monkeypatch.setattr(common, "OUTPUTS_ROOT", tmp_path)
    result = invoke(ns_inverse_app, ["compare"])
    assert "No runs" not in result.output


def test_cylinder_wake_lifecycle(tmp_path, monkeypatch):
    """Cylinder wake inverse: train -> predict -> compare with DNS data."""
    import numpy as np

    data_path = Path(".workspace/input/cylinder_nektar_wake.mat")
    if not data_path.exists():
        pytest.skip("cylinder_nektar_wake.mat not found in .workspace/input/")

    run_dir = tmp_path / "cylinder_wake" / "run1"

    invoke(cylinder_app, [
        "train", "-e", "3", "--n-train", "100", "--n-physics", "200",
        "--seed", "0", "--no-show", "-o", str(run_dir),
    ])
    for artifact in TRAIN_ARTIFACTS:
        assert (run_dir / artifact).exists(), f"missing {artifact}"

    invoke(cylinder_app, ["predict", "--run", str(run_dir), "--no-show"])
    data = np.load(run_dir / "predictions.npz")
    assert "lambda_1" in data
    assert "lambda_2" in data
    assert (run_dir / "prediction_snapshot.png").exists()

    monkeypatch.setattr(common, "OUTPUTS_ROOT", tmp_path)
    result = invoke(cylinder_app, ["compare"])
    assert "No runs" not in result.output
