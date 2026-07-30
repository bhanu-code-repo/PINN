"""CLI smoke tests: full train -> predict -> compare lifecycle per experiment.

Uses Typer's in-process CliRunner (no subprocesses), tiny epoch counts, and
temp directories — this codifies the manual smoke sequence used during
development.
"""

import pytest
from typer.testing import CliRunner

import experiments.common as common
from experiments.burgers.train import app as burgers_app
from experiments.harmonic_oscillator.train import app as harmonic_app
from experiments.schrodinger.train import app as schrodinger_app

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
