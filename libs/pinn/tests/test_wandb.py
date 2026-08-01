"""Tests for W&B integration (mocked — no real W&B calls)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_wandb():
    """Mock the wandb module so tests run without it installed."""
    mock = MagicMock()
    mock.init.return_value = MagicMock(name="test-run", url="https://wandb.ai/test")
    with patch.dict("sys.modules", {"wandb": mock}):
        yield mock


class TestWandbInit:
    def test_creates_run(self, mock_wandb):
        from pinn.wandb_integration import wandb_init

        run = wandb_init(project="test", config={"lr": 1e-3}, name="my-run")
        mock_wandb.init.assert_called_once_with(
            project="test", config={"lr": 1e-3}, name="my-run",
            tags=None, group=None,
        )
        assert run is not None

    def test_forwards_kwargs(self, mock_wandb):
        from pinn.wandb_integration import wandb_init

        wandb_init(project="p", tags=["test"], group="burgers", reinit=True)
        mock_wandb.init.assert_called_once_with(
            project="p", config=None, name=None,
            tags=["test"], group="burgers", reinit=True,
        )


class TestWandbCallback:
    def test_logs_every_epoch(self, mock_wandb):
        from pinn.wandb_integration import wandb_callback

        cb = wandb_callback()
        cb(0, {"ic": 1.0, "physics": 0.5, "total": 1.5})
        cb(1, {"ic": 0.8, "physics": 0.4, "total": 1.2})

        assert mock_wandb.log.call_count == 2
        first_call = mock_wandb.log.call_args_list[0]
        assert first_call[0][0] == {"ic": 1.0, "physics": 0.5, "total": 1.5, "epoch": 0}
        assert first_call[1] == {"step": 0}

    def test_log_every_n(self, mock_wandb):
        from pinn.wandb_integration import wandb_callback

        cb = wandb_callback(log_every=3)
        for i in range(9):
            cb(i, {"total": float(i)})

        # Epochs 0, 3, 6 logged
        assert mock_wandb.log.call_count == 3

    def test_prefix(self, mock_wandb):
        from pinn.wandb_integration import wandb_callback

        cb = wandb_callback(prefix="train/")
        cb(5, {"ic": 0.1, "total": 0.2})

        logged = mock_wandb.log.call_args[0][0]
        assert "train/ic" in logged
        assert "train/total" in logged
        assert "epoch" in logged


class TestWandbFinish:
    def test_finish_without_artifacts(self, mock_wandb):
        from pinn.wandb_integration import wandb_finish

        wandb_finish()
        mock_wandb.finish.assert_called_once()
        mock_wandb.Artifact.assert_not_called()

    def test_finish_with_artifacts(self, mock_wandb, tmp_path):
        from pinn.wandb_integration import wandb_finish

        # Create some fake artifacts
        (tmp_path / "checkpoint.pt").write_text("fake")
        (tmp_path / "metrics.json").write_text("{}")
        (tmp_path / "loss_history.png").write_text("fake")

        wandb_finish(run_dir=tmp_path)

        mock_wandb.Artifact.assert_called_once_with("run-artifacts", type="model")
        artifact = mock_wandb.Artifact.return_value
        assert artifact.add_file.call_count == 3
        mock_wandb.log_artifact.assert_called_once_with(artifact)
        mock_wandb.finish.assert_called_once()

    def test_custom_artifact_name(self, mock_wandb, tmp_path):
        from pinn.wandb_integration import wandb_finish

        (tmp_path / "checkpoint.pt").write_text("fake")
        wandb_finish(run_dir=tmp_path, artifact_name="burgers-run")

        mock_wandb.Artifact.assert_called_once_with("burgers-run", type="model")


class TestImportError:
    def test_clear_error_when_wandb_missing(self):
        """Importing without wandb installed gives a helpful message."""
        with patch.dict("sys.modules", {"wandb": None}):
            from pinn.wandb_integration import _import_wandb

            with pytest.raises(ImportError, match="uv add wandb"):
                _import_wandb()
