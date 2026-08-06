"""Tests for training feedback: health monitor, adaptive weighting, quality evaluation."""

import torch
from pinn import (
    PINN,
    AdaptiveLossWeighter,
    PINNTrainer,
    TrainingHealthMonitor,
    evaluate_quality,
)

# -------------------------------------------------------- TrainingHealthMonitor


class TestTrainingHealthMonitor:
    def test_tracks_losses(self):
        model = PINN(input_dim=1, hidden_layers=1, hidden_neurons=4)
        monitor = TrainingHealthMonitor(model)
        monitor(0, {"ic": 1.0, "physics": 0.5, "total": 1.5})
        monitor(1, {"ic": 0.8, "physics": 0.4, "total": 1.2})

        assert len(monitor.total_losses) == 2
        assert monitor.total_losses == [1.5, 1.2]

    def test_loss_smoothness_stable_training(self):
        model = PINN(input_dim=1, hidden_layers=1, hidden_neurons=4)
        monitor = TrainingHealthMonitor(model)
        # Simulate smooth decay
        for i in range(100):
            loss = 1.0 * (0.99**i)
            monitor(i, {"total": loss})

        assert monitor.loss_smoothness > 0.8

    def test_loss_smoothness_unstable_training(self):
        model = PINN(input_dim=1, hidden_layers=1, hidden_neurons=4)
        monitor = TrainingHealthMonitor(model)
        # Simulate oscillating loss
        for i in range(100):
            loss = 1.0 + (-1) ** i * 0.5
            monitor(i, {"total": loss})

        assert monitor.loss_smoothness < 0.5

    def test_gradient_health_tracked(self):
        model = PINN(input_dim=1, hidden_layers=1, hidden_neurons=4)
        monitor = TrainingHealthMonitor(model)

        # Do a forward/backward to create gradients
        x = torch.rand(10, 1, requires_grad=True)
        loss = model(x).sum()
        loss.backward()

        monitor(0, {"total": loss.item()})
        assert len(monitor.grad_norms) == 1
        assert monitor.grad_norms[0] > 0

    def test_convergence_epoch_found(self):
        model = PINN(input_dim=1, hidden_layers=1, hidden_neurons=4)
        monitor = TrainingHealthMonitor(model)
        monitor(0, {"total": 1.0})
        monitor(1, {"total": 0.01})
        monitor(2, {"total": 1e-5})

        assert monitor.convergence_epoch == 2

    def test_convergence_epoch_not_reached(self):
        model = PINN(input_dim=1, hidden_layers=1, hidden_neurons=4)
        monitor = TrainingHealthMonitor(model)
        monitor(0, {"total": 1.0})
        monitor(1, {"total": 0.5})

        assert monitor.convergence_epoch is None

    def test_report_returns_all_fields(self):
        model = PINN(input_dim=1, hidden_layers=1, hidden_neurons=4)
        monitor = TrainingHealthMonitor(model)
        monitor(0, {"total": 1.0})

        report = monitor.report()
        assert "loss_smoothness" in report
        assert "gradient_healthy" in report
        assert "mean_grad_norm" in report
        assert "convergence_epoch" in report
        assert "final_loss" in report
        assert "epochs_tracked" in report

    def test_integrates_with_trainer(self, tiny_model, cpu):
        monitor = TrainingHealthMonitor(tiny_model)
        trainer = PINNTrainer(tiny_model, device=cpu)

        t = torch.linspace(0, 1, 20).unsqueeze(1).requires_grad_(True)

        def physics_loss(model):
            u = model(t)
            du = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
            return torch.mean((du + u) ** 2)

        def ic_loss(model):
            return (model(torch.zeros(1, 1)) - 1.0) ** 2

        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        trainer.train(
            n_epochs=50, optimizer=optimizer,
            loss_functions={"ic": ic_loss, "physics": physics_loss},
            callbacks=[monitor], verbose=False,
        )

        assert len(monitor.total_losses) == 50
        assert len(monitor.grad_norms) == 50
        report = monitor.report()
        assert report["epochs_tracked"] == 50


# -------------------------------------------------------- AdaptiveLossWeighter


class TestAdaptiveLossWeighter:
    def test_no_rebalance_when_balanced(self):
        weights = {"ic": 1.0, "physics": 1.0}
        weighter = AdaptiveLossWeighter(weights, rebalance_every=5, ratio_threshold=5.0)

        # Feed balanced losses
        for i in range(10):
            weighter(i, {"ic": 0.1, "physics": 0.1, "total": 0.2})

        # Weights should be unchanged
        assert weights["ic"] == 1.0
        assert weights["physics"] == 1.0

    def test_rebalance_when_imbalanced(self):
        weights = {"ic": 1.0, "physics": 1.0}
        weighter = AdaptiveLossWeighter(
            weights, rebalance_every=5, ratio_threshold=3.0, window=5,
        )

        # Feed heavily imbalanced losses: physics dominates
        for i in range(10):
            weighter(i, {"ic": 0.01, "physics": 1.0, "total": 1.01})

        # Physics should have been downweighted, IC upweighted
        assert weights["physics"] < weights["ic"]

    def test_weights_clamped(self):
        weights = {"ic": 1.0, "physics": 1.0}
        weighter = AdaptiveLossWeighter(
            weights, rebalance_every=5, ratio_threshold=2.0,
            max_weight=10.0, min_weight=0.1, window=5,
        )

        # Extreme imbalance
        for i in range(10):
            weighter(i, {"ic": 1e-8, "physics": 100.0, "total": 100.0})

        assert weights["ic"] <= 10.0
        assert weights["physics"] >= 0.1

    def test_rebalance_count_tracked(self):
        weights = {"ic": 1.0, "physics": 1.0}
        weighter = AdaptiveLossWeighter(
            weights, rebalance_every=5, ratio_threshold=2.0, window=5,
        )

        for i in range(20):
            weighter(i, {"ic": 0.01, "physics": 10.0, "total": 10.01})

        assert weighter.rebalance_count > 0

    def test_integrates_with_trainer(self, tiny_model, cpu):
        weights = {"ic": 1.0, "physics": 1.0}
        weighter = AdaptiveLossWeighter(
            weights, rebalance_every=10, ratio_threshold=3.0, window=10,
        )
        trainer = PINNTrainer(tiny_model, device=cpu)

        t = torch.linspace(0, 1, 20).unsqueeze(1).requires_grad_(True)

        def physics_loss(model):
            u = model(t)
            du = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
            return torch.mean((du + u) ** 2)

        def ic_loss(model):
            return (model(torch.zeros(1, 1)) - 1.0) ** 2

        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        trainer.train(
            n_epochs=30, optimizer=optimizer,
            loss_functions={"ic": ic_loss, "physics": physics_loss},
            weights=weights, callbacks=[weighter], verbose=False,
        )

        # Should have run without errors; weights may have been adjusted
        assert len(trainer.loss_history) == 30


# ----------------------------------------------------------- evaluate_quality


class TestEvaluateQuality:
    def test_empty_history(self):
        result = evaluate_quality([])
        assert result["quality_score"] == 0.0

    def test_converged_run(self):
        # Simulate a run that converges quickly
        history = [{"total": 1.0 * (0.9**i)} for i in range(100)]
        # Add some that are below threshold
        history.extend([{"total": 1e-5}] * 50)

        result = evaluate_quality(history)
        assert result["effectiveness"] > 0.8
        assert result["efficiency"] > 0.0
        assert result["quality_score"] > 0.5
        assert result["convergence_epoch"] is not None

    def test_non_converged_run(self):
        history = [{"total": 1.0}] * 100
        result = evaluate_quality(history)
        assert result["efficiency"] == 0.0
        assert result["convergence_epoch"] is None

    def test_smooth_training_high_robustness(self):
        history = [{"total": 1.0 * (0.99**i)} for i in range(200)]
        result = evaluate_quality(history)
        assert result["robustness"] > 0.7

    def test_oscillating_training_low_robustness(self):
        history = [{"total": 1.0 + (-1) ** i * 0.8} for i in range(200)]
        result = evaluate_quality(history)
        assert result["robustness"] < 0.5

    def test_all_fields_present(self):
        history = [{"total": 0.5}, {"total": 0.1}]
        result = evaluate_quality(history)
        expected_keys = {
            "effectiveness", "efficiency", "robustness",
            "quality_score", "final_loss", "convergence_epoch",
        }
        assert set(result.keys()) == expected_keys

    def test_quality_score_bounded(self):
        history = [{"total": float(i)} for i in range(1, 101)]
        result = evaluate_quality(history)
        assert 0.0 <= result["quality_score"] <= 1.0
