"""Tests for Residual-based Adaptive Refinement (RAR)."""

import torch
from pinn import PINN, PINNTrainer, adaptive_train, select_rar_points

# ------------------------------------------------------------------ fixtures


def _simple_residual_fn(model, points):
    """Residual = model output magnitude (simple proxy for testing)."""
    with torch.no_grad():
        return model(points).squeeze(-1).abs()


def _multi_component_residual_fn(model, points):
    """Residual with 2 components (tests L2-norm branch)."""
    with torch.no_grad():
        out = model(points).squeeze(-1)
        return torch.stack([out, out * 2], dim=-1)


# --------------------------------------------------------- select_rar_points


class TestSelectRarPoints:
    def test_returns_correct_count(self, tiny_model):
        candidates = torch.randn(200, 1)
        selected = select_rar_points(tiny_model, candidates, _simple_residual_fn, n_select=30)
        assert selected.shape == (30, 1)

    def test_selects_highest_residuals(self):
        """Points with known high residuals should be selected."""
        model = PINN(input_dim=1, hidden_layers=1, hidden_neurons=4)
        # Create candidates where we know the model output pattern
        candidates = torch.linspace(-3, 3, 100).unsqueeze(1)

        # Get all residuals to find ground truth top-k
        with torch.no_grad():
            all_res = model(candidates).squeeze(-1).abs()
        _, expected_idx = torch.topk(all_res, 10)
        expected_points = candidates[expected_idx]

        selected = select_rar_points(model, candidates, _simple_residual_fn, n_select=10)

        # Selected points should match the ground truth top-k
        assert torch.allclose(
            selected.sort(dim=0).values,
            expected_points.sort(dim=0).values,
        )

    def test_n_select_larger_than_candidates(self, tiny_model):
        """Requesting more points than available should return all candidates."""
        candidates = torch.randn(5, 1)
        selected = select_rar_points(tiny_model, candidates, _simple_residual_fn, n_select=100)
        assert selected.shape == (5, 1)

    def test_multi_component_residual(self, tiny_model):
        """Multi-component residuals should be L2-normed before selection."""
        candidates = torch.randn(50, 1)
        selected = select_rar_points(
            tiny_model, candidates, _multi_component_residual_fn, n_select=10
        )
        assert selected.shape == (10, 1)

    def test_selected_points_are_detached(self, tiny_model):
        candidates = torch.randn(50, 1, requires_grad=True)
        selected = select_rar_points(tiny_model, candidates, _simple_residual_fn, n_select=5)
        assert not selected.requires_grad

    def test_multidim_input(self):
        """Works with 2D input (e.g. x, t)."""
        model = PINN(input_dim=2, hidden_layers=1, hidden_neurons=4)
        candidates = torch.randn(100, 2)

        def res_fn(m, pts):
            with torch.no_grad():
                return m(pts).squeeze(-1).abs()

        selected = select_rar_points(model, candidates, res_fn, n_select=15)
        assert selected.shape == (15, 2)


# ----------------------------------------------------------- adaptive_train


class TestAdaptiveTrain:
    def _setup(self):
        """Create a minimal ODE setup: u' = -u, u(0) = 1."""
        model = PINN(input_dim=1, hidden_layers=2, hidden_neurons=16)
        device = torch.device("cpu")
        trainer = PINNTrainer(model, device=device)

        t_ic = torch.zeros(1, 1, device=device)

        def build_losses(physics_points):
            t_phys = physics_points.clone().requires_grad_(True)

            def ic_loss(m):
                return (m(t_ic) - 1.0) ** 2

            def physics_loss(m):
                u = m(t_phys)
                du = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
                return torch.mean((du + u) ** 2)

            return {"ic": ic_loss, "physics": physics_loss}

        def residual_fn(m, pts):
            pts = pts.clone().requires_grad_(True)
            u = m(pts)
            du = torch.autograd.grad(u, pts, torch.ones_like(u), create_graph=True)[0]
            return (du + u).squeeze(-1)

        def candidate_sampler(n):
            return torch.rand(n, 1, device=device) * 3.0

        initial_points = torch.rand(50, 1, device=device) * 3.0

        def optimizer_fn(m):
            return torch.optim.Adam(m.parameters(), lr=1e-3)

        return trainer, build_losses, residual_fn, candidate_sampler, initial_points, optimizer_fn

    def test_points_grow_each_phase(self):
        trainer, build_losses, res_fn, sampler, init_pts, opt_fn = self._setup()
        result = adaptive_train(
            trainer=trainer,
            build_losses=build_losses,
            residual_fn=res_fn,
            candidate_sampler=sampler,
            initial_points=init_pts,
            optimizer_fn=opt_fn,
            n_phases=3,
            epochs_per_phase=10,
            n_candidates=100,
            n_select=20,
            verbose=False,
        )
        # Points should grow: 50 -> 70 -> 90 (no addition after last phase)
        assert result["points_per_phase"] == [50, 70, 90]
        assert result["points"].shape[0] == 90

    def test_loss_history_spans_all_phases(self):
        trainer, build_losses, res_fn, sampler, init_pts, opt_fn = self._setup()
        result = adaptive_train(
            trainer=trainer,
            build_losses=build_losses,
            residual_fn=res_fn,
            candidate_sampler=sampler,
            initial_points=init_pts,
            optimizer_fn=opt_fn,
            n_phases=2,
            epochs_per_phase=15,
            n_candidates=50,
            n_select=10,
            verbose=False,
        )
        # Total epochs = 2 phases * 15 epochs = 30
        assert len(result["loss_history"]) == 30

    def test_single_phase_no_refinement(self):
        """With n_phases=1, no refinement happens — just normal training."""
        trainer, build_losses, res_fn, sampler, init_pts, opt_fn = self._setup()
        result = adaptive_train(
            trainer=trainer,
            build_losses=build_losses,
            residual_fn=res_fn,
            candidate_sampler=sampler,
            initial_points=init_pts,
            optimizer_fn=opt_fn,
            n_phases=1,
            epochs_per_phase=10,
            n_candidates=50,
            n_select=10,
            verbose=False,
        )
        assert result["points_per_phase"] == [50]
        assert result["points"].shape[0] == 50

    def test_loss_decreases(self):
        """Verify loss drops over multi-phase RAR training."""
        trainer, build_losses, res_fn, sampler, init_pts, opt_fn = self._setup()
        result = adaptive_train(
            trainer=trainer,
            build_losses=build_losses,
            residual_fn=res_fn,
            candidate_sampler=sampler,
            initial_points=init_pts,
            optimizer_fn=opt_fn,
            n_phases=2,
            epochs_per_phase=200,
            n_candidates=200,
            n_select=25,
            verbose=False,
        )
        history = result["loss_history"]
        assert history[-1]["total"] < history[0]["total"]

    def test_weights_forwarded(self):
        """Weights are passed through to the trainer."""
        trainer, build_losses, res_fn, sampler, init_pts, opt_fn = self._setup()
        result = adaptive_train(
            trainer=trainer,
            build_losses=build_losses,
            residual_fn=res_fn,
            candidate_sampler=sampler,
            initial_points=init_pts,
            optimizer_fn=opt_fn,
            n_phases=1,
            epochs_per_phase=5,
            n_candidates=50,
            n_select=10,
            weights={"ic": 10.0, "physics": 1.0},
            verbose=False,
        )
        assert len(result["loss_history"]) == 5
