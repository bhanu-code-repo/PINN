"""Tests for PDESpec and ArchitectureRec data structures."""

from lang_pinn.schemas import ArchitectureRec, PDESpec


class TestPDESpec:
    def test_minimal_ode(self):
        spec = PDESpec(
            name="Simple ODE",
            equation="u_t + u = 0",
            independent_vars=["t"],
            dependent_var="u",
            order=1,
            spatial_dim=0,
            domain={"t": (0.0, 1.0)},
        )
        assert spec.order == 1
        assert spec.spatial_dim == 0
        assert spec.is_linear is True
        assert spec.output_dim == 1
        assert spec.initial_conditions == []
        assert spec.parameters == {}

    def test_pde_with_all_fields(self):
        spec = PDESpec(
            name="Burgers",
            equation="u_t + u*u_x = nu*u_xx",
            independent_vars=["x", "t"],
            dependent_var="u",
            order=2,
            spatial_dim=1,
            domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
            initial_conditions=["u(x,0) = -sin(pi*x)"],
            boundary_conditions=["u(-1,t) = 0", "u(1,t) = 0"],
            parameters={"nu": 0.01},
            is_linear=False,
            has_sharp_gradients=True,
        )
        assert spec.is_linear is False
        assert spec.has_sharp_gradients is True
        assert len(spec.independent_vars) == 2
        assert spec.parameters["nu"] == 0.01

    def test_complex_output(self):
        spec = PDESpec(
            name="Schrodinger",
            equation="i*h_t + 0.5*h_xx + |h|^2*h = 0",
            independent_vars=["x", "t"],
            dependent_var="h",
            order=2,
            spatial_dim=1,
            domain={"x": (-5.0, 5.0), "t": (0.0, 3.14)},
            output_dim=2,
            has_periodic_bc=True,
        )
        assert spec.output_dim == 2
        assert spec.has_periodic_bc is True


class TestArchitectureRec:
    def test_defaults(self):
        rec = ArchitectureRec(input_dim=1, output_dim=1,
                              hidden_layers=3, hidden_neurons=32)
        assert rec.activation == "tanh"
        assert rec.learning_rate == 1e-3
        assert rec.use_ansatz is False
        assert rec.ansatz_type is None

    def test_with_ansatz(self):
        rec = ArchitectureRec(
            input_dim=1, output_dim=1,
            hidden_layers=3, hidden_neurons=64,
            use_ansatz=True, ansatz_type="sinusoidal",
        )
        assert rec.use_ansatz is True
        assert rec.ansatz_type == "sinusoidal"
