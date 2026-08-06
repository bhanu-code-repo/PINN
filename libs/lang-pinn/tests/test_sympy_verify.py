"""Tests for SymPy-based PDE verification."""

from lang_pinn.schemas import PDESpec
from lang_pinn.sympy_verify import (
    _detect_max_derivative_order,
    _extract_variables_from_equation,
    verify_spec,
)


def _make_spec(**kwargs) -> PDESpec:
    defaults = {
        "name": "Test",
        "equation": "u_tt + u = 0",
        "independent_vars": ["t"],
        "dependent_var": "u",
        "order": 2,
        "spatial_dim": 0,
        "domain": {"t": (0.0, 1.0)},
    }
    defaults.update(kwargs)
    return PDESpec(**defaults)


class TestDerivativeOrderDetection:
    def test_first_order(self):
        assert _detect_max_derivative_order("u_t + u = 0", "u") == 1

    def test_second_order(self):
        assert _detect_max_derivative_order("u_tt + mu*u_t + k*u = 0", "u") == 2

    def test_mixed_orders(self):
        assert _detect_max_derivative_order("u_t + u*u_x = nu*u_xx", "u") == 2

    def test_no_derivatives(self):
        assert _detect_max_derivative_order("u = 0", "u") is None

    def test_different_dep_var(self):
        assert _detect_max_derivative_order("h_tt + h = 0", "h") == 2
        assert _detect_max_derivative_order("h_tt + h = 0", "u") is None


class TestVariableExtraction:
    def test_ode_single_var(self):
        vars = _extract_variables_from_equation("u_tt + u_t = 0", "u")
        assert vars == {"t"}

    def test_pde_two_vars(self):
        vars = _extract_variables_from_equation("u_t + u*u_x = nu*u_xx", "u")
        assert vars == {"t", "x"}

    def test_ignores_other_symbols(self):
        vars = _extract_variables_from_equation("u_t + k*u = 0", "u")
        assert vars == {"t"}


class TestVerifySpec:
    def test_valid_ode_no_issues(self):
        spec = _make_spec()
        issues = verify_spec(spec)
        assert len(issues) == 0

    def test_invalid_domain_bounds(self):
        spec = _make_spec(domain={"t": (1.0, 0.0)})
        issues = verify_spec(spec)
        assert any("lower bound" in i for i in issues)

    def test_missing_domain_for_var(self):
        spec = _make_spec(independent_vars=["t", "x"])
        issues = verify_spec(spec)
        assert any("no domain bounds" in i for i in issues)

    def test_order_mismatch(self):
        spec = _make_spec(equation="u_t + u = 0", order=2)
        issues = verify_spec(spec)
        assert any("order" in i.lower() for i in issues)

    def test_order_match(self):
        spec = _make_spec(equation="u_tt + u = 0", order=2)
        issues = verify_spec(spec)
        order_issues = [i for i in issues if "order" in i.lower()]
        assert len(order_issues) == 0

    def test_spatial_dim_mismatch(self):
        spec = _make_spec(
            equation="u_t + u_x = 0",
            independent_vars=["x", "t"],
            spatial_dim=0,
            domain={"x": (0, 1), "t": (0, 1)},
            order=1,
        )
        issues = verify_spec(spec)
        assert any("spatial_dim" in i for i in issues)

    def test_pde_var_not_declared(self):
        spec = _make_spec(
            equation="u_t + u_x = 0",
            independent_vars=["t"],  # missing x
            order=1,
        )
        issues = verify_spec(spec)
        assert any("'x' appears in equation" in i for i in issues)

    def test_sympy_parseable_equation(self):
        spec = _make_spec(equation="u_tt + mu*u_t + k*u = 0")
        issues = verify_spec(spec)
        sympy_issues = [i for i in issues if "parseable" in i.lower()]
        assert len(sympy_issues) == 0

    def test_valid_pde_passes(self):
        spec = PDESpec(
            name="Burgers",
            equation="u_t + u*u_x = nu*u_xx",
            independent_vars=["x", "t"],
            dependent_var="u",
            order=2,
            spatial_dim=1,
            domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
            parameters={"nu": 0.01},
            is_linear=False,
            has_sharp_gradients=True,
        )
        issues = verify_spec(spec)
        assert len(issues) == 0
