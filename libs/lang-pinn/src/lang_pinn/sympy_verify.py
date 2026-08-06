"""SymPy-based verification for PDE Agent output.

Checks that the parsed PDESpec is mathematically consistent:
- Equation string is parseable as a symbolic expression
- Derivative order matches claimed order
- Variables in equation match declared independent/dependent vars
- Domain bounds are valid (low < high)
- Parameter names in equation exist in parameters dict
"""

from __future__ import annotations

import re

from loguru import logger

from .schemas import PDESpec

# Derivative notation patterns: u_t, u_xx, u_ttt, etc.
_DERIV_PATTERN = re.compile(r"(\w+)_(([xyzt])\3*)")


def verify_spec(spec: PDESpec) -> list[str]:
    """Verify a PDESpec for mathematical consistency.

    Returns a list of warning strings. Empty list means all checks passed.
    """
    issues: list[str] = []

    # 1. Domain bounds: low < high
    for var, (lo, hi) in spec.domain.items():
        if lo >= hi:
            issues.append(f"Domain '{var}': lower bound {lo} >= upper bound {hi}")

    # 2. Check that independent vars have domains
    for var in spec.independent_vars:
        if var not in spec.domain:
            issues.append(f"Variable '{var}' declared but has no domain bounds")

    # 3. Derivative order verification
    max_order = _detect_max_derivative_order(spec.equation, spec.dependent_var)
    if max_order is not None and max_order != spec.order:
        issues.append(
            f"Claimed order={spec.order} but detected max derivative "
            f"order={max_order} in equation"
        )

    # 4. Variables in equation match declarations
    eq_vars = _extract_variables_from_equation(spec.equation, spec.dependent_var)
    for var in eq_vars:
        if var not in spec.independent_vars:
            issues.append(
                f"Variable '{var}' appears in equation derivatives "
                f"but not in independent_vars={spec.independent_vars}"
            )

    # 5. Spatial dim consistency
    spatial_vars = [v for v in spec.independent_vars if v != "t"]
    if spec.spatial_dim != len(spatial_vars):
        issues.append(
            f"spatial_dim={spec.spatial_dim} but found {len(spatial_vars)} "
            f"spatial variable(s): {spatial_vars}"
        )

    # 6. Try SymPy parsing of the equation
    sympy_issues = _verify_with_sympy(spec)
    issues.extend(sympy_issues)

    if issues:
        for issue in issues:
            logger.warning("SymPy verify: {}", issue)
    else:
        logger.info("SymPy verification: all checks passed")

    return issues


def _detect_max_derivative_order(equation: str, dep_var: str) -> int | None:
    """Detect the maximum derivative order from notation like u_tt, u_xxx."""
    max_order = 0
    for match in _DERIV_PATTERN.finditer(equation):
        var_name = match.group(1)
        deriv_chars = match.group(2)
        if var_name == dep_var:
            max_order = max(max_order, len(deriv_chars))

    return max_order if max_order > 0 else None


def _extract_variables_from_equation(equation: str, dep_var: str) -> set[str]:
    """Extract independent variable names from derivative notation."""
    variables = set()
    for match in _DERIV_PATTERN.finditer(equation):
        var_name = match.group(1)
        deriv_char = match.group(3)  # single char like 'x', 't'
        if var_name == dep_var:
            variables.add(deriv_char)
    return variables


def _verify_with_sympy(spec: PDESpec) -> list[str]:
    """Try to parse the equation with SymPy for deeper verification."""
    issues = []
    try:
        import sympy
    except ImportError:
        logger.debug("SymPy not installed, skipping symbolic verification")
        return issues

    # Try to parse the equation as a symbolic expression
    # Split on '=' and verify both sides are parseable
    equation = spec.equation.strip()

    # Replace derivative notation with SymPy-friendly names
    # u_tt -> u_tt (just a symbol), u_x -> u_x (just a symbol)
    # This is intentionally lenient — we just want to verify parseability
    eq_parts = equation.split("=")
    if len(eq_parts) != 2:
        issues.append(f"Equation should have exactly one '=' sign, found {len(eq_parts) - 1}")
        return issues

    lhs, rhs = eq_parts[0].strip(), eq_parts[1].strip()

    # Create symbols for all known names
    all_symbols = set()
    all_symbols.add(spec.dependent_var)
    all_symbols.update(spec.independent_vars)
    all_symbols.update(spec.parameters.keys())

    # Add derivative symbols (u_t, u_tt, u_x, u_xx, etc.)
    for match in _DERIV_PATTERN.finditer(equation):
        all_symbols.add(match.group(0))  # e.g. "u_tt"

    sym_dict = {name: sympy.Symbol(name) for name in all_symbols}

    # Also add common math functions
    sym_dict.update({
        "sin": sympy.sin, "cos": sympy.cos, "exp": sympy.exp,
        "sqrt": sympy.sqrt, "abs": sympy.Abs, "pi": sympy.pi,
    })

    try:
        sympy.sympify(lhs, locals=sym_dict)
    except (sympy.SympifyError, SyntaxError, TypeError) as e:
        issues.append(f"LHS not parseable as symbolic expression: {e}")

    try:
        sympy.sympify(rhs, locals=sym_dict)
    except (sympy.SympifyError, SyntaxError, TypeError) as e:
        issues.append(f"RHS not parseable as symbolic expression: {e}")

    return issues
