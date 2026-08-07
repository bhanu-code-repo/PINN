# Poisson Equation

## Equation Type
Elliptic PDE: -laplacian(u) = f(x) in Omega, u = g on boundary
Special case: Laplace equation (f=0). Steady-state heat/diffusion.
Prototype for elliptic problems: no time dependence, pure boundary value problem.

## Key Findings
- Easiest class of PDEs for PINNs — the baseline benchmark
- Well-conditioned problem: solution is smooth (as smooth as f and the domain)
- Vanilla PINN achieves L2 error < 1e-5 on unit square with 5,000 collocation points
- Maximum principle: solution is bounded by boundary data — constrains the solution space
- Galerkin PINN (weak form) often outperforms strong-form PINN for Poisson
- High-contrast coefficients: -div(a(x)*grad(u)) = f with discontinuous a is significantly harder
- For Poisson on the unit square with f=1 and zero Dirichlet BCs: trivially solved in minutes
- In irregular geometry: boundary integral formulation can outperform domain collocation

## Recommended Architecture
- Layers: 3-4 hidden layers x 32-64 neurons — smaller networks are fine
- Activation: tanh (standard)
- Collocation points: 2,000-5,000 interior; 500 boundary
- Latin hypercube sampling (LHS) preferred over uniform for interior points
- Hard BC ansatz: u(x) = phi(x) * NN(x) where phi(x) is distance to boundary; exactly enforces u=0
- This is the ideal testbed for new ansatz methods — ground truth is cheaply available

## Known Failure Modes
- Almost none for smooth f and regular geometry — this problem "just works"
- Discontinuous f or a(x): gradient of solution has jump; network cannot represent discontinuous derivative
- Domain with re-entrant corners (L-shape, crack): solution has singularity u ~ r^(pi/angle); convergence is slow
- High-contrast coefficient: if a_max/a_min > 100, residual gradient is dominated by high-a region
- Very irregular domain: boundary representation must be accurate; discretization artifacts if boundary poorly sampled

## Techniques
- Baseline benchmark: use this to validate any new technique before applying to harder PDEs
- Hard BC ansatz: u = distance_to_boundary * NN; eliminates BC loss term entirely
- LHS sampling: always prefer to uniform for interior collocation in 2D+
- Weak form (Galerkin PINN): integrate by parts to reduce derivative order; improves accuracy
- For singular corners: use graded mesh / weighted sampling near singularity: rho(x) ~ |x - x_corner|^(-alpha)
- For high-contrast: weight residual by 1/a(x) to equalize loss contributions
- L-BFGS as final optimizer: converges to machine precision on Poisson problems

## References
- Lagaris, Likas, Fotiadis (1998) — Artificial neural networks for solving ODE/PDE boundary value problems
- Berg, Nystrom (2018) — A unified deep artificial neural network approach to PDE in complex geometries
- Kharazmi, Zhang, Karniadakis (2019) — Variational physics-informed neural networks for solving partial differential equations
- Sukumar, Srivastava (2022) — Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks
