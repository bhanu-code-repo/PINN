# Periodic Boundary Conditions

## Equation Type
Applies to any PDE on a periodic domain: u(x + L, t) = u(x, t) for all x, t.
Standard in: wave propagation, soliton problems, atmospheric models, plasma physics.
Requires matching function values AND all derivatives at the periodic boundary.

## Key Findings
- Soft periodic BC enforcement is almost always insufficient for smooth solutions
- Naive soft enforcement: L_periodic = ||u(-L,t) - u(L,t)||^2 — only enforces value, not derivatives
- For C^k PDEs, need to enforce u^(j)(-L,t) = u^(j)(L,t) for j=0,1,...,k-1
- Hard enforcement via Fourier embedding is exact and has zero extra training cost
- Schrodinger NLS benchmark uses periodic x in [-5, 5]; enforcing all 3 derivative levels needed
- KdV (3rd order): need to enforce u, u_x, u_xx at boundary — soft fails reliably without all three
- Fourier-encoded inputs (sin, cos) are the gold standard for periodic problems
- When period L is known: always use hard enforcement; only use soft if L is unknown (rare)

## Recommended Architecture
Hard periodic BC (preferred for known L):
- Replace input x with [sin(2*pi*x/L), cos(2*pi*x/L)] — network is periodic by construction
- For higher harmonics: include [sin(2*pi*k*x/L), cos(2*pi*k*x/L)] for k=1,...,K (K=5-10 typical)
- This is the same as Fourier feature embedding but driven by periodicity requirement, not spectral bias
- Network: NN([sin(2pi*x/L), cos(2pi*x/L), t]) — input dimension is 3 for 1D+time, not 2

Soft periodic BC (fallback):
- Enforce at N_periodic boundary points: u(-L/2,t_i) = u(L/2,t_i) for i=1..N_periodic
- For 2nd-order PDE: also enforce u_x(-L/2,t_i) = u_x(L/2,t_i)
- Weight: lambda_periodic = 100 * lambda_residual (needs heavy weighting)
- N_periodic: 100-500 time points

## Known Failure Modes
- Derivative mismatch: function values match but u_x is discontinuous at boundary — solution has kink
  This is detectable by plotting u_x near boundary; often invisible from u alone
- Under-weighted soft BC: optimizer learns near-periodic solution that satisfies PDE but has boundary seam
- High-frequency periodic solution: if solution has k periods in domain, need k harmonics in embedding
- Wrong period: if L is estimated incorrectly, hard embedding fails; use L slightly larger than true domain
- Spatial derivative through embedding: d/dx[sin(2pi*x/L)] = (2pi/L)*cos(2pi*x/L) — correct, but must chain-rule

## Techniques
- Fourier embedding for periodic BCs: input_periodic = stack([sin(2pi*k*x/L) for k in 1..K], [cos(2pi*k*x/L) for k in 1..K])
  K=5 usually sufficient; K=10 for solutions with many harmonics
- Verify periodicity during evaluation: check max(|u(x_left,t) - u(x_right,t)|) on test grid
- Soft BC diagnostic: separately log L_periodic_0 (value), L_periodic_1 (first derivative), etc.
- For 2D periodic: use [sin(2pi*x/Lx), cos(2pi*x/Lx), sin(2pi*y/Ly), cos(2pi*y/Ly)] — extends naturally
- Anti-periodic: u(x+L) = -u(x) — use [sin((2k+1)*pi*x/L), cos((2k+1)*pi*x/L)] for k=0,1,...
- Data augmentation: reflect periodic domain; add points at x+L and x-L with same target values
- Conservation law check: int_domain(u dx) should be exactly the same at all t for conservation laws

## References
- Raissi, Perdikaris, Karniadakis (2019) — Physics-informed neural networks, JCP (NLS with periodic BCs)
- Dong, Ni (2021) — A method for representing periodic functions and enforcing exactly periodic boundary conditions with deep neural networks
- Tancik, Srinivasan, Mildenhall et al. (2020) — Fourier features let networks learn high frequency functions in low dimensional domains, NeurIPS
- Pu, Li, Chen (2021) — Solving localized wave solutions of the derivative nonlinear Schrodinger equation using an improved PINN method
