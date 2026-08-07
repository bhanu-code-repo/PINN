# Navier-Stokes 2D

## Equation Type
Incompressible 2D NS: u_t + u*u_x + v*u_y = -p_x + nu*(u_xx+u_yy)
                       v_t + u*v_x + v*v_y = -p_y + nu*(v_xx+v_yy)
                       u_x + v_y = 0 (continuity)
Four unknowns: (u, v, p) with continuity as a hard constraint.

## Key Findings
- Pressure p is not uniquely defined (gauge invariance: p + const is also a solution)
- Must fix pressure gauge: pin p at one point, or use mean-zero constraint
- Streamfunction-vorticity formulation eliminates pressure and automatically satisfies continuity
- Streamfunction psi: u = psi_y, v = -psi_x — reduces 3 unknowns to 1, continuity is exact
- For Re < 100 (laminar): PINNs perform well with ~50,000 collocation points
- For Re > 1000: turbulent regime, PINNs struggle without additional physics or data
- Data-driven NS (inverse problem to find nu from velocity field) is well-solved by PINNs
- Cavity flow (lid-driven) at Re=100: L2 error ~1e-3 with standard PINN

## Recommended Architecture
- Layers: 8 hidden layers x 64 neurons
- Activation: tanh
- Output heads: [u, v, p] (3 outputs) — or [psi] (1 output) for streamfunction formulation
- Collocation points: 50,000 interior for Re=100; 200,000 for Re=1000
- Boundary points: 1,000 per boundary segment
- Temporal collocation: if unsteady, use 100,000+ space-time points
- Pressure gauge fix: add constraint loss L_gauge = (p(x0,y0) - p_ref)^2 with large weight

## Known Failure Modes
- Pressure non-uniqueness: without gauge fixing, p drifts; continuity residual appears converged but p is wrong
- Continuity soft constraint: often under-satisfied; enforce with lambda_continuity = 10-100
- High Reynolds number: insufficient collocation near boundary layers; exponential boundary layer profile
- Pressure-velocity decoupling: optimizer minimizes momentum and continuity independently, can get locally feasible but globally inconsistent solution
- Stagnation point singularity: u=v=0 at corners with conflicting BCs causes large residuals

## Techniques
- Streamfunction formulation: psi -> (u,v) by differentiation, eliminates continuity and pressure gauge
- Vorticity-streamfunction: omega = v_x - u_y, psi_xx + psi_yy = -omega (Poisson)
- Pressure gauge: pin p at domain center or add mean(p)=0 loss
- Adaptive sampling near walls: boundary layer requires fine resolution (y+ < 1 equivalent)
- For inverse NS: observe (u,v) at sparse points, treat nu as trainable parameter
- Domain decomposition for complex geometries; interface conditions via XPINNs
- Physics-informed PointNet for irregular geometries

## References
- Raissi, Yazdani, Karniadakis (2020) — Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations, Science
- Jin, Cai, Li, Karniadakis (2021) — NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations
- Cai, Mao, Wang, Yin, Karniadakis (2021) — Physics-informed neural networks for heat transfer problems
