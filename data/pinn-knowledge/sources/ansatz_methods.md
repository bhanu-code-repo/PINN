# Ansatz Methods (Hard Constraint Enforcement)

## Equation Type
Not a PDE — a construction strategy for enforcing BCs/ICs exactly within the network architecture.
Soft constraints use loss penalties; hard constraints embed conditions into the network structure.

## Key Findings
- Hard constraints guarantee zero BC/IC error regardless of training — soft constraints only approximate
- For Dirichlet BC (u=g on boundary): hard ansatz u(x) = g(x) + phi(x)*NN(x), where phi=0 on boundary
- phi(x) can be: signed distance function, distance to boundary, product of boundary-vanishing terms
- For IC at t=0: u(x,t) = u0(x) + t*NN(x,t) — exactly enforces u(x,0) = u0(x)
- Hard constraints remove BC/IC from loss, freeing optimizer to focus entirely on PDE residual
- Performance gain: typically 2-10x fewer training steps; L2 error often 1-2 orders of magnitude lower
- Lagaris et al. (1998) first proposed this; modern versions use learned distance functions
- Sinusoidal ansatz for high-frequency problems: u(x) = sum_k [a_k * sin(k*pi*x/L)] for eigenfunction problems

## Recommended Architecture
Hard ansatz construction:
- Dirichlet BC on [0,1]: u(x) = x*(1-x)*NN(x) + (1-x)*g(0) + x*g(1)
- Dirichlet BC on [a,b]: u(x) = (x-a)*(b-x)/((b-a)^2)*NN(x) + boundary_interpolation
- IC at t=0: u(x,t) = u0(x) + t*NN(x,t) — zero extra parameters, exact IC
- Mixed BC: combine distance functions: phi(x) = product of (x-x_i) for each Dirichlet boundary
- Periodic BC: u(x) = NN(sin(pi*x/L), cos(pi*x/L)) — inputs are automatically periodic
- Signed distance function: phi(x) = dist(x, boundary); works for arbitrary geometries

Sinusoidal ansatz for eigenvalue problems:
- Helmholtz/wave: u(x) = sum_{k=1}^{K} c_k * sin(k*pi*x/L), train c_k
- Hybrid: u(x) = sin(k*pi*x/L) * NN(x) — forces leading frequency structure

## Known Failure Modes
- Distance function accuracy: for complex geometries, phi may not be smooth or accurate
- t=0 IC ansatz limitation: u0(x) must be known analytically; cannot use for data-driven IC
- Over-constrained: hard ansatz for Robin/Neumann BCs requires differentiation of phi — error-prone
- Periodic BC: sin/cos embedding works for regular domains; complex geometry periodic BCs harder
- Hard IC + long time: t*NN grows unbounded for large t, causing optimization difficulties
- Wrong phi: if phi does not vanish exactly on boundary, constraint is not exactly enforced

## Techniques
- Distance-to-boundary: use approximate signed distance (|x| - 1 for [-1,1] domain) as phi
- Exact IC: always prefer hard IC constraint u(x,t) = u0(x) + t*NN when u0 is analytically known
- Hard periodic BC: embed (sin(2pi*x/L), cos(2pi*x/L)) as inputs instead of raw x
- R3 ansatz (Sukumar & Srivastava 2022): implicit boundary representation via R-functions
- Combine hard IC with soft residual: eliminate IC loss, let optimizer focus on physics
- Hard Dirichlet + soft Neumann: enforce essential BCs hard, natural BCs soft
- Ansatz for symmetry: if solution is even, use u(x) = NN(x^2) to enforce symmetry exactly

## References
- Lagaris, Likas, Fotiadis (1998) — Artificial neural networks for solving ODE/PDE boundary value problems, IEEE TNN
- Sukumar, Srivastava (2022) — Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks, CMAME
- Dong, Ni (2021) — A method for representing periodic functions and enforcing exactly periodic boundary conditions with deep neural networks
- McFall, Mahan (2009) — Artificial neural network method for solution of boundary value problems with exact satisfaction of arbitrary boundary conditions
