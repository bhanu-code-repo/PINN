# Korteweg-de Vries (KdV) Equation

## Equation Type
Nonlinear dispersive PDE: u_t + 6*u*u_x + u_xxx = 0
Models shallow water waves. Supports soliton solutions that travel without distortion.
Third-order spatial derivative (u_xxx) requires C^2 continuity — cannot use ReLU.

## Key Findings
- KdV is integrable: exact soliton solutions known analytically — useful for benchmarking
- Single soliton: u(x,t) = (c/2)*sech^2(sqrt(c)/2 * (x - c*t)) for speed c
- Two-soliton collision: solitons pass through each other, emerge unchanged — hard for PINNs
- Periodic boundary conditions on [-L, L] are standard; L must be large enough to avoid edge effects
- Dispersive term u_xxx requires computing 3rd-order derivative — computationally expensive in AD
- Conservation laws: mass (int u dx), momentum (int u^2 dx), energy (int u^3 - 0.5*u_x^2 dx)
- For single soliton on [−10, 10], t in [0, 1]: PINNs achieve L2 error ~5e-3 with 20k points
- Two-soliton interaction: error typically 5-10x higher than single soliton

## Recommended Architecture
- Layers: 6 hidden layers x 128 neurons
- Activation: tanh (mandatory — u_xxx requires 3 smooth derivatives)
- No ReLU, no leaky ReLU, no ELU beyond 3rd derivative
- Collocation points: 20,000 interior; 1,000 per periodic boundary; 500 IC points
- Domain: x in [-20, 20] to prevent soliton reaching boundary during integration
- IC: exact soliton formula; dense IC sampling near soliton peak (x near 0)
- Third-order derivative: verify AD implementation produces correct u_xxx; numerical check against FD

## Known Failure Modes
- Third-order derivative: u_xxx computed via AD is expensive and can have floating-point error; verify
- Soliton drift: network learns qualitatively correct soliton shape but wrong speed c; speed error accumulates
- Periodic BC for two solitons: after collision, solitons may be near boundaries; wrapping creates artifacts
- Phase shift: KdV two-soliton has a phase shift after collision; network may learn wrong phase
- Conservation law violation: each loss minimization step may violate conservation; add auxiliary losses
- Long-time integration (t > 5): soliton position error grows linearly; time-marching recommended

## Techniques
- Conservation law auxiliary losses: L_mass, L_momentum, L_energy added with weight ~0.01
- Dense IC sampling near soliton peak: use 1000 points in [-2, 2] and 100 points in rest of domain
- Exact IC: always use analytical soliton formula, not an approximation
- Periodic BC: enforce u(-L,t) = u(L,t), u_x(-L,t) = u_x(L,t), u_xx(-L,t) = u_xx(L,t) (3rd-order PDE)
- Time-marching for long-time integration: windows of size 0.5 time units
- Gradient clipping: u_xxx can produce large gradients; clip at norm=10
- For two-soliton: use higher-speed soliton as additional IC reference to enforce collision timing

## References
- Raissi, Perdikaris, Karniadakis (2019) — Physics-informed neural networks, JCP
- Pu, Chen (2022) — Physics-informed neural networks for solving forward and inverse problems in complex beam systems
- Li, Li, Ying, Cai, Tong (2022) — A semigroup method for high dimensional elliptic PDEs and eigenvalue problems based on neural networks
- Breen, Foley, Boekholt, Zwart (2020) — Newton vs the machine: Solving the chaotic three-body problem using deep neural networks
