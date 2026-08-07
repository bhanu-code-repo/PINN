# Schrodinger Equation

## Equation Type
Complex-valued PDE: i*u_t + 0.5*u_xx + |u|^2 * u = 0 (nonlinear Schrodinger, NLS)
Or linear: i*hbar*psi_t = -hbar^2/(2m) * psi_xx + V(x)*psi
Solution is complex-valued; standard real-valued NNs require decomposition.

## Key Findings
- Must decompose: u = v + iw, then solve coupled real system for (v, w)
- Periodic boundary conditions are standard (soliton/breather solutions on periodic domain)
- NLS supports soliton and breather solutions — these are well-captured by PINNs
- Raissi et al. (2019) demonstrated PINN on NLS with periodic BCs, L2 error ~1e-3
- Conservation laws (mass, momentum, energy) provide valuable auxiliary loss terms
- Phase-sensitive: small errors in phase accumulate over time — more critical than amplitude error
- Linear Schrodinger with smooth V(x): relatively easy, similar to heat equation
- NLS with strong nonlinearity (|u|^2 term dominates): harder, needs more collocation near peaks

## Recommended Architecture
- Layers: 5 hidden layers x 100 neurons (Raissi benchmark)
- Activation: tanh
- Output: 2 neurons [v(x,t), w(x,t)] representing real and imaginary parts
- Collocation points: 20,000 interior (x in [-5,5], t in [0,pi/2]) for standard soliton benchmark
- Boundary points: 50 per time snapshot for periodic BC enforcement
- IC points: 50 spatial points for u(x,0)
- Periodic BC: enforce v(-L,t) = v(L,t) and v_x(-L,t) = v_x(L,t), same for w

## Known Failure Modes
- Phase error accumulation: network amplitude may be correct but phase drifts — undetectable from |u| alone
- Periodic BC violation: soft enforcement often insufficient; consider hard ansatz with Fourier basis
- Breather solutions: near-periodic in time with slow modulation — network confuses periods
- Conservation law drift: mass integral int(|u|^2 dx) should be constant; soft constraint helps
- Complex-valued network (native): if using complex weights, initialization and optimization are non-standard

## Techniques
- Real/imaginary decomposition: always split into two real networks or two output heads
- Conservation law loss: add L_mass = ||int_x(v^2+w^2)dx - M_0||^2 as auxiliary term
- Periodic BC hard constraint: use Fourier series ansatz u(x,t) = sum_k a_k(t)*exp(ikx)
- IC: exact soliton initial condition u(x,0) = 2*sech(x) for NLS benchmark
- Gradient clipping: max_norm=1.0 prevents explosion from |u|^2 nonlinearity
- Importance sampling near soliton peak (x near 0) helps accuracy

## References
- Raissi, Perdikaris, Karniadakis (2019) — Physics-informed neural networks, JCP
- Pu, Li, Chen (2021) — Soliton, breather and rogue wave solutions for solving the nonlinear Schrodinger equation using a deep learning method with time-stepping
- Jagtap, Kawaguchi, Karniadakis (2020) — Adaptive activation functions accelerate convergence in deep and physics-informed neural networks
