# High-Frequency ODEs and Stiff Oscillators

## Equation Type
Damped harmonic oscillator: u'' + 2*zeta*omega*u' + omega^2*u = f(t)
High-frequency limit: omega >> 1 (many oscillations in [0, T])
Stiff limit: zeta << 1 (underdamped, solution is cos/sin with slowly decaying envelope)

## Key Findings
- For omega > 10, vanilla PINN with tanh fails completely — spectral bias prevents learning oscillations
- For omega = 50, standard PINN error > 1e-0; sinusoidal ansatz reduces to ~1e-3
- Sinusoidal ansatz is mandatory: u(t) = A(t)*cos(omega*t) + B(t)*sin(omega*t) where A,B are NNs
- This splits the problem: fast oscillation is handled by exact analytic structure, NN learns slow envelope
- Underdamped (0 < zeta < 1): envelope decays as exp(-zeta*omega*t) — this is smooth, NN handles it
- Overdamped (zeta > 1): no oscillation, two real modes — tanh PINN works fine
- Driven systems: f(t) = cos(omega_d * t) — resonance if omega_d near omega, PINN needs both freq
- Van der Pol oscillator (nonlinear): strongly oscillatory, requires special handling for mu >> 1

## Recommended Architecture
For high-frequency linear oscillator (omega > 10):
- Sinusoidal ansatz: u(t) = A(t)*cos(omega*t) + B(t)*sin(omega*t)
- A(t) and B(t): 2-3 layer x 32 neuron tanh networks — smooth functions, easy to learn
- IC: u(0) = A(0)*1 + B(0)*0 = A(0), u'(0) = A'(0) + omega*B(0) — two IC conditions in (A,B)
- Collocation: 5,000 points in t in [0, T], uniform (envelope is smooth, no RAR needed)
- omega as network input: if parametric in omega, embed log(omega) as input

For general stiff ODEs:
- Identify stiff and non-stiff components (spectral decomposition if linear)
- Handle stiff part analytically; use NN for slow component only
- SIREN: sin activation useful for moderate omega (10-50); exact ansatz better for omega > 50

## Known Failure Modes
- No ansatz for high omega: tanh network cannot represent cos(50*t) — fundamentally wrong architecture choice
- Ansatz with wrong frequency: if omega is approximate, ansatz oscillates at wrong rate, residuals are huge
- Phase error: even with correct ansatz, A(t) and B(t) may have small errors that cause accumulating phase error
- Van der Pol stiffness: for mu > 100, stiff limit ODE — must use time-marching with very small windows
- IVP vs BVP mismatch: shooting methods for BVP with high omega are unstable; PINN BVP formulation is better

## Techniques
- Exact sinusoidal ansatz: u(t) = exp(-zeta*omega*t) * [A(t)*cos(omega_d*t) + B(t)*sin(omega_d*t)]
  where omega_d = omega*sqrt(1-zeta^2) (damped natural frequency)
- A(t) and B(t) represent slow residual deviations from analytic form — should be near-constant for linear
- For nonlinear oscillators: split u = u_linear + delta_u, solve for delta_u with PINN
- Multi-scale time embedding: provide [t, t*omega, t*omega^2] as inputs; lets network attend to different scales
- Normalization: normalize t to [0, 2*pi] per period — omega-independent problem in normalized coordinates
- Dominant balance: use asymptotic approximation as initial guess / warm start for PINN
- Fourier features: embed t -> [cos(k*omega*t), sin(k*omega*t)] for k=1..5 — explicit frequency injection

## References
- Sitzmann, Martel, Bergman, Lindell, Wetzstein (2020) — Implicit neural representations with periodic activation functions (SIREN), NeurIPS
- Raissi, Perdikaris, Karniadakis (2019) — Physics-informed neural networks, JCP
- Fang (2021) — A high-efficient hybrid physics-informed neural networks based on convolutional neural network
- Chen, Hu, Xu (2022) — Meta-learning for physics-informed neural networks: A case study of stiff ordinary differential equations
