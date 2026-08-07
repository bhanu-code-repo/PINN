# Advection Equation

## Equation Type
Linear first-order hyperbolic PDE: u_t + c*u_x = 0
Solution: u(x,t) = u0(x - c*t) — exact translation of initial condition at speed c.
No dissipation, no smoothing. Hardest class for standard PINNs.

## Key Findings
- Widely regarded as one of the hardest problems for PINNs
- For c=1, standard PINN achieves ~1e-2 L2 error; for c=40, error typically > 1e-1
- Root cause: no diffusion means no smoothing — any approximation error propagates without decay
- Characteristics are lines x - c*t = const; network must align with these exactly
- Spectral bias: network prefers smooth, low-frequency solutions but advection requires exact IC propagation
- Causal training is necessary but not sufficient for large c
- At c > 10: vanilla PINN essentially fails without Fourier features or physics-aware architecture
- Periodic IC (sin wave) easier than non-periodic; discontinuous IC (step function) nearly impossible

## Recommended Architecture
- Layers: 6-8 hidden layers x 128 neurons
- Activation: tanh with Fourier feature embedding (mandatory for c > 5)
- Fourier features: embed (x,t) as [sin(2pi*k*x), cos(2pi*k*x), sin(2pi*k*c*t), cos(2pi*k*c*t)] for k=1..20
- Collocation points: 20,000-50,000; characteristic-aligned sampling strongly preferred
- IC points: dense, 2,000+; IC must be weighted very high (lambda_IC = 100-1000)
- Characteristic-aware sampling: place collocation points along characteristics x - c*t = const

## Known Failure Modes
- Trivial solution: u(x,t) = 0 satisfies the PDE if IC weight is too low — network learns zero solution
- Phase velocity error: network propagates wave at wrong speed; solution looks correct but is translated
- Discontinuous IC: step function IC causes Gibbs-like oscillations; network cannot represent jump
- Large c with periodic domain: solution wraps around multiple times; network sees aliased signal
- Training on residuals only: without sufficient IC weight, PDE residual is small for many wrong solutions

## Techniques
- Characteristics-based sampling: explicitly compute characteristic lines and place collocation on them
- Dense IC enforcement: lambda_IC = 1000, 5000+ IC points — this is the primary anchor
- Causal temporal weighting: weight early-time residuals first (epsilon-causal scheme)
- Fourier feature embedding tuned to advection speed: include frequencies at multiples of c
- Method of characteristics hybrid: solve along characteristics analytically, use PINN only for interpolation
- Consider finite-difference pretraining: warm-start network with FD solution before PINN fine-tuning
- Time-stepping with small windows: solve [0, L/c] as one window, avoid multi-period wrapping

## References
- Krishnapriyan, Gholami, Zhe, Kirby, Mahoney (2021) — Characterizing possible failure modes in physics-informed neural networks, NeurIPS
- Wang, Wang, Perdikaris (2022) — Respecting causality is all you need for training physics-informed neural networks
- Peng, Liu, Wen, Li (2022) — Attention-based physics-informed neural networks for the advection-diffusion equation
