# Allen-Cahn Equation

## Equation Type
Nonlinear reaction-diffusion PDE: u_t = epsilon^2 * u_xx + u - u^3
Models phase separation; u in [-1, 1] represents two phases. Sharp interface between phases.
Related to Cahn-Hilliard (4th order), which is significantly harder.

## Key Findings
- Sharp interface width scales as epsilon; for epsilon = 0.01, interface is extremely thin
- Primary challenge: resolving the interface requires very fine collocation in interface region
- Vanilla PINN fails for epsilon < 0.1: interface region under-sampled, network smooths it out
- Benchmark: x in [-1,1], t in [0,1], u(x,0) = x^2*cos(pi*x), periodic BCs
- At epsilon=0.01: standard PINN achieves only ~10% accuracy; causal training + RAR reduces error to ~1%
- Interface dynamics: phases evolve to minimize surface energy; final state has two flat regions
- u^3 - u nonlinearity has bistable equilibria at u = +/-1 — network must find both

## Recommended Architecture
- Layers: 8 hidden layers x 64 neurons
- Activation: tanh (ReLU cannot represent smooth interface profiles)
- Collocation points: 25,000 interior minimum; adaptive refinement critical
- Interface-aware sampling: oversample near |u| < 0.5 region (interface zone)
- IC points: 512 uniformly spaced in x at t=0
- Periodic BC: hard enforcement via Fourier ansatz or explicit periodization layer
- Time collocation: bias toward early times (t < 0.2) where dynamics are fastest

## Known Failure Modes
- Interface smearing: network cannot represent sharp transition; learns broad smooth profile
- Phase pinning: network gets stuck with interface at wrong location; hard to escape this basin
- Mass non-conservation: int(u dx) should be approximately constant; often violated
- Stiff dynamics: for small epsilon, time scales are O(epsilon^2) — very stiff system
- Periodic BC soft enforcement: gradient at boundaries must match; soft constraint often insufficient

## Techniques
- Causal training: essential for this PDE — start from t=0, propagate forward with epsilon-causal weights
- RAR (Residual-based Adaptive Refinement): query |residual| every 500 epochs, add 1000 points where large
- Interface tracking: if interface location is approximately known, bias sampling there explicitly
- Hard periodic BC: u(x,t) = u(-x,t) is NOT the BC; periodic means u(-1,t) = u(1,t), u_x(-1,t) = u_x(1,t)
- Conservation loss: add L_mass = (mean(u) - mean(u0))^2 to stabilize long-time dynamics
- Adaptive epsilon curriculum: start with epsilon=0.1 (easy), reduce to epsilon=0.01 (target)
- Time-marching with small windows is often more reliable than full space-time solve

## References
- Wang, Teng, Perdikaris (2022) — Understanding and mitigating gradient pathology in physics-informed neural networks
- Wight, Zhao (2021) — Solving Allen-Cahn and Cahn-Hilliard equations using the adaptive physics informed neural networks
- McClenny, Braga-Neto (2023) — Self-adaptive physics-informed neural networks
