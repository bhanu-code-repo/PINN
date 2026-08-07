# Burgers Equation

## Equation Type
Nonlinear hyperbolic PDE: u_t + u*u_x = nu*u_xx
Combines nonlinear advection with viscous diffusion. Standard test case for shock-capturing in PINNs.

## Key Findings
- Low viscosity (nu < 0.01) causes shock formation — this is the primary difficulty
- Standard PINN fails without modification for nu < 0.01: residual stagnates near shock
- At nu = 0.01/pi (Raissi et al. 2019 benchmark), vanilla PINN achieves ~1e-3 L2 error with 10k collocation points
- At nu = 0.001, vanilla PINN typically fails; RAR or causal training required
- Shock location must be well-covered by collocation; uniform sampling misses it
- Time domain [0, 1] with spatial domain [-1, 1] is the standard benchmark setup

## Recommended Architecture
- Layers: 8 hidden layers x 20 neurons (Raissi original); modern recommendation is 5-6 layers x 64 neurons
- Activation: tanh (smooth, avoids kinks in residual)
- Weight initialization: Glorot uniform
- Collocation points: 10,000 interior + 200 boundary + 100 initial condition points minimum
- For nu < 0.01: increase to 20,000-50,000 interior points, use RAR to concentrate near shock
- No ansatz needed for Dirichlet BCs (standard soft constraint works)
- Input normalization to [-1, 1] for both x and t

## Known Failure Modes
- Shock region starvation: residual loss dominated by smooth regions; shock region under-sampled
- Gradient explosion at shock: automatic differentiation produces large gradients that destabilize training
- Mode collapse in time: network learns t=0 IC but fails to propagate solution forward
- Causality violation: network may satisfy final-time state but have wrong transient — use causal weighting
- High-frequency spectral bias: tanh saturates, smooth output cannot represent sharp shocks

## Techniques
- RAR (Residual-based Adaptive Refinement): add points where |residual| > threshold every 1000 epochs
- Causal training (Wang et al. 2022): weight time slices so early time residuals are low before training late time
- Loss weighting: BC/IC weights typically need lambda=10-100x physics loss to enforce properly
- Curriculum: train at nu=0.1 first, reduce to target nu progressively
- For nu < 0.005: consider domain decomposition (XPINNs) with interface at shock location
- Fourier feature embedding can help but tanh is usually sufficient for Burgers

## References
- Raissi, Perdikaris, Karniadakis (2019) — Physics-informed neural networks, JCP
- Wang, Teng, Perdikaris (2022) — Understanding and mitigating gradient pathology, SIAM
- Lu et al. (2021) — DeepXDE: A deep learning library for solving differential equations
- Mao, Jagtap, Karniadakis (2020) — Physics-informed neural networks for high-speed flows
