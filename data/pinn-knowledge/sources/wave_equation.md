# Wave Equation

## Equation Type
Hyperbolic PDE: u_tt = c^2 * u_xx (1D) or u_tt = c^2 * laplacian(u) (nD)
Requires two initial conditions: u(x,0) = f(x) and u_t(x,0) = g(x).

## Key Findings
- Significantly harder than heat equation for PINNs — hyperbolic character, no dissipation
- High wave speed c causes rapid oscillation; spectral bias prevents learning high-freq waves
- Standard PINN fails for c > 10 without Fourier features or special architecture
- For c=1 on [0,1]x[0,1]: vanilla PINN achieves ~1e-3 L2 error with 10k points
- For c=10: error typically > 1e-1 without Fourier features; Fourier features reduce to ~1e-2
- Two ICs (displacement and velocity) must both be weighted carefully
- Long-time integration accumulates error — wave packets drift from true solution
- Causal training critical: PINN can fit t=T while violating t in (0,T)

## Recommended Architecture
- Layers: 6-8 hidden layers x 64-128 neurons
- Activation: tanh (preferred) or sin (for wave-dominated problems)
- Fourier feature embedding: map (x,t) -> [sin(2pi*k*x/L), cos(2pi*k*x/L), ...] for k=1..K, K=10-20
- Collocation points: 10,000-20,000 interior; 1,000 boundary; 500 per IC (u and u_t separately)
- For c > 5: mandatory Fourier feature embedding or sinusoidal activation (SIREN)
- Two separate IC losses: L_IC_u = ||NN(x,0) - f(x)||^2 and L_IC_ut = ||NN_t(x,0) - g(x)||^2

## Known Failure Modes
- Spectral bias: network learns low-frequency components, misses high-freq wave structure
- IC velocity violation: u_t IC is a derivative condition — often weighted too low, poorly enforced
- Causality violation: without causal weighting, late-time fit dominates and early dynamics are wrong
- Resonance instability: for certain domain sizes and c, residual oscillates and fails to converge
- Reflection artifacts at boundaries: absorbing BCs harder than Dirichlet/Neumann

## Techniques
- Causal training (Wang et al. 2022): epsilon-causal weights w_i = exp(-epsilon * sum of earlier residuals)
- Fourier feature embedding: essential for c > 5
- SIREN (Sitzmann et al. 2020): sinusoidal activation, specialized init — excellent for wave problems
- Separate loss terms for u and u_t ICs with independent weights; tune lambda_u_t = 10-100
- Time-marching: split domain into overlapping time windows, use solution from window k as IC for k+1
- Domain decomposition (XPINNs): particularly effective for wave propagation problems

## References
- Sitzmann, Martel, Bergman, Lindell, Wetzstein (2020) — Implicit neural representations with periodic activation functions (SIREN), NeurIPS
- Wang, Teng, Perdikaris (2022) — Understanding and mitigating gradient pathology, SIAM
- Moseley, Markham, Nissen-Meyer (2020) — Solving the wave equation with physics-informed deep learning
- Waheed, Haghighat, Alkhalifah, Song, Hao (2021) — PINNeik: Eikonal solution using physics-informed neural networks
