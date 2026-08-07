# Heat Equation (Diffusion)

## Equation Type
Parabolic PDE: u_t = alpha * u_xx (1D) or u_t = alpha * laplacian(u) (nD)
Linear, well-posed, strongly regularizing. The canonical "easy" PINN benchmark.

## Key Findings
- Most tractable class for PINNs: parabolic character means smooth solutions and no shocks
- Vanilla PINN achieves L2 error < 1e-4 with 5000 collocation points for standard setups
- Diffusivity alpha has minimal impact on difficulty (contrast: Burgers low-nu)
- 2D heat equation on unit square with Dirichlet BCs: converges in ~10k Adam + 1k L-BFGS steps
- IC must be enforced strongly; soft weighting often sufficient with lambda_IC = 10
- Steady-state limit (Poisson equation) is even easier — see poisson_equation.md
- Multi-scale heat equation (multiple alpha values in different regions) becomes harder

## Recommended Architecture
- Layers: 3-4 hidden layers x 32-64 neurons — larger is unnecessary, slower, no gain
- Activation: tanh or sigmoid; ReLU not recommended (second derivatives are zero)
- Collocation points: 3,000-5,000 interior; 500 boundary; 200 IC points
- For high-dimensional heat equation (d > 3): use 10,000+ points, consider DeepGalerkin method
- Hard IC ansatz: u(x,t) = u0(x) + t * NN(x,t) ensures exact IC satisfaction at t=0
- This ansatz is particularly effective here because the exact IC form is simple

## Known Failure Modes
- Essentially none for simple geometries — this is the "works out of the box" case
- Multi-scale domains with different thermal conductivities: interface discontinuity in derivatives
- High-dimensional inputs (d > 5): curse of dimensionality in collocation, use QMC sampling
- Long time horizons (t >> 1/alpha): solution becomes nearly constant, gradient signal weakens
- Oscillatory IC (e.g., u0 = sin(20*pi*x)): spectral bias slows learning of high-freq IC

## Techniques
- Standard Adam optimizer with lr=1e-3 followed by L-BFGS for polishing
- No special techniques needed for standard 1D/2D problems
- For oscillatory IC: apply Fourier feature embedding or increase IC weight substantially (lambda=1000)
- For long time domains: time-marching (split [0,T] into windows, solve sequentially)
- IC hard constraint: u_hat(x,t) = u0(x) + t*NN(x,t) — removes IC as training variable

## References
- Raissi, Perdikaris, Karniadakis (2019) — Physics-informed neural networks, JCP
- Lagaris, Likas, Fotiadis (1998) — Artificial neural networks for solving ODE/PDE boundary value problems
- Berg, Nystrom (2018) — A unified deep artificial neural network approach to PDE in complex geometries
