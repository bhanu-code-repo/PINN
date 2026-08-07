# Inverse Problems

## Equation Type
Goal: identify unknown parameters theta in F(u; theta) = 0 given sparse observations of u.
Examples: find nu in Burgers from velocity snapshots; find k in Helmholtz from scattered field measurements.
Loss: L = L_residual(theta) + L_data(theta) where L_data penalizes mismatch with observations.

## Key Findings
- Inverse problems are often well-suited to PINNs — physics constrains the solution even with sparse data
- theta is treated as a trainable variable (alongside network weights); optimized jointly
- Works best when: forward problem is well-posed, theta enters linearly (or mildly nonlinearly)
- Fails when: problem is ill-posed, multiple theta give similar observations (non-identifiable)
- Raissi et al. (2019) demonstrated nu identification in NS from 10 scattered observations — works reliably
- Over-parameterization: if theta is a function (spatially varying), use separate NN or basis expansion for theta
- Identifiability: if u -> theta is many-to-one, regularization on theta is necessary
- Data noise amplification: measurement noise in u amplifies into theta estimate via inversion

## Recommended Architecture
- Network for u: standard PINN architecture (4-8 layers, tanh)
- theta as scalar: torch.nn.Parameter([theta_init]) — one extra variable
- theta as function: separate small NN (2-3 layers x 32 neurons) for theta(x)
- Data loss: L_data = sum_{i=1}^{N_data} |NN(x_i, t_i) - u_obs_i|^2 / N_data
- Physics loss: standard residual on collocation points
- Loss weights: lambda_data = 1-100 (higher than physics if data is clean, lower if noisy)
- Collocation: fill domain densely even where no observations; physics constrains interpolation
- N_data: 10-100 observations usually sufficient for scalar theta identification; more for theta(x)

## Known Failure Modes
- Non-identifiability: u depends on theta only through certain combinations; rank-deficient sensitivity
  Example: diffusivity k in heat equation is non-identifiable from final-time-only data if t is large
- Local minima in theta: loss landscape in (weights, theta) can have multiple basins
- Data overfitting: if lambda_data too high and N_data too small, network fits noise in observations
- Physics stiffness at wrong theta: during optimization, theta may pass through values that make PDE stiff
- Regularization bias: adding L2 regularization on theta introduces bias; theta is pulled toward zero

## Techniques
- Sensitivity analysis: compute d(u)/d(theta) at init to verify theta influences the observable quantities
- Initialization: start theta at physical prior (e.g., nu=0.01 for Burgers flow); don't start at zero
- Alternating optimization: fix theta, update network weights for K steps; then update theta one step
- Bayesian PINN: treat theta as random variable with prior; use variational inference for posterior
- Ensemble-based UQ: run 20 inversions with different random seeds; spread in theta estimates = uncertainty
- Regularization for function-valued theta: total variation or L2 gradient penalty on theta(x)
- Data normalization: normalize observations to O(1) before computing L_data; prevents scale mismatch
- Hessian analysis: compute d^2(L)/d(theta)^2 at converged solution; large Hessian = well-identified
- Tikhonov regularization: L_reg = ||theta - theta_prior||^2 with weight tuned by cross-validation

## References
- Raissi, Perdikaris, Karniadakis (2019) — Physics-informed neural networks, JCP
- Chen, Koohy, Hochholzer, Babaee (2020) — Physics-informed neural networks for inverse problems in structural dynamics
- Yang, Meng, Karniadakis (2021) — B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data
- Tartakovsky, Marrero, Perdikaris, Tartakovsky, Barajas-Solano (2020) — Physics-informed deep neural networks for learning parameters and constitutive relationships in subsurface flow problems
