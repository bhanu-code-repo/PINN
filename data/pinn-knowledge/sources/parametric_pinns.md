# Parametric PINNs

## Equation Type
Family of PDEs parameterized by mu: F(u; mu) = 0 in Omega, mu in parameter space P.
Example: Burgers with nu in [0.001, 0.1], or NS with Re in [100, 1000].
Goal: single network u(x, t; mu) that solves for all mu simultaneously.

## Key Findings
- Standard PINN solves one instance at a time; parametric PINN solves the entire family
- mu appended as additional network input: NN(x, t, mu) — simplest approach, often surprisingly effective
- Accuracy degrades as parameter range widens; interpolation works, extrapolation rarely does
- For smooth parameter dependence: parametric PINN achieves near-single-instance accuracy
- For parameter-dependent features (shock at different x for different nu): harder, needs more capacity
- Neural operators (DeepONet, FNO) are an alternative — learn solution operator rather than single instance
- Uncertainty quantification: ensemble of parametric PINNs trained with different parameter subsets
- Inference speedup: once trained, evaluation is instantaneous for any mu in training range

## Recommended Architecture
- Input: [x, t, mu] concatenated — mu can be scalar or vector if multiple parameters
- Layers: increase by 1-2 layers vs single-instance baseline (more capacity needed)
- Typically: 6-8 layers x 64-128 neurons for 1D PDE with 1 parameter
- Parameter embedding: if mu has known structure (e.g., log-uniform), embed as log(mu) or (mu - mu_mean)/mu_std
- Collocation: sample (x, t, mu) jointly — use LHS over the full product space
- For K parameters: N_total = N_base * K^0.5 (sublinear growth works for smooth parameter dependence)
- Conditional architecture: use mu to modulate network via FiLM layers (gamma, beta per layer)

## Known Failure Modes
- Parameter extrapolation: never trust predictions outside training parameter range
- Competing modes: for Burgers, shock location depends on nu; network must learn nu-conditional location
- Capacity bottleneck: single small network cannot represent all parameter instances; increase size
- Unbalanced parameter sampling: if uniform in mu but solution varies more for small mu, use log-uniform
- Gradient conflict: gradients from different mu values can point in opposing directions — GradNorm helps

## Techniques
- Log-space parameter embedding: for scale parameters (nu, Re, k), always embed as log(mu)
- FiLM conditioning: gamma(mu)*NN_hidden + beta(mu) where gamma, beta are small MLPs on mu
  This lets mu modulate each hidden layer independently — significantly better than concatenation
- Curriculum over mu: train easy (large nu) first, progressively add harder (small nu)
- Deep ensembles for UQ: train 5-10 parametric PINNs with different init; variance gives uncertainty
- Parameter-to-solution map (P2S): post-train a cheap surrogate that maps mu to key solution statistics
- Active learning over parameters: identify mu values where ensemble variance is high, add training instances
- Transfer learning: warm-start hard-mu training from easy-mu weights

## References
- Raissi, Perdikaris, Karniadakis (2019) — Physics-informed neural networks, JCP (parametric setup)
- Sun, Liu, Wang, Sun (2020) — Surrogate modeling for fluid flows using physics-informed neural networks
- Mao, Jagtap, Karniadakis (2020) — Physics-informed neural networks for high-speed flows
- Psaros, Meng, Zou, Guo, Karniadakis (2023) — Uncertainty quantification in scientific machine learning
- De Ryck, Jagtap, Mishra (2023) — Error estimates for physics-informed neural networks approximating the Navier-Stokes equations
