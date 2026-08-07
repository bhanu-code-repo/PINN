# Collocation Strategies

## Equation Type
Not a PDE — a sampling strategy for selecting where to evaluate the PDE residual during training.
Collocation point placement is one of the highest-leverage design choices in PINN setup.

## Key Findings
- Uniform random sampling is the default but rarely optimal
- Latin Hypercube Sampling (LHS) provides better space-filling than uniform; use for d >= 2
- RAR (Residual-based Adaptive Refinement) is the most impactful single improvement for hard problems
- Sobol sequences and Halton sequences: low-discrepancy quasi-Monte Carlo; better than random for smooth problems
- Optimal transport for collocation: distribute points proportional to sqrt(|residual|) — theoretical optimum
- For time-dependent PDEs: temporal collocation density must be higher at early times (fast dynamics)
- For problems with sharp features (shocks, interfaces): 10-100x more points needed in feature region
- Empirical rule: total points N should satisfy N > 10^d * k_max where d=dimension, k_max=highest frequency

## Recommended Architecture
Sampling strategy selection guide:
- Smooth elliptic problems (Poisson, Laplace): LHS or Sobol, uniform in time if parabolic
- Parabolic with smooth solution (heat): LHS with mild temporal bias toward t=0
- Hyperbolic with shocks (Burgers, Advection): uniform + RAR (mandatory)
- High-frequency (Helmholtz k>5, Wave c>5): uniform + Fourier-frequency-aligned sampling
- Phase-field (Allen-Cahn): RAR with interface detection
- Stiff ODE systems: dense sampling near fast transient regions

RAR protocol:
1. Train for K=1000 epochs with initial collocation set
2. Evaluate |residual| on dense evaluation grid (10x the collocation points)
3. Add M new points sampled with probability proportional to |residual|^p, p=1 or p=2
4. Retrain; repeat every K epochs for 5-10 rounds

## Known Failure Modes
- Too few points: residual is satisfied at collocation points but wildly wrong elsewhere — overfitting to points
- Clustered points: if all points cluster in one region, the rest of the domain is unconstrained
- RAR feedback loop: residual peaks at true solution discontinuity, RAR adds points, network fits discontinuity incorrectly
- Static collocation: re-using same points every epoch can lead to memorization rather than generalization
- Temporal bias absent: uniform sampling in time leads to poor IC enforcement relative to final-time behavior

## Techniques
- LHS: use scipy.stats.qmc.LatinHypercube for space-filling sampling; strictly better than random in 2D+
- RAR with warmup: do not start RAR until after 2000 training epochs; early residuals are uninformative
- Residual-proportional sampling: new_points ~ p(x) where p(x) = |r(x)|/int(|r|)
- Temporal bias: sample t ~ Beta(0.5, 2.0) to oversample early times, or use t = t_max * u^2 where u ~ Uniform
- Domain boundary oversampling: always use 10-20x more boundary points than interior per unit area
- Resampling every epoch: dynamically resample collocation each epoch (more expensive, avoids memorization)
- For 1D problems: rejection sampling with density proportional to |d^2u/dx^2| of a preliminary solution
- Minimum points per region: if domain has N distinct subregions, ensure each has at least 100 points

## References
- Lu, Meng, Mao, Karniadakis (2021) — DeepXDE: A deep learning library for solving differential equations, SIAM Review
- Wu, Zhu, Tan, Kartha, Lu (2023) — A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks
- Daw, Bu, Wang, Perdikaris, Karpatne (2022) — Mitigating propagation failures in physics-informed neural networks using retain-resample-release (R3) sampling
- Tang, Haahr, Chen (2023) — DAS: A deep adaptive sampling method for solving partial differential equations
