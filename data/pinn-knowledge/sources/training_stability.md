# Training Stability

## Equation Type
Not a PDE — a collection of diagnostics and remedies for unstable PINN training.
Applies to all PINNs; critical for production use.

## Key Findings
- Adam optimizer with lr=1e-3 is the standard starting point; rarely needs changing
- L-BFGS as a second phase (after Adam convergence) often achieves 1-2 orders lower error
- NaN loss: almost always caused by exploding gradients from automatic differentiation of high-order PDEs
- Gradient clipping at max_norm=1.0 prevents most NaN events
- Learning rate that is too high (> 5e-3) causes oscillation; lr too low (< 1e-5) means no progress
- Cosine annealing or exponential decay schedules improve final accuracy over fixed lr
- Batch size: PINN uses full collocation set per step (full-batch) — mini-batching is unusual and risky
- Weight initialization: Glorot (Xavier) uniform is standard for tanh; He for ReLU (rare in PINNs)
- Double precision (float64): 2-4x slower but essential for L-BFGS convergence and high-accuracy problems

## Recommended Architecture
Training protocol:
1. Adam, lr=1e-3, for 10,000-50,000 epochs
2. Reduce lr to 1e-4 for next 10,000 epochs if loss still decreasing
3. Switch to L-BFGS (max_iter=5000, tolerance_grad=1e-7) for final polish
4. Monitor all loss components separately — total loss hiding individual failures
5. Log gradient norms every 100 epochs; clip if > 10

Optimizer hyperparameters:
- Adam: beta1=0.9, beta2=0.999, epsilon=1e-8 (defaults)
- L-BFGS: history_size=100, line_search_fn='strong_wolfe'
- Cosine annealing: T_max=10000, eta_min=1e-6

## Known Failure Modes
- NaN propagation: once NaN appears in loss, all gradients become NaN; must restart with clipping
- Premature L-BFGS: if Adam hasn't converged sufficiently, L-BFGS can diverge from a bad starting point
- Oscillating loss: indicates lr too high or conflicting loss components; reduce lr by 10x
- Loss plateaus at high value: local minimum or saddle point; try reinitialization with different seed
- Gradient norm explosion: third-order derivatives (KdV, Cahn-Hilliard) routinely produce large gradients
- Memory overflow: large networks + many collocation points + full AD graph — check peak memory

## Techniques
- Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) before every step
- NaN detection: check torch.isnan(loss) before backward(); log and stop if detected
- Learning rate finder: train for 100 steps at geometrically increasing lr, pick lr before loss explosion
- Warmup: linearly ramp lr from 0 to target over first 500 epochs; prevents early NaN
- Loss normalization: divide each loss component by its initial value L_k(0) so all start at 1.0
- Checkpoint saving: save model every 1000 epochs; resume from last good checkpoint if NaN
- Float64 for L-BFGS: cast model and data to float64 before L-BFGS phase; cast back after
- Stochastic weight averaging (SWA): average weights over last 20% of training epochs; improves generalization
- Early stopping: stop if validation residual stops improving for 5000 epochs

## References
- Liu, Nocedal (1989) — On the limited memory BFGS method for large scale optimization, Mathematical Programming
- Raissi, Perdikaris, Karniadakis (2019) — Physics-informed neural networks, JCP
- Wang, Teng, Perdikaris (2021) — Understanding and mitigating gradient pathology, SIAM
- Jagtap, Kawaguchi, Karniadakis (2020) — Adaptive activation functions accelerate convergence in deep and physics-informed neural networks
