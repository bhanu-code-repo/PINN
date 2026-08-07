# Curriculum Training and Time-Marching

## Equation Type
Not a PDE — a training strategy for time-dependent PDEs where standard full-domain training fails.
Most useful for: hyperbolic PDEs (advection, wave), stiff problems (Allen-Cahn, Van der Pol), long time horizons.

## Key Findings
- Full space-time training often fails for long time windows due to causality violation
- Root cause: optimizer satisfies final-time state while violating intermediate dynamics
- Time-marching: partition [0, T] into windows [t0, t1], [t1, t2], ...; solve each as separate PINN
- Each window uses solution from previous window as IC — propagates information forward causally
- Window width: choose so that within each window, the solution is "simple" (no major events)
- For Burgers with nu=0.01: window width ~0.1 is safe; width 1.0 (full domain) often fails
- Causal training (Wang 2022): soft version of time-marching within a single training run — no domain splitting
- Curriculum over difficulty: train easy version of problem first, progressively increase difficulty

## Recommended Architecture
Time-marching protocol:
1. Divide [0, T] into K windows: T_k = [k*Delta_t, (k+1)*Delta_t] for k=0,...,K-1
2. For k=0: train PINN with IC u(x, 0) = u_0(x)
3. For k>0: evaluate PINN_{k-1} on dense grid at t=(k*Delta_t); use as IC for PINN_k
4. Each PINN: 5000-20000 epochs; reduce epochs for later windows if solution becomes simpler
5. Window size Delta_t: aim for solution to vary by < 50% of dynamic range within each window
6. Overlap: use small overlap region (10% of window) to ensure smooth handoff

Causal training protocol (within single PINN):
- Divide time into M temporal bins: B_j = [j*dt_bin, (j+1)*dt_bin]
- Weight w_j = exp(-epsilon * sum_{k<j} L_residual(B_k))
- Large epsilon: strict causality (must solve early bins before late bins)
- Small epsilon: weak causality (approximate)
- Tune epsilon: start with 1.0, increase to 10.0 if temporal causality is still violated

## Known Failure Modes
- Window interface discontinuity: if PINN_{k-1} has error at t_k, this error propagates as wrong IC into PINN_k
- Error accumulation: small errors in each window compound — final window may have large error
- Window width too large: within-window dynamics too complex; same failure mode as full-domain training
- Window width too small: too many windows, excessive training time; also, IC interpolation errors compound more
- Overlap region conflicts: if overlap is large, optimizer sees contradicting constraints at handoff

## Techniques
Curriculum over problem difficulty:
- Parameter curriculum: train at nu=0.1 (easy), progressively reduce to nu=0.01 (target)
- Domain curriculum: train on [0, 0.5] first, expand to [0, 1.0]; solution learned early is warm start
- Noise curriculum: add small noise to IC, reduce noise over training; prevents overfitting to exact IC
- Resolution curriculum: start with 1000 collocation points, increase to 20000 over training

Progressive domain expansion:
- Spatial: train on [-1, 0], then extend to [-1, 1] using previous solution as initial guess
- Temporal: train on [0, 0.1], extend to [0, 0.2] using weights from [0, 0.1] model as init

Time-marching specifics:
- IC for each window: evaluate previous network at 1000+ points; fit IC with lambda=1000
- Transfer learning: initialize PINN_k weights from PINN_{k-1} — typically 2-5x fewer training steps
- Adaptive window width: if window error is large, split window; if small, merge with next

## References
- Wang, Wang, Perdikaris (2022) — Respecting causality is all you need for training physics-informed neural networks
- Krishnapriyan, Gholami, Zhe, Kirby, Mahoney (2021) — Characterizing possible failure modes in physics-informed neural networks, NeurIPS
- Mattey, Ghosh (2022) — A novel sequential method to train physics-informed neural networks for Allen-Cahn and Cahn-Hilliard equations
- Wight, Zhao (2021) — Solving Allen-Cahn and Cahn-Hilliard equations using the adaptive physics informed neural networks
- Penwarden, Zhe, Narayan, Kirby (2023) — A unified scalable framework for causal sweeping strategies for physics-informed neural networks
