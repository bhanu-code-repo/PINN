# Loss Weighting and Gradient Pathology

## Equation Type
Not a PDE — a training methodology concern affecting all multi-loss PINN formulations.
Total loss: L = lambda_r * L_residual + lambda_bc * L_bc + lambda_ic * L_ic + ...

## Key Findings
- Gradient pathology (Wang et al. 2021): different loss terms have different gradient magnitudes; optimizer is dominated by the largest
- If lambda_bc << lambda_r in terms of gradient norms, BCs are effectively ignored
- Fixed lambda values almost always suboptimal; the optimal ratio changes during training
- NTK analysis shows: gradient of each loss component has different Jacobian norm
- Empirical rule: lambda_bc/lambda_r = max(|grad L_r|) / max(|grad L_bc|) should be ~1 for balanced training
- Adaptive NTK weighting: compute this ratio every K epochs (K=1000 is common) and update lambdas
- GradNorm (Chen et al. 2018): normalize gradient magnitudes across tasks — effective for PINN
- IC weights often need to be 10-1000x physics residual weight; this is normal and expected
- Soft-Adapt (Heydari et al. 2019): weight by rate of loss decrease; fast-decreasing terms get lower weight

## Recommended Architecture
No single architecture change, but loss monitoring is critical:
- Log each loss component separately every epoch (not just total loss)
- Log gradient norms for each component
- Use AdaptiveLossWeighter if available in the codebase
- Typical starting weights: lambda_r=1, lambda_bc=10, lambda_ic=100
- Adjust based on monitoring: if L_bc stagnates while L_r decreases, increase lambda_bc

## Known Failure Modes
- Over-weighting IC: lambda_ic too high causes network to perfectly fit IC but ignore PDE — overfitting
- Under-weighting BC: solution drifts away from boundary; problem looks solved but BCs are violated
- Silent gradient starvation: one loss term drives gradients entirely, others receive near-zero updates
- Lambda oscillation: adaptive weights oscillate if update rate is too high; use exponential moving average
- Conflicting objectives: for some stiff PDEs, IC and residual loss are inherently in tension; weights alone cannot fix this

## Techniques
- NTK-based adaptive weighting (Wang et al. 2021):
  lambda_k = <max gradient norm across all losses> / <gradient norm of loss k>
  Update every 1000 epochs; use EMA with rate 0.9 to smooth
- GradNorm: maintain a target gradient norm ratio; backprop through the norms to update lambdas
- Soft-Adapt: lambda_k proportional to 1/|dL_k/dt| — rewards slowly-decreasing terms
- Relative loss monitoring: track L_k(t)/L_k(0) for each component; all should decrease similarly
- Manual tuning heuristic: first train with only residual loss, then add BC/IC losses progressively
- Log-space lambda optimization: keep lambdas as exp(log_lambda) to ensure positivity during updates
- Our AdaptiveLossWeighter (in libs/pinn_core): implements NTK-based scheme with EMA smoothing

## References
- Wang, Teng, Perdikaris (2021) — Understanding and mitigating gradient pathology in physics-informed neural networks, SIAM
- Chen, Badrinarayanan, Lee, Rabinovich (2018) — GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks, ICML
- Heydari, Thompson, Mehmood (2019) — Softadapt: Techniques for adaptive loss weighting of neural networks with multi-part loss functions
- McClenny, Braga-Neto (2023) — Self-adaptive physics-informed neural networks using a soft attention mechanism
