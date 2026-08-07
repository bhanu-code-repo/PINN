# Spectral Bias (F-Principle)

## Equation Type
Not a PDE — a fundamental property of neural network training dynamics.
Affects all PINNs. Understanding spectral bias is prerequisite for diagnosing PINN failures.

## Key Findings
- F-principle (Xu et al. 2019): neural networks trained with gradient descent learn low-frequency components first
- Convergence rate for frequency omega scales as ~exp(-omega^2 * t) during early training
- Implication: high-frequency solution components (from high-k PDEs, sharp features) are learned last or not at all
- NTK (Neural Tangent Kernel) perspective: kernel eigenvalue spectrum governs convergence rate per frequency
- Low eigenvalues in NTK = slow learning for that frequency component
- Tanh networks have NTK spectrum that decays rapidly with frequency
- ReLU networks have faster high-freq learning than tanh — but lose smoothness for PDE residuals
- Magnitude of spectral bias grows with depth: deeper networks are more biased toward low frequencies
- Practical threshold: if target solution requires frequencies > ~10 (in normalized domain), expect spectral bias issues

## Recommended Architecture
Mitigation strategies ranked by effectiveness:
1. Fourier feature embedding: map x -> [sin(B*x), cos(B*x)] where B sampled from N(0, sigma^2)
2. SIREN: sinusoidal activations with special initialization (Sitzmann et al. 2020)
3. Multiscale architecture: parallel branches for different frequency ranges, combine outputs
4. Random feature networks: fixed random Fourier features in first layer, train only output layer
5. Modified MLP (Wang et al. 2022): input encoding with UV transform to normalize NTK spectrum

## Known Failure Modes
- Silent failure: network appears to train (loss decreasing) but high-frequency components are missing
- Diagnostic: compare Fourier spectrum of PINN output vs reference solution
- Fourier feature sigma too small: only captures low freq; sigma too large: optimization is unstable
- SIREN divergence: sin activation with wrong init diverges; must use Sitzmann initialization exactly
- Frequency aliasing: if dominant frequency is not in embedding, network learns wrong modes

## Techniques
- Fourier feature embedding: sigma for B should be ~k_max/3 where k_max is highest relevant frequency
  - For Helmholtz k=10: sigma ~ 10; embed 256 frequencies
  - For Burgers shock (effective frequency ~50): sigma ~ 15-20
- NTK monitoring: compute eigenvalue spectrum of NTK at initialization; check if target frequencies have non-zero eigenvalues
- Frequency sweep: train on single-frequency targets to measure per-frequency convergence rate for given architecture
- Modified MLP: u(x) = W_out * (U(x) * V(x)) where U,V are tanh-encoded branch networks
- Spectral normalization of weights: reduces variance in NTK spectrum
- Input gradient monitoring: if d(loss)/d(input) is low at high spatial frequencies, spectral bias is present

## References
- Xu, Zhang, Luo, Xiao, Ma (2019) — Frequency principle: Fourier analysis sheds light on implicit regularization of deep neural networks, ICLR
- Tancik, Srinivasan, Mildenhall et al. (2020) — Fourier features let networks learn high frequency functions in low dimensional domains, NeurIPS
- Wang, Teng, Perdikaris (2022) — Understanding and mitigating gradient pathology in physics-informed neural networks, SIAM
- Sitzmann, Martel, Bergman, Lindell, Wetzstein (2020) — Implicit neural representations with periodic activation functions, NeurIPS
- Jacot, Gabriel, Hongler (2018) — Neural tangent kernel: Convergence and generalization in neural networks, NeurIPS
