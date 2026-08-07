# Helmholtz Equation

## Equation Type
Elliptic PDE: u_xx + u_yy + k^2 * u = f(x,y)
Arises from time-harmonic wave problems (acoustics, electromagnetics, optics).
Parameter k is the wavenumber; large k means rapidly oscillating solution.

## Key Findings
- High wavenumber (k > 10) is a catastrophic failure mode for standard PINNs
- Root cause: F-principle (spectral bias) — networks learn low frequencies first, cannot represent k-th mode
- For k=1: vanilla PINN converges easily, error < 1e-4 with 5000 points
- For k=10: vanilla PINN error typically > 1e-1; Fourier features reduce to ~1e-2
- For k=50: even Fourier features struggle without careful tuning; SIREN recommended
- Solution oscillates k/pi times per unit length; collocation density must resolve this
- Minimum collocation density: at least 10 points per wavelength (2pi/k)
- Boundary conditions are Dirichlet (absorbing) or Robin (radiation condition)
- Negative k^2 term can make the operator indefinite — optimization landscape is non-convex

## Recommended Architecture
- For k < 5: 4 layers x 64 neurons, tanh activation
- For k in [5, 20]: 6-8 layers x 256 neurons + Fourier feature embedding
- For k > 20: SIREN architecture (sin activation, specific initialization scheme)
- Fourier features: include frequencies up to 2k in the embedding
- Collocation density: N_points > 10 * k * L (L = domain size) in each spatial direction
- For k=20 on [0,1]^2: minimum 200x200 = 40,000 points; use 100,000 for reliability
- Output: single real-valued function (u) or [u_real, u_imag] for complex amplitude

## Known Failure Modes
- Spectral bias: network converges to smooth (k=0) solution; residual appears to decrease but wrong
- Under-resolved collocation: if fewer than ~10 pts/wavelength, gradient signal is aliased
- SIREN instability: sin activation with wrong initialization leads to divergence; must use official init
- Indefinite operator: for certain k and geometry, matrix has negative eigenvalues — optimizer oscillates
- Radiation BCs: first-order absorbing BC (u_n + ik*u = 0) is a Robin condition; must be exactly implemented

## Techniques
- Fourier feature embedding: B_ij ~ N(0, sigma^2), sigma tuned to k; embed x -> [sin(Bx), cos(Bx)]
- SIREN: u(x) = sin(omega_0 * (W*x + b)) with omega_0 = 30; init W ~ U(-sqrt(6/n), sqrt(6/n)) for hidden
- Input scaling: normalize domain to [-pi, pi] so that frequency k maps to O(1) wavenumber
- Multiscale network: parallel sub-networks for different frequency bands, combine outputs
- Preconditioning: solve preconditioned system with known Green's function
- For high k: consider splitting u = u_incident + u_scattered, solve only for scattered field
- Gauss-Legendre quadrature for collocation (better than uniform) for smooth integrands

## References
- Sitzmann, Martel, Bergman, Lindell, Wetzstein (2020) — Implicit neural representations with periodic activation functions (SIREN), NeurIPS
- Tancik, Srinivasan, Mildenhall et al. (2020) — Fourier features let networks learn high frequency functions in low dimensional domains, NeurIPS
- Wang, Wang, Perdikaris (2021) — On the eigenvector bias of Fourier feature networks for neural field reconstruction, CVPR
- Moseley, Markham, Nissen-Meyer (2020) — Finite basis physics-informed neural networks as a Schwarz domain decomposition method
