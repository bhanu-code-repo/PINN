# How Prediction Works in a PINN

"Prediction" in a PINN means something quite different from normal ML. This document explains
the mental model; the companion notebook
[`notebooks/04_model_as_solution.ipynb`](../notebooks/04_model_as_solution.ipynb) demonstrates
every claim on a trained model.

## The model *is* the solution function

After training, the network is a **continuous approximation of `u(t)` itself** — a stand-in for
the closed-form solution. So "prediction" is not "generalize to unseen data"; it is simply
**evaluating the learned function at whatever points you want**:

```python
u = model(t)        # that's it — inference is just a forward pass
```

A trained harmonic-oscillator run has learned
`u(t) ≈ e^(−dt)·(envelope)·sin(a·t + b)` encoded in the weights. The `predict` CLI command just
samples that function on a grid and compares against the exact formula.

## What you can legitimately ask of the trained model

1. **The solution at any point, any resolution — mesh-free.** Unlike a classical solver that
   gives you values on a fixed grid (and interpolates between), you can query
   `t = 0.123456789` exactly. Want a 10,000-point ultra-smooth curve? Just a bigger forward
   pass — no re-solving.

2. **Derivatives, for free.** Because the model is differentiable, autograd gives you
   physically meaningful quantities the network was never explicitly asked for:

   ```python
   u_t = autograd.grad(model(t), t, ...)   # velocity u'(t)
   ```

   Position → velocity → acceleration, all from the same weights.

3. **A quality self-check.** Plug the model back into the residual `u'' + μu' + ku`. A
   near-zero residual everywhere means the solution is trustworthy *even where you have no
   exact reference* — this is how you would validate the Burgers shock, which has no closed
   form.

## What you must NOT expect

1. **No extrapolation outside the trained domain.** A model trained on `t ∈ [0, 1]` is valid
   there only — that is where collocation points enforced the physics. At `t = 1.5` the
   network outputs *something*, but it is unconstrained garbage. PINNs interpolate the physics
   inside the domain; they do not discover it outside.

2. **One model = one problem instance.** A checkpoint trained at `w0=80, d=2` solves that
   problem specifically. Ask about `w0=40` → retrain. The physics parameters are baked into
   the *loss*, not inputs to the network. (Lifting this restriction is exactly what parametric
   PINNs and operator-learning methods like DeepONet / FNO exist for — the network takes
   `(t, w0, d)` and learns the whole solution *family*.)

3. **No uncertainty awareness.** A low final loss ≠ guaranteed accuracy everywhere; the
   residual check above is your honest diagnostic.

## So in this repo

```
train    →  compress the ODE's solution into ~3k weights
predict  →  decompress: evaluate that function on a grid, no physics re-solved
```

That is why `predict` runs in milliseconds while `train` takes many seconds: solving happened
once, at training time. The checkpoint is effectively a **compiled solution** to that specific
ODE — and that is the core PINN trade: expensive solve, then essentially free, continuous,
differentiable evaluation forever after.

## See it in action

- [`notebooks/04_model_as_solution.ipynb`](../notebooks/04_model_as_solution.ipynb) — loads a
  trained harmonic checkpoint and demonstrates high-resolution evaluation, autograd
  derivatives (`u'`, `u''`), the pointwise residual check, and the extrapolation failure mode.
- `uv run train-<experiment> predict` — the CLI form: re-evaluate any saved run without
  retraining (see the experiment READMEs).
