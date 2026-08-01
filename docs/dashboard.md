# PINN Dashboard

Interactive Streamlit dashboard for browsing training runs, comparing
experiments, and running live parametric predictions.

---

## Quick Start

```bash
# Install dependencies (if not done already)
uv sync --all-packages

# Launch the dashboard
uv run streamlit run dashboard.py
```

The dashboard opens in your browser at `http://localhost:8501`. No training
runs are required for the app to launch, but pages will be empty until you
train at least one experiment:

```bash
uv run train-harmonic train -e 500 --no-show        # quick single-instance run
uv run train-parametric train -e 5000 --no-show      # parametric (needed for Predictor page)
```

---

## Pages

### 1. Overview (Run Browser)

Shows all experiments and runs discovered under `outputs/`. Each run displays:

- **Run name** (timestamp-based directory name)
- **Config summary** — epochs, seed
- **Final loss** and key experiment-specific metrics (rel-L2 errors, inferred
  parameters, etc.)

Runs are grouped by experiment and sorted newest-first. Use this page to get a
quick snapshot of all training activity.

### 2. Run Detail

Drill into a single run. Select an experiment and run from the dropdowns to see:

- **Configuration** — all hyperparameters stored in the checkpoint metadata
  (epochs, learning rate, network size, seed, etc.)
- **Metrics** — all values from `metrics.json` (final losses, rel-L2 errors,
  inferred parameters)
- **Loss history** — interactive matplotlib plot (log scale) showing total loss
  and all individual loss terms over training epochs. Loaded directly from the
  checkpoint.
- **Artifacts** — file listing with sizes for every artifact in the run
  directory (checkpoints, plots, logs, predictions)
- **Plots** — all PNG images generated during training and prediction, displayed
  inline

### 3. Compare

Side-by-side comparison of multiple runs within the same experiment.

- **Metrics table** — all runs in a single dataframe with config and metrics
  columns, easy to sort and scan
- **Overlaid loss curves** — total loss histories from all selected runs plotted
  on the same axes (log scale), making it easy to compare convergence speed,
  final loss, and stability

Select which runs to include via the multiselect widget (defaults to the 5 most
recent runs).

### 4. Parametric Predictor

Interactive live inference for all four parametric experiments. This page:

1. Detects which parametric experiments have trained runs
2. Loads the checkpoint (cached across Streamlit reruns via `@st.cache_resource`)
3. Presents experiment-specific parameter sliders
4. Runs the model on a dense grid and plots the result in real time

#### Supported experiments

| Experiment | Parameters | Sliders | Plots |
|------------|-----------|---------|-------|
| `parametric_harmonic` | `w0`, `d` | Frequency 20-100, damping 0.1-4.0 | u(t) vs exact, +/- 2 sigma band |
| `parametric_burgers` | `nu` | Viscosity (from config range) | Contour u(x,t), IC + t=1 snapshots |
| `parametric_schrodinger` | `A` | Amplitude (from config range) | Contour \|h(x,t)\|, snapshots vs exact IC |
| `parametric_taylor_green` | `nu` | Viscosity 0.001-0.1 | Speed field: exact, PINN, error at t=T/2 |

For ensemble runs (trained with `--ensemble N`), the predictor shows the
ensemble mean and +/- 2 sigma epistemic uncertainty bands.

**Tip:** Drag a slider slowly to watch the solution family evolve continuously.
This is the payoff of parametric PINNs — one model, instant evaluation at any
parameter value.

---

## How It Works

### Run discovery

The dashboard scans `outputs/` for directories containing `metrics.json`. The
directory structure follows the convention established by the experiments:

```
outputs/
├── harmonic_oscillator/
│   ├── 20260801-120000/       # each run is a timestamped directory
│   │   ├── checkpoint.pt
│   │   ├── metrics.json
│   │   ├── loss_history.png
│   │   └── logs/
│   └── 20260801-130000/
├── parametric_harmonic/
│   └── ...
└── ...
```

### Checkpoint loading

Loss histories and model weights are loaded from `checkpoint.pt` files. The
checkpoint format is self-describing — training config is stored in the
`metadata` field, so the dashboard can reconstruct the model architecture
without any external configuration.

### Model caching

Parametric model loading uses `@st.cache_resource` so models are loaded once
and reused across slider interactions. Switching to a different run or
experiment clears and reloads the cache.

---

## Customisation

### Changing the outputs directory

The dashboard reads from `outputs/` relative to the working directory. To point
it at a different location, either:

- Run from a different directory: `cd /path/to/project && uv run streamlit run dashboard.py`
- Or modify `OUTPUTS_ROOT` at the top of `dashboard.py`

### Adding a new parametric experiment

To make a new parametric experiment appear in the Predictor page:

1. Ensure the experiment module exports `build_model` and `ensemble_predict_grid`
   (or equivalent) functions
2. Add an `elif` branch in `_cached_load_models()` to import the right
   `build_model`
3. Add a `_parametric_<name>()` function for the slider UI and plotting
4. Wire it into `page_parametric()` with an `elif` branch

### Network port

```bash
uv run streamlit run dashboard.py --server.port 8502
```

### Headless mode (no browser auto-open)

```bash
uv run streamlit run dashboard.py --server.headless true
```

---

## Requirements

- `streamlit >= 1.45.0` (included in project dependencies)
- `torch`, `numpy`, `matplotlib` (already required by the core library)
- Trained experiment runs in `outputs/` (for non-empty pages)
