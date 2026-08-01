"""PINN Monorepo Dashboard — run browser, loss explorer, and parametric predictor.

Launch:  uv run streamlit run dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

OUTPUTS_ROOT = Path("outputs")
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_runs() -> dict[str, list[Path]]:
    """Return {experiment_name: [run_dir, ...]} sorted newest-first."""
    experiments: dict[str, list[Path]] = {}
    if not OUTPUTS_ROOT.exists():
        return experiments
    for exp_dir in sorted(OUTPUTS_ROOT.iterdir()):
        if not exp_dir.is_dir():
            continue
        runs = sorted(
            (d for d in exp_dir.iterdir() if d.is_dir() and (d / "metrics.json").exists()),
            reverse=True,
        )
        if runs:
            experiments[exp_dir.name] = runs
    return experiments


def load_metrics(run_dir: Path) -> dict:
    """Load metrics.json from a run directory."""
    path = run_dir / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def load_loss_history(run_dir: Path) -> list[dict[str, float]] | None:
    """Load loss history from checkpoint."""
    ckpt_path = run_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("loss_history")


def fig_to_streamlit(fig):
    """Display a matplotlib figure in Streamlit and close it."""
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview():
    """Run browser — see all experiments and their runs."""
    st.header("Run Browser")

    experiments = discover_runs()
    if not experiments:
        st.warning("No runs found. Train an experiment first: `uv run train-harmonic train -e 100 --no-show`")
        return

    st.markdown(f"**{sum(len(r) for r in experiments.values())} runs** across **{len(experiments)} experiments**")

    for exp_name, runs in experiments.items():
        with st.expander(f"{exp_name} ({len(runs)} runs)", expanded=False):
            for run_dir in runs:
                data = load_metrics(run_dir)
                config = data.get("config", {})
                metrics = data.get("metrics", {})

                cols = st.columns([2, 1, 1, 2])
                cols[0].markdown(f"**`{run_dir.name}`**")
                cols[1].markdown(f"Epochs: {config.get('epochs', '?')}")
                cols[2].markdown(f"Seed: {config.get('seed', '?')}")

                loss = metrics.get("final_total_loss")
                if loss is not None:
                    cols[3].markdown(f"Loss: `{loss:.4e}`")

                # Show key experiment-specific metrics
                interesting = {k: v for k, v in metrics.items()
                               if k not in ("final_total_loss", "epochs_run") and isinstance(v, float)}
                if interesting:
                    metric_str = " | ".join(f"{k}: `{v:.4e}`" for k, v in list(interesting.items())[:4])
                    st.caption(metric_str)
                st.divider()


def page_run_detail():
    """Detailed view of a single run — config, metrics, loss curves, artifacts."""
    st.header("Run Detail")

    experiments = discover_runs()
    if not experiments:
        st.warning("No runs found.")
        return

    col1, col2 = st.columns(2)
    exp_name = col1.selectbox("Experiment", list(experiments.keys()))
    runs = experiments[exp_name]
    run_dir = col2.selectbox("Run", runs, format_func=lambda p: p.name)

    data = load_metrics(run_dir)
    config = data.get("config", {})
    metrics = data.get("metrics", {})

    # --- Config ---
    st.subheader("Configuration")
    config_cols = st.columns(min(len(config), 4) or 1)
    for i, (key, val) in enumerate(config.items()):
        config_cols[i % len(config_cols)].metric(key, val)

    # --- Metrics ---
    st.subheader("Metrics")
    float_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
    if float_metrics:
        metric_cols = st.columns(min(len(float_metrics), 4))
        for i, (key, val) in enumerate(float_metrics.items()):
            metric_cols[i % len(metric_cols)].metric(key, f"{val:.6e}")

    # --- Loss History ---
    st.subheader("Loss History")
    history = load_loss_history(run_dir)
    if history:
        fig, ax = plt.subplots(figsize=(10, 5))
        loss_keys = [k for k in history[0] if k != "total"]
        for key in ["total", *loss_keys]:
            values = [h[key] for h in history]
            ax.plot(values, label=key, linewidth=1.2, alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss History — {exp_name}/{run_dir.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig_to_streamlit(fig)
    else:
        st.info("No loss history found in checkpoint.")

    # --- Artifacts ---
    st.subheader("Artifacts")
    artifacts = sorted(run_dir.glob("*"))
    for a in artifacts:
        if a.is_file():
            size_kb = a.stat().st_size / 1024
            st.text(f"  {a.name}  ({size_kb:.1f} KB)")
        elif a.is_dir():
            n_files = len(list(a.iterdir()))
            st.text(f"  {a.name}/  ({n_files} files)")

    # --- Show plots if they exist ---
    st.subheader("Plots")
    png_files = sorted(run_dir.glob("*.png"))
    if png_files:
        for png in png_files:
            st.image(str(png), caption=png.name, width="stretch")
    else:
        st.info("No plot images found.")


def page_compare():
    """Compare runs across experiments — side-by-side loss curves and metrics."""
    st.header("Run Comparison")

    experiments = discover_runs()
    if not experiments:
        st.warning("No runs found.")
        return

    exp_name = st.selectbox("Experiment", list(experiments.keys()), key="compare_exp")
    runs = experiments[exp_name]

    if len(runs) < 2:
        st.info("Need at least 2 runs to compare. Train more runs first.")

    selected = st.multiselect(
        "Select runs to compare",
        runs,
        default=runs[:min(5, len(runs))],
        format_func=lambda p: p.name,
    )

    if not selected:
        return

    # --- Metrics comparison table ---
    st.subheader("Metrics")
    all_data = []
    for run_dir in selected:
        data = load_metrics(run_dir)
        row = {"run": run_dir.name}
        row.update(data.get("config", {}))
        row.update(data.get("metrics", {}))
        all_data.append(row)

    st.dataframe(all_data, width="stretch")

    # --- Overlaid loss curves ---
    st.subheader("Loss Curves")
    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False
    for run_dir in selected:
        history = load_loss_history(run_dir)
        if history:
            totals = [h["total"] for h in history]
            ax.plot(totals, label=run_dir.name, linewidth=1.2, alpha=0.8)
            has_data = True

    if has_data:
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title(f"Loss Comparison — {exp_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig_to_streamlit(fig)
    else:
        plt.close(fig)
        st.info("No loss histories found in checkpoints.")


def page_parametric():
    """Interactive parametric prediction — slide nu/Re and see the field update."""
    st.header("Parametric Predictor")

    # Detect which parametric experiments have runs
    experiments = discover_runs()
    parametric_exps = {k: v for k, v in experiments.items() if k.startswith("parametric_")}

    if not parametric_exps:
        st.warning(
            "No parametric experiment runs found. Train one first:\n\n"
            "```\nuv run train-parametric-tg train -e 5000 --no-show\n```"
        )
        return

    exp_name = st.selectbox("Parametric Experiment", list(parametric_exps.keys()), key="param_exp")
    runs = parametric_exps[exp_name]
    run_dir = st.selectbox("Run", runs, format_func=lambda p: p.name, key="param_run")

    data = load_metrics(run_dir)
    config = data.get("config", {})

    # --- Experiment-specific parameter sliders ---
    if exp_name == "parametric_taylor_green":
        _parametric_taylor_green(run_dir, config)
    elif exp_name == "parametric_burgers":
        _parametric_burgers(run_dir, config)
    elif exp_name == "parametric_harmonic":
        _parametric_harmonic(run_dir, config)
    elif exp_name == "parametric_schrodinger":
        _parametric_schrodinger(run_dir, config)
    else:
        st.info(f"Interactive prediction not yet implemented for {exp_name}.")


def _load_parametric_models(run_dir: Path, build_model_fn, config: dict):
    """Load all ensemble members from a run directory."""
    models = []
    ckpts = sorted(run_dir.glob("checkpoint*.pt"))
    # Ensure checkpoint.pt is first
    main_ckpt = run_dir / "checkpoint.pt"
    if main_ckpt.exists():
        other_ckpts = [c for c in ckpts if c.name != "checkpoint.pt"]
        ckpts = [main_ckpt, *other_ckpts]

    for path in ckpts:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model = build_model_fn(ckpt.get("metadata", config))
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models.append(model)
    return models


@st.cache_resource
def _cached_load_models(run_dir_str: str, exp_name: str):
    """Cache model loading across reruns."""
    run_dir = Path(run_dir_str)
    if exp_name == "parametric_taylor_green":
        from experiments.parametric_taylor_green.train import build_model
    elif exp_name == "parametric_burgers":
        from experiments.parametric_burgers.train import build_model
    elif exp_name == "parametric_harmonic":
        from experiments.parametric_harmonic.train import build_model
    elif exp_name == "parametric_schrodinger":
        from experiments.parametric_schrodinger.train import build_model
    else:
        return []

    data = load_metrics(run_dir)
    config = data.get("config", {})
    return _load_parametric_models(run_dir, build_model, config)


def _parametric_taylor_green(run_dir: Path, config: dict):
    """Interactive Taylor-Green parametric predictor."""
    nu_range = config.get("nu_range", [0.001, 0.1])

    col1, col2 = st.columns(2)
    nu = col1.slider(
        "Viscosity (nu)", min_value=nu_range[0], max_value=nu_range[1],
        value=0.01, format="%.4f", step=0.001,
    )
    re = 1.0 / nu
    col2.metric("Reynolds Number", f"{re:.0f}")

    models = _cached_load_models(str(run_dir), "parametric_taylor_green")
    if not models:
        st.error("Failed to load models.")
        return

    st.info(f"Loaded {len(models)} ensemble member(s). Computing prediction...")

    from experiments.parametric_taylor_green.train import (
        ensemble_predict_grid,
        exact_taylor_green,
    )

    n_xy = 40
    arrays = ensemble_predict_grid(models, nu, n_xy=n_xy, n_t=40, device=DEVICE)
    x_np, y_np, t_np = arrays["X"], arrays["Y"], arrays["T"]
    u_exact, v_exact, p_exact = exact_taylor_green(x_np, y_np, t_np, nu)

    # Velocity error
    vel_err = np.sqrt(np.sum((arrays["u_mean"] - u_exact) ** 2 + (arrays["v_mean"] - v_exact) ** 2))
    vel_ref = np.sqrt(np.sum(u_exact**2 + v_exact**2))
    rel_l2_vel = vel_err / vel_ref if vel_ref > 0 else vel_err

    st.metric("Velocity Rel-L2 Error", f"{rel_l2_vel:.4e}")

    # Plot at t = T/2
    xy = arrays["xy"]
    X, Y = np.meshgrid(xy, xy, indexing="ij")
    t_mid = arrays["T"].shape[2] // 2

    speed_pred = np.sqrt(arrays["u_mean"][:, :, t_mid] ** 2 + arrays["v_mean"][:, :, t_mid] ** 2)
    speed_exact = np.sqrt(u_exact[:, :, t_mid] ** 2 + v_exact[:, :, t_mid] ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmax = max(speed_exact.max(), speed_pred.max())

    im0 = axes[0].contourf(X, Y, speed_exact, levels=20, vmin=0, vmax=vmax)
    axes[0].set_title("Exact |vel|")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, Y, speed_pred, levels=20, vmin=0, vmax=vmax)
    axes[1].set_title("PINN |vel|")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(X, Y, np.abs(speed_pred - speed_exact), levels=20)
    axes[2].set_title("Error")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    plt.suptitle(f"Taylor-Green at nu={nu:.4f} (Re={re:.0f}), t = T/2", fontsize=13)
    plt.tight_layout()
    fig_to_streamlit(fig)


def _parametric_burgers(run_dir: Path, config: dict):
    """Interactive Burgers parametric predictor."""
    nu_range = config.get("nu_range", [0.01 / np.pi, 0.1])

    nu = st.slider(
        "Viscosity (nu)", min_value=float(nu_range[0]), max_value=float(nu_range[1]),
        value=0.03, format="%.4f", step=0.005,
    )

    models = _cached_load_models(str(run_dir), "parametric_burgers")
    if not models:
        st.error("Failed to load models.")
        return

    from experiments.parametric_burgers.train import ensemble_predict_grid

    arrays = ensemble_predict_grid(models, nu, n_x=200, n_t=100, device=DEVICE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im = axes[0].contourf(arrays["T"], arrays["X"], arrays["u_mean"], 20, cmap="viridis")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    axes[0].set_title(f"u(x, t) at nu={nu:.4f}")
    plt.colorbar(im, ax=axes[0])

    x = arrays["x"][:, 0]
    u_exact_0 = -np.sin(np.pi * x)
    axes[1].plot(x, u_exact_0, "k-", lw=2, label="Exact IC")
    axes[1].plot(x, arrays["u_mean_0"][:, 0], "r--", lw=2, label="PINN t=0")
    axes[1].plot(x, arrays["u_mean_1"][:, 0], "b--", lw=2, label="PINN t=1")
    if len(models) > 1:
        axes[1].fill_between(
            x,
            arrays["u_mean_1"][:, 0] - 2 * arrays["u_std_1"][:, 0],
            arrays["u_mean_1"][:, 0] + 2 * arrays["u_std_1"][:, 0],
            alpha=0.3, color="orange", label="+/- 2 sigma",
        )
    axes[1].legend()
    axes[1].set_title("Snapshots")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_to_streamlit(fig)


def _parametric_harmonic(run_dir: Path, config: dict):
    """Interactive harmonic oscillator parametric predictor."""
    col1, col2 = st.columns(2)
    w0 = col1.slider("Frequency (w0)", min_value=20.0, max_value=100.0, value=50.0, step=5.0)
    d = col2.slider("Damping (d)", min_value=0.1, max_value=4.0, value=1.5, step=0.1)

    models = _cached_load_models(str(run_dir), "parametric_harmonic")
    if not models:
        st.error("Failed to load models.")
        return

    t = torch.linspace(0, 1, 300).view(-1, 1)
    w0_t = torch.full_like(t, w0)
    d_t = torch.full_like(t, d)

    preds = []
    with torch.no_grad():
        for m in models:
            preds.append(m(t, w0_t, d_t).numpy().flatten())

    preds = np.array(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)

    # Exact solution
    w = np.sqrt(w0**2 - d**2) if w0 > d else 0.0
    t_np = t.numpy().flatten()
    u_exact = np.exp(-d * t_np) * np.cos(w * t_np) if w > 0 else np.exp(-d * t_np)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_np, u_exact, "k-", lw=2, label="Exact")
    ax.plot(t_np, mean, "r--", lw=2, label="PINN mean")
    if len(models) > 1:
        ax.fill_between(t_np, mean - 2 * std, mean + 2 * std,
                         alpha=0.3, color="orange", label="+/- 2 sigma")
    ax.set_xlabel("t")
    ax.set_ylabel("u")
    ax.set_title(f"Harmonic Oscillator: w0={w0:.0f}, d={d:.1f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_to_streamlit(fig)


def _parametric_schrodinger(run_dir: Path, config: dict):
    """Interactive Schrodinger parametric predictor."""
    a_range = config.get("a_range", [0.75, 2.0])
    A = st.slider("Amplitude (A)", min_value=float(a_range[0]), max_value=float(a_range[1]),
                   value=1.0, step=0.05)

    models = _cached_load_models(str(run_dir), "parametric_schrodinger")
    if not models:
        st.error("Failed to load models.")
        return

    from experiments.parametric_schrodinger.train import ensemble_predict_grid

    arrays = ensemble_predict_grid(models, A, n_x=200, n_t=100, device=DEVICE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im = axes[0].contourf(arrays["T"], arrays["X"], arrays["h_mag_mean"], 20, cmap="viridis")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    axes[0].set_title(f"|h(x,t)| at A={A:.2f}")
    plt.colorbar(im, ax=axes[0])

    x = arrays["x"][:, 0]
    axes[1].plot(x, arrays["h_mag_mean"][:, 0], "r-", lw=2, label="t=0")
    axes[1].plot(x, arrays["h_mag_mean"][:, -1], "b--", lw=2, label="t=T")
    exact_0 = A / np.cosh(A * x)
    axes[1].plot(x, exact_0, "k:", lw=2, label="Exact t=0")
    if len(models) > 1:
        axes[1].fill_between(
            x,
            arrays["h_mag_mean"][:, 0] - 2 * arrays["h_mag_std"][:, 0],
            arrays["h_mag_mean"][:, 0] + 2 * arrays["h_mag_std"][:, 0],
            alpha=0.3, color="orange", label="+/- 2 sigma",
        )
    axes[1].legend()
    axes[1].set_title("Snapshots")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_to_streamlit(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.set_page_config(page_title="PINN Dashboard", page_icon="🔬", layout="wide")
st.title("PINN Monorepo Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Run Detail", "Compare", "Parametric Predictor"],
)

if page == "Overview":
    page_overview()
elif page == "Run Detail":
    page_run_detail()
elif page == "Compare":
    page_compare()
elif page == "Parametric Predictor":
    page_parametric()
