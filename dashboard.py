"""PINN Monorepo Dashboard — run browser, loss explorer, and parametric predictor.

Launch:  uv run streamlit run dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

OUTPUTS_ROOT = Path("outputs")
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Theme & styling
# ---------------------------------------------------------------------------

# Dark matplotlib theme for consistent look
plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#1A1D23",
    "axes.edgecolor": "#3D4150",
    "axes.labelcolor": "#FAFAFA",
    "text.color": "#FAFAFA",
    "xtick.color": "#A0A4B0",
    "ytick.color": "#A0A4B0",
    "grid.color": "#2D3040",
    "legend.facecolor": "#1A1D23",
    "legend.edgecolor": "#3D4150",
    "legend.labelcolor": "#FAFAFA",
})

# Plotly dark template
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#1A1D23",
    font=dict(color="#FAFAFA", family="Inter, system-ui, sans-serif"),
    margin=dict(l=50, r=30, t=50, b=40),
    legend=dict(bgcolor="rgba(26,29,35,0.8)", bordercolor="#3D4150", borderwidth=1),
)

ACCENT = "#FF6B6B"
ACCENT_BLUE = "#4ECDC4"
ACCENT_YELLOW = "#FFE66D"
ACCENT_PURPLE = "#A78BFA"

CUSTOM_CSS = """
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #161B22 100%);
    }
    section[data-testid="stSidebar"] hr {
        border-color: #21262D;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1D23 0%, #21262D 100%);
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #8B949E !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-weight: 600;
    }

    /* Expander styling */
    div[data-testid="stExpander"] {
        border: 1px solid #21262D;
        border-radius: 8px;
        background: #161B22;
    }

    /* Headers */
    h1 {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    h2 {
        color: #E6EDF3 !important;
        border-bottom: 2px solid #21262D;
        padding-bottom: 8px;
    }

    /* Container borders */
    div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
        border-radius: 10px;
    }

    /* Subtle dividers */
    hr {
        border-color: #21262D !important;
    }

    /* Table/dataframe */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
"""

PAGE_ICONS = {
    "Overview": "home",
    "Run Detail": "search",
    "Compare": "bar_chart",
    "Parametric Predictor": "tune",
}


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


def format_loss(val: float) -> str:
    """Format a loss value with color indicator."""
    if val < 1e-4:
        return f"🟢 `{val:.4e}`"
    if val < 1e-2:
        return f"🟡 `{val:.4e}`"
    return f"🔴 `{val:.4e}`"


def format_experiment_name(name: str) -> str:
    """Turn snake_case into a readable title."""
    return name.replace("_", " ").title()


def plotly_loss_chart(
    histories: dict[str, list[dict[str, float]]],
    title: str = "Loss History",
    show_components: bool = True,
) -> go.Figure:
    """Build an interactive Plotly loss chart from one or more run histories."""
    colors = [ACCENT, ACCENT_BLUE, ACCENT_YELLOW, ACCENT_PURPLE, "#F97583", "#79C0FF"]
    fig = go.Figure()

    for i, (label, history) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        epochs = list(range(1, len(history) + 1))

        # Total loss — always visible
        fig.add_trace(go.Scatter(
            x=epochs,
            y=[h["total"] for h in history],
            name=f"{label} — total" if len(histories) > 1 else "total",
            line=dict(color=color, width=2.5),
            hovertemplate="Epoch %{x}<br>Loss: %{y:.4e}<extra></extra>",
        ))

        # Component losses — initially hidden for multi-run, visible for single
        if show_components and history:
            component_keys = [k for k in history[0] if k != "total"]
            for key in component_keys:
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=[h[key] for h in history],
                    name=f"{label} — {key}" if len(histories) > 1 else key,
                    line=dict(color=color, width=1.2, dash="dot"),
                    opacity=0.6,
                    visible="legendonly" if len(histories) > 1 else True,
                    hovertemplate=f"{key}<br>Epoch %{{x}}<br>Loss: %{{y:.4e}}<extra></extra>",
                ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Epoch", gridcolor="#2D3040"),
        yaxis=dict(title="Loss", type="log", gridcolor="#2D3040"),
        height=450,
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview():
    """Run browser — see all experiments and their runs."""
    experiments = discover_runs()
    if not experiments:
        st.warning(
            "No runs found. Train an experiment first:\n\n"
            "```bash\nuv run train-harmonic train -e 500 --no-show\n```"
        )
        return

    total_runs = sum(len(r) for r in experiments.values())
    total_exps = len(experiments)

    # Summary bar
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experiments", total_exps)
    c2.metric("Total Runs", total_runs)

    # Best overall loss
    best_loss = float("inf")
    best_exp = ""
    for exp_name, runs in experiments.items():
        for run_dir in runs:
            data = load_metrics(run_dir)
            loss = data.get("metrics", {}).get("final_total_loss")
            if loss is not None and loss < best_loss:
                best_loss = loss
                best_exp = exp_name
    if best_loss < float("inf"):
        c3.metric("Best Loss", f"{best_loss:.4e}")
        c4.metric("Best Experiment", format_experiment_name(best_exp))

    st.markdown("---")

    # Experiment cards
    for exp_name, runs in experiments.items():
        with st.expander(
            f":material/{_exp_icon(exp_name)}: **{format_experiment_name(exp_name)}** — {len(runs)} run{'s' if len(runs) != 1 else ''}",
            expanded=len(experiments) <= 3,
        ):
            for run_dir in runs:
                data = load_metrics(run_dir)
                config = data.get("config", {})
                metrics = data.get("metrics", {})

                with st.container(border=True):
                    cols = st.columns([3, 1, 1, 2])
                    cols[0].markdown(f"**`{run_dir.name}`**")
                    cols[1].caption(f"Epochs: **{config.get('epochs', '?')}**")
                    cols[2].caption(f"Seed: **{config.get('seed', '?')}**")

                    loss = metrics.get("final_total_loss")
                    if loss is not None:
                        cols[3].markdown(format_loss(loss))

                    # Key experiment-specific metrics
                    interesting = {
                        k: v for k, v in metrics.items()
                        if k not in ("final_total_loss", "epochs_run") and isinstance(v, float)
                    }
                    if interesting:
                        metric_str = "  ".join(
                            f"`{k}` = {v:.4e}" for k, v in list(interesting.items())[:4]
                        )
                        st.caption(metric_str)


def _exp_icon(name: str) -> str:
    """Return a Material icon name for an experiment."""
    if "harmonic" in name:
        return "waves"
    if "burgers" in name:
        return "local_fire_department"
    if "schrodinger" in name:
        return "science"
    if "taylor_green" in name or "cavity" in name or "cylinder" in name or "navier" in name:
        return "water_drop"
    return "experiment"


def page_run_detail():
    """Detailed view of a single run — config, metrics, loss curves, artifacts."""
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
    with st.container(border=True):
        n_cols = min(len(config), 4) or 1
        config_cols = st.columns(n_cols)
        for i, (key, val) in enumerate(config.items()):
            config_cols[i % n_cols].metric(key, val)

    # --- Metrics ---
    st.subheader("Metrics")
    float_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
    if float_metrics:
        with st.container(border=True):
            n_cols = min(len(float_metrics), 4)
            metric_cols = st.columns(n_cols)
            for i, (key, val) in enumerate(float_metrics.items()):
                metric_cols[i % n_cols].metric(key, f"{val:.6e}")

    # --- Loss History (Plotly) ---
    st.subheader("Loss History")
    history = load_loss_history(run_dir)
    if history:
        fig = plotly_loss_chart(
            {run_dir.name: history},
            title=f"Loss History — {format_experiment_name(exp_name)}",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No loss history found in checkpoint.")

    # --- Artifacts + Plots side by side ---
    art_col, plot_col = st.columns([1, 2])

    with art_col:
        st.subheader("Artifacts")
        with st.container(border=True):
            artifacts = sorted(run_dir.glob("*"))
            for a in artifacts:
                if a.is_file():
                    size_kb = a.stat().st_size / 1024
                    icon = _file_icon(a.name)
                    st.markdown(f":{icon}: `{a.name}` — {size_kb:.1f} KB")
                elif a.is_dir():
                    n_files = len(list(a.iterdir()))
                    st.markdown(f":file_folder: `{a.name}/` — {n_files} files")

    with plot_col:
        st.subheader("Plots")
        png_files = sorted(run_dir.glob("*.png"))
        if png_files:
            # Show plots in a 2-column grid
            img_cols = st.columns(2)
            for i, png in enumerate(png_files):
                img_cols[i % 2].image(str(png), caption=png.name, width="stretch")
        else:
            st.info("No plot images found.")


def _file_icon(name: str) -> str:
    """Return an emoji for a file type."""
    if name.endswith(".pt"):
        return "floppy_disk"
    if name.endswith(".json"):
        return "page_facing_up"
    if name.endswith(".png"):
        return "frame_with_picture"
    if name.endswith(".npz"):
        return "package"
    if name.endswith(".log"):
        return "scroll"
    return "page_facing_up"


def page_compare():
    """Compare runs across experiments — side-by-side loss curves and metrics."""
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

    # --- Overlaid loss curves (Plotly) ---
    st.subheader("Loss Curves")
    histories = {}
    for run_dir in selected:
        history = load_loss_history(run_dir)
        if history:
            histories[run_dir.name] = history

    if histories:
        fig = plotly_loss_chart(
            histories,
            title=f"Loss Comparison — {format_experiment_name(exp_name)}",
            show_components=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No loss histories found in checkpoints.")


def page_parametric():
    """Interactive parametric prediction — slide nu/Re and see the field update."""
    st.markdown(
        "Drag a slider to solve at a **never-trained** parameter value. "
        "The model runs inference live — no retraining needed."
    )

    # Detect which parametric experiments have runs
    experiments = discover_runs()
    parametric_exps = {k: v for k, v in experiments.items() if k.startswith("parametric_")}

    if not parametric_exps:
        st.warning(
            "No parametric experiment runs found. Train one first:\n\n"
            "```bash\nuv run train-parametric train -e 5000 --no-show\n```"
        )
        return

    col1, col2 = st.columns(2)
    exp_name = col1.selectbox("Parametric Experiment", list(parametric_exps.keys()), key="param_exp")
    runs = parametric_exps[exp_name]
    run_dir = col2.selectbox("Run", runs, format_func=lambda p: p.name, key="param_run")

    data = load_metrics(run_dir)
    config = data.get("config", {})

    st.markdown("---")

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

    with st.spinner("Loading model and computing prediction..."):
        models = _cached_load_models(str(run_dir), "parametric_taylor_green")
        if not models:
            st.error("Failed to load models.")
            return

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

    m1, m2, m3 = st.columns(3)
    m1.metric("Velocity Rel-L2 Error", f"{rel_l2_vel:.4e}")
    m2.metric("Ensemble Members", len(models))
    m3.metric("Grid Resolution", f"{n_xy} x {n_xy}")

    # Plot at t = T/2
    xy = arrays["xy"]
    X, Y = np.meshgrid(xy, xy, indexing="ij")
    t_mid = arrays["T"].shape[2] // 2

    speed_pred = np.sqrt(arrays["u_mean"][:, :, t_mid] ** 2 + arrays["v_mean"][:, :, t_mid] ** 2)
    speed_exact = np.sqrt(u_exact[:, :, t_mid] ** 2 + v_exact[:, :, t_mid] ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    vmax = max(speed_exact.max(), speed_pred.max())

    cmap = "inferno"
    im0 = axes[0].contourf(X, Y, speed_exact, levels=20, vmin=0, vmax=vmax, cmap=cmap)
    axes[0].set_title("Exact |vel|", fontsize=12, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].contourf(X, Y, speed_pred, levels=20, vmin=0, vmax=vmax, cmap=cmap)
    axes[1].set_title("PINN |vel|", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], shrink=0.85)

    im2 = axes[2].contourf(X, Y, np.abs(speed_pred - speed_exact), levels=20, cmap="magma")
    axes[2].set_title("Pointwise Error", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], shrink=0.85)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    plt.suptitle(
        f"Taylor-Green Vortex  |  nu = {nu:.4f}  (Re = {re:.0f})  |  t = T/2",
        fontsize=14, fontweight="bold", color=ACCENT,
    )
    plt.tight_layout()
    fig_to_streamlit(fig)


def _parametric_burgers(run_dir: Path, config: dict):
    """Interactive Burgers parametric predictor."""
    nu_range = config.get("nu_range", [0.01 / np.pi, 0.1])

    nu = st.slider(
        "Viscosity (nu)", min_value=float(nu_range[0]), max_value=float(nu_range[1]),
        value=0.03, format="%.4f", step=0.005,
    )

    with st.spinner("Loading model and computing prediction..."):
        models = _cached_load_models(str(run_dir), "parametric_burgers")
        if not models:
            st.error("Failed to load models.")
            return

        from experiments.parametric_burgers.train import ensemble_predict_grid

        arrays = ensemble_predict_grid(models, nu, n_x=200, n_t=100, device=DEVICE)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].contourf(arrays["T"], arrays["X"], arrays["u_mean"], 20, cmap="inferno")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    axes[0].set_title(f"u(x, t) at nu = {nu:.4f}", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=axes[0], shrink=0.85)

    x = arrays["x"][:, 0]
    u_exact_0 = -np.sin(np.pi * x)
    axes[1].plot(x, u_exact_0, color="#FAFAFA", lw=2.5, label="Exact IC")
    axes[1].plot(x, arrays["u_mean_0"][:, 0], color=ACCENT, lw=2, ls="--", label="PINN t=0")
    axes[1].plot(x, arrays["u_mean_1"][:, 0], color=ACCENT_BLUE, lw=2, ls="--", label="PINN t=1")
    if len(models) > 1:
        axes[1].fill_between(
            x,
            arrays["u_mean_1"][:, 0] - 2 * arrays["u_std_1"][:, 0],
            arrays["u_mean_1"][:, 0] + 2 * arrays["u_std_1"][:, 0],
            alpha=0.25, color=ACCENT_YELLOW, label="+/- 2 sigma",
        )
    axes[1].legend(facecolor="#1A1D23", edgecolor="#3D4150")
    axes[1].set_title("Snapshots", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.2)

    plt.suptitle(
        f"Burgers' Equation  |  nu = {nu:.4f}",
        fontsize=14, fontweight="bold", color=ACCENT,
    )
    plt.tight_layout()
    fig_to_streamlit(fig)


def _parametric_harmonic(run_dir: Path, config: dict):
    """Interactive harmonic oscillator parametric predictor."""
    col1, col2, col3 = st.columns(3)
    w0 = col1.slider("Frequency (w0)", min_value=20.0, max_value=100.0, value=50.0, step=5.0)
    d = col2.slider("Damping (d)", min_value=0.1, max_value=4.0, value=1.5, step=0.1)

    regime = "Underdamped" if w0 > d else ("Critical" if w0 == d else "Overdamped")
    col3.metric("Regime", regime)

    with st.spinner("Loading model and computing prediction..."):
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

    # Plotly interactive chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_np, y=u_exact, name="Exact",
        line=dict(color="#FAFAFA", width=2.5),
        hovertemplate="t=%{x:.3f}<br>u=%{y:.4f}<extra>Exact</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t_np, y=mean, name="PINN mean",
        line=dict(color=ACCENT, width=2, dash="dash"),
        hovertemplate="t=%{x:.3f}<br>u=%{y:.4f}<extra>PINN</extra>",
    ))
    if len(models) > 1:
        fig.add_trace(go.Scatter(
            x=np.concatenate([t_np, t_np[::-1]]),
            y=np.concatenate([mean + 2 * std, (mean - 2 * std)[::-1]]),
            fill="toself", fillcolor="rgba(255, 230, 109, 0.2)",
            line=dict(width=0), name="+/- 2 sigma",
            hoverinfo="skip",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"Harmonic Oscillator  |  w0 = {w0:.0f}, d = {d:.1f}  |  {regime}",
            font=dict(size=16, color=ACCENT),
        ),
        xaxis=dict(title="t", gridcolor="#2D3040"),
        yaxis=dict(title="u(t)", gridcolor="#2D3040"),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)


def _parametric_schrodinger(run_dir: Path, config: dict):
    """Interactive Schrodinger parametric predictor."""
    a_range = config.get("a_range", [0.75, 2.0])
    A = st.slider(
        "Amplitude (A)", min_value=float(a_range[0]), max_value=float(a_range[1]),
        value=1.0, step=0.05,
    )

    with st.spinner("Loading model and computing prediction..."):
        models = _cached_load_models(str(run_dir), "parametric_schrodinger")
        if not models:
            st.error("Failed to load models.")
            return

        from experiments.parametric_schrodinger.train import ensemble_predict_grid

        arrays = ensemble_predict_grid(models, A, n_x=200, n_t=100, device=DEVICE)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].contourf(arrays["T"], arrays["X"], arrays["h_mag_mean"], 20, cmap="inferno")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    axes[0].set_title(f"|h(x,t)| at A = {A:.2f}", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=axes[0], shrink=0.85)

    x = arrays["x"][:, 0]
    axes[1].plot(x, arrays["h_mag_mean"][:, 0], color=ACCENT, lw=2, label="PINN t=0")
    axes[1].plot(x, arrays["h_mag_mean"][:, -1], color=ACCENT_BLUE, lw=2, ls="--", label="PINN t=T")
    exact_0 = A / np.cosh(A * x)
    axes[1].plot(x, exact_0, color="#FAFAFA", lw=2, ls=":", label="Exact t=0")
    if len(models) > 1:
        axes[1].fill_between(
            x,
            arrays["h_mag_mean"][:, 0] - 2 * arrays["h_mag_std"][:, 0],
            arrays["h_mag_mean"][:, 0] + 2 * arrays["h_mag_std"][:, 0],
            alpha=0.25, color=ACCENT_YELLOW, label="+/- 2 sigma",
        )
    axes[1].legend(facecolor="#1A1D23", edgecolor="#3D4150")
    axes[1].set_title("Soliton Profile", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.2)

    plt.suptitle(
        f"Schrodinger Soliton  |  A = {A:.2f}",
        fontsize=14, fontweight="bold", color=ACCENT,
    )
    plt.tight_layout()
    fig_to_streamlit(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PINN Dashboard",
    page_icon=":material/science:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## :material/science: PINN Dashboard")
    st.caption("Physics-Informed Neural Networks")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            ":material/home: Overview",
            ":material/search: Run Detail",
            ":material/bar_chart: Compare",
            ":material/tune: Parametric Predictor",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Sidebar stats
    experiments = discover_runs()
    if experiments:
        total_runs = sum(len(r) for r in experiments.values())
        st.caption(f"{len(experiments)} experiments  ·  {total_runs} runs")
    else:
        st.caption("No runs yet")

# --- Page title ---
page_clean = page.split(": ", 1)[-1] if ": " in page else page
st.title(page_clean)

# --- Route ---
if "Overview" in page:
    page_overview()
elif "Run Detail" in page:
    page_run_detail()
elif "Compare" in page:
    page_compare()
elif "Parametric" in page:
    page_parametric()
