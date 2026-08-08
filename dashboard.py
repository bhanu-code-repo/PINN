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
# Brand colors & light theme
# ---------------------------------------------------------------------------

BRAND = "#00205B"
BRAND_LIGHT = "#003380"
BRAND_LIGHTER = "#E8EDF5"
BRAND_SUBTLE = "#F4F6FA"

CHART_COLORS = [BRAND, "#0066CC", "#E85D04", "#2D8B4E", "#9B2C8A", "#C4820E"]

# Matplotlib — light theme
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#F8F9FB",
    "axes.edgecolor": "#D0D7E2",
    "axes.labelcolor": "#1A1A2E",
    "text.color": "#1A1A2E",
    "xtick.color": "#5A6277",
    "ytick.color": "#5A6277",
    "grid.color": "#E8EDF5",
    "legend.facecolor": "#FFFFFF",
    "legend.edgecolor": "#D0D7E2",
    "legend.labelcolor": "#1A1A2E",
})

# Plotly layout defaults
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#1A1A2E", family="Inter, system-ui, sans-serif"),
    margin=dict(l=50, r=30, t=50, b=40),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#D0D7E2", borderwidth=1),
)

CUSTOM_CSS = f"""
<style>
    /* Sidebar — brand navy gradient */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BRAND} 0%, {BRAND_LIGHT} 100%);
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.15);
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
        font-size: 1.05rem !important;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: linear-gradient(135deg, {BRAND_LIGHTER} 0%, {BRAND_SUBTLE} 100%);
        border: 1px solid #D0D7E2;
        border-radius: 10px;
        padding: 12px 16px;
    }}
    div[data-testid="stMetric"] label {{
        color: #5A6277 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: #1A1A2E !important;
        font-weight: 600;
    }}

    /* Expander */
    div[data-testid="stExpander"] {{
        border: 1px solid #D0D7E2;
        border-radius: 8px;
        background: #F8F9FB;
    }}

    /* Headers */
    h1 {{ color: {BRAND} !important; font-weight: 800 !important; }}
    h2 {{
        color: {BRAND} !important;
        border-bottom: 2px solid #D0D7E2;
        padding-bottom: 8px;
    }}

    /* Containers */
    div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {{
        border-radius: 10px;
    }}

    hr {{ border-color: #D0D7E2 !important; }}

    .stDataFrame {{
        border-radius: 8px;
        overflow: hidden;
    }}
</style>
"""


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
    fig = go.Figure()

    for i, (label, history) in enumerate(histories.items()):
        color = CHART_COLORS[i % len(CHART_COLORS)]
        epochs = list(range(1, len(history) + 1))

        fig.add_trace(go.Scatter(
            x=epochs,
            y=[h["total"] for h in history],
            name=f"{label} — total" if len(histories) > 1 else "total",
            line=dict(color=color, width=2.5),
            hovertemplate="Epoch %{x}<br>Loss: %{y:.4e}<extra></extra>",
        ))

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
        title=dict(text=title, font=dict(size=16, color=BRAND)),
        xaxis=dict(title="Epoch", gridcolor="#E8EDF5"),
        yaxis=dict(title="Loss", type="log", gridcolor="#E8EDF5"),
        height=450,
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Plot expand dialog
# ---------------------------------------------------------------------------


@st.dialog("Plot Viewer", width="large")
def _show_plot_dialog(image_path: str):
    """Show an image in a full-width dialog."""
    st.image(image_path, width="stretch")


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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experiments", total_exps)
    c2.metric("Total Runs", total_runs)

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

    st.subheader("Configuration")
    with st.container(border=True):
        n_cols = min(len(config), 4) or 1
        config_cols = st.columns(n_cols)
        for i, (key, val) in enumerate(config.items()):
            display_val = str(val) if isinstance(val, (dict, list)) else val
            config_cols[i % n_cols].metric(key, display_val)

    st.subheader("Metrics")
    float_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
    if float_metrics:
        with st.container(border=True):
            n_cols = min(len(float_metrics), 4)
            metric_cols = st.columns(n_cols)
            for i, (key, val) in enumerate(float_metrics.items()):
                metric_cols[i % n_cols].metric(key, f"{val:.6e}")

    st.subheader("Loss History")
    history = load_loss_history(run_dir)
    if history:
        fig = plotly_loss_chart(
            {run_dir.name: history},
            title=f"Loss History — {format_experiment_name(exp_name)}",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No loss history found in checkpoint.")

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
            img_cols = st.columns(2)
            for i, png in enumerate(png_files):
                with img_cols[i % 2]:
                    st.image(str(png), caption=png.name, width="stretch")
                    if st.button(
                        ":material/open_in_full: Expand",
                        key=f"expand_{png.name}",
                        use_container_width=True,
                    ):
                        _show_plot_dialog(str(png))
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

    st.subheader("Metrics")
    all_data = []
    for run_dir in selected:
        data = load_metrics(run_dir)
        row = {"run": run_dir.name}
        row.update(data.get("config", {}))
        row.update(data.get("metrics", {}))
        all_data.append(row)

    st.dataframe(all_data, width="stretch")

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
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No loss histories found in checkpoints.")


def page_parametric():
    """Interactive parametric prediction — slide nu/Re and see the field update."""
    st.markdown(
        "Drag a slider to solve at a **never-trained** parameter value. "
        "The model runs inference live — no retraining needed."
    )

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

    vel_err = np.sqrt(np.sum((arrays["u_mean"] - u_exact) ** 2 + (arrays["v_mean"] - v_exact) ** 2))
    vel_ref = np.sqrt(np.sum(u_exact**2 + v_exact**2))
    rel_l2_vel = vel_err / vel_ref if vel_ref > 0 else vel_err

    m1, m2, m3 = st.columns(3)
    m1.metric("Velocity Rel-L2 Error", f"{rel_l2_vel:.4e}")
    m2.metric("Ensemble Members", len(models))
    m3.metric("Grid Resolution", f"{n_xy} x {n_xy}")

    xy = arrays["xy"]
    X, Y = np.meshgrid(xy, xy, indexing="ij")
    t_mid = arrays["T"].shape[2] // 2

    speed_pred = np.sqrt(arrays["u_mean"][:, :, t_mid] ** 2 + arrays["v_mean"][:, :, t_mid] ** 2)
    speed_exact = np.sqrt(u_exact[:, :, t_mid] ** 2 + v_exact[:, :, t_mid] ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmax = max(speed_exact.max(), speed_pred.max())

    im0 = axes[0].contourf(X, Y, speed_exact, levels=20, vmin=0, vmax=vmax, cmap="viridis")
    axes[0].set_title("Exact |vel|", fontsize=12, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].contourf(X, Y, speed_pred, levels=20, vmin=0, vmax=vmax, cmap="viridis")
    axes[1].set_title("PINN |vel|", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], shrink=0.85)

    im2 = axes[2].contourf(X, Y, np.abs(speed_pred - speed_exact), levels=20, cmap="Reds")
    axes[2].set_title("Pointwise Error", fontsize=12, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], shrink=0.85)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    plt.suptitle(
        f"Taylor-Green Vortex  |  nu = {nu:.4f}  (Re = {re:.0f})  |  t = T/2",
        fontsize=14, fontweight="bold", color=BRAND,
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

    im = axes[0].contourf(arrays["T"], arrays["X"], arrays["u_mean"], 20, cmap="viridis")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    axes[0].set_title(f"u(x, t) at nu = {nu:.4f}", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=axes[0], shrink=0.85)

    x = arrays["x"][:, 0]
    u_exact_0 = -np.sin(np.pi * x)
    axes[1].plot(x, u_exact_0, color="#1A1A2E", lw=2.5, label="Exact IC")
    axes[1].plot(x, arrays["u_mean_0"][:, 0], color=CHART_COLORS[0], lw=2, ls="--", label="PINN t=0")
    axes[1].plot(x, arrays["u_mean_1"][:, 0], color=CHART_COLORS[2], lw=2, ls="--", label="PINN t=1")
    if len(models) > 1:
        axes[1].fill_between(
            x,
            arrays["u_mean_1"][:, 0] - 2 * arrays["u_std_1"][:, 0],
            arrays["u_mean_1"][:, 0] + 2 * arrays["u_std_1"][:, 0],
            alpha=0.25, color=CHART_COLORS[3], label="+/- 2 sigma",
        )
    axes[1].legend()
    axes[1].set_title("Snapshots", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Burgers' Equation  |  nu = {nu:.4f}",
        fontsize=14, fontweight="bold", color=BRAND,
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

    w = np.sqrt(w0**2 - d**2) if w0 > d else 0.0
    t_np = t.numpy().flatten()
    u_exact = np.exp(-d * t_np) * np.cos(w * t_np) if w > 0 else np.exp(-d * t_np)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_np, y=u_exact, name="Exact",
        line=dict(color="#1A1A2E", width=2.5),
        hovertemplate="t=%{x:.3f}<br>u=%{y:.4f}<extra>Exact</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t_np, y=mean, name="PINN mean",
        line=dict(color=CHART_COLORS[0], width=2, dash="dash"),
        hovertemplate="t=%{x:.3f}<br>u=%{y:.4f}<extra>PINN</extra>",
    ))
    if len(models) > 1:
        fig.add_trace(go.Scatter(
            x=np.concatenate([t_np, t_np[::-1]]),
            y=np.concatenate([mean + 2 * std, (mean - 2 * std)[::-1]]),
            fill="toself", fillcolor="rgba(45,139,78,0.15)",
            line=dict(width=0), name="+/- 2 sigma",
            hoverinfo="skip",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"Harmonic Oscillator  |  w0 = {w0:.0f}, d = {d:.1f}  |  {regime}",
            font=dict(size=16, color=BRAND),
        ),
        xaxis=dict(title="t", gridcolor="#E8EDF5"),
        yaxis=dict(title="u(t)", gridcolor="#E8EDF5"),
        height=480,
    )
    st.plotly_chart(fig, width="stretch")


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
    im = axes[0].contourf(arrays["T"], arrays["X"], arrays["h_mag_mean"], 20, cmap="viridis")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    axes[0].set_title(f"|h(x,t)| at A = {A:.2f}", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=axes[0], shrink=0.85)

    x = arrays["x"][:, 0]
    axes[1].plot(x, arrays["h_mag_mean"][:, 0], color=CHART_COLORS[0], lw=2, label="PINN t=0")
    axes[1].plot(x, arrays["h_mag_mean"][:, -1], color=CHART_COLORS[1], lw=2, ls="--", label="PINN t=T")
    exact_0 = A / np.cosh(A * x)
    axes[1].plot(x, exact_0, color="#1A1A2E", lw=2, ls=":", label="Exact t=0")
    if len(models) > 1:
        axes[1].fill_between(
            x,
            arrays["h_mag_mean"][:, 0] - 2 * arrays["h_mag_std"][:, 0],
            arrays["h_mag_mean"][:, 0] + 2 * arrays["h_mag_std"][:, 0],
            alpha=0.25, color=CHART_COLORS[3], label="+/- 2 sigma",
        )
    axes[1].legend()
    axes[1].set_title("Soliton Profile", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Schrodinger Soliton  |  A = {A:.2f}",
        fontsize=14, fontweight="bold", color=BRAND,
    )
    plt.tight_layout()
    fig_to_streamlit(fig)


def page_lang_pinn():
    """Lang-PINN — describe a PDE in natural language and get a PINN solution."""
    st.markdown(
        "Describe a differential equation in **natural language**. "
        "Lang-PINN will parse it, recommend an architecture, and generate "
        "runnable experiment code — all powered by LLM agents."
    )

    # Mode selector
    col_mode, col_info = st.columns([1, 2])
    mode = col_mode.selectbox(
        "Mode",
        ["library", "code-agent", "hybrid"],
        index=2,
        help=(
            "**library** — PDE parsed by LLM, architecture and code are deterministic.\n\n"
            "**code-agent** — all agents use LLM.\n\n"
            "**hybrid** — LLM generates code targeting the pinn library API, "
            "with iterative feedback refinement."
        ),
    )
    mode_descriptions = {
        "library": "Deterministic rules + templates. Fast, reproducible.",
        "code-agent": "Full LLM pipeline. Flexible, creative.",
        "hybrid": "LLM + feedback loop. Best of both worlds.",
    }
    col_info.info(mode_descriptions[mode])

    # Input
    description = st.text_area(
        "Describe your PDE",
        placeholder=(
            "e.g. 'Solve the damped harmonic oscillator u'' + 2u' + 6400u = 0 "
            "with u(0)=1, u'(0)=0 on [0,1]'"
        ),
        height=100,
    )

    col_btn, col_verify = st.columns([1, 1])
    run_button = col_btn.button(
        ":material/play_arrow: Generate", type="primary",
        disabled=not description.strip(),
        use_container_width=True,
    )
    verify = col_verify.checkbox("SymPy verification", value=True)

    if not run_button:
        # Show example cards
        st.markdown("---")
        st.subheader("Examples")
        examples = [
            ("Exponential Decay (ODE)", "Solve u' + u = 0, u(0) = 1, on [0, 3]"),
            ("Damped Oscillator (ODE)", "u'' + 2u' + 6400u = 0, u(0)=1, u'(0)=0 on [0,1]"),
            ("Burgers (PDE)", "Burgers equation u_t + u*u_x = 0.01*u_xx on x in [-1,1], t in [0,1]"),
            ("Heat Equation (PDE)", "Heat equation u_t = 0.1*u_xx on [0,1] x [0,0.5]"),
        ]
        cols = st.columns(2)
        for i, (title, desc) in enumerate(examples):
            with cols[i % 2], st.container(border=True):
                st.markdown(f"**{title}**")
                st.code(desc, language=None)
        return

    # Run the pipeline
    with st.spinner("Running Lang-PINN pipeline..."):
        try:
            from lang_pinn import Orchestrator
            orch = Orchestrator(mode=mode)
            result = orch.solve(description)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return

    st.markdown("---")

    # PDE Specification
    st.subheader("Parsed PDE Specification")
    spec = result.spec
    with st.container(border=True):
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Name", spec.name)
        sc2.metric("Order", spec.order)
        sc3.metric("Spatial Dim", spec.spatial_dim)
        sc4.metric("Output Dim", spec.output_dim)

        st.markdown(f"**Equation:** `{spec.equation}`")
        st.markdown(f"**Variables:** {' → '.join(spec.independent_vars)} → {spec.dependent_var}")
        st.markdown(f"**Domain:** {spec.domain}")

        if spec.initial_conditions:
            st.markdown(f"**ICs:** {', '.join(spec.initial_conditions)}")
        if spec.boundary_conditions:
            st.markdown(f"**BCs:** {', '.join(spec.boundary_conditions)}")
        if spec.parameters:
            st.markdown(f"**Parameters:** {', '.join(f'{k}={v}' for k, v in spec.parameters.items())}")

        features = []
        if spec.has_high_frequency:
            features.append("high-frequency")
        if spec.has_sharp_gradients:
            features.append("sharp-gradients")
        if spec.has_periodic_bc:
            features.append("periodic-BC")
        if not spec.is_linear:
            features.append("nonlinear")
        if features:
            st.markdown(f"**Features:** {', '.join(features)}")

    # SymPy verification
    if verify:
        try:
            from lang_pinn import verify_spec
            issues = verify_spec(spec)
            if issues:
                for issue in issues:
                    st.warning(f"SymPy: {issue}")
            else:
                st.success("SymPy verification passed")
        except Exception as e:
            st.caption(f"SymPy verification skipped: {e}")

    # Architecture Recommendation
    st.subheader("Architecture Recommendation")
    arch = result.architecture
    with st.container(border=True):
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Network", f"{arch.hidden_layers}×{arch.hidden_neurons}")
        ac2.metric("Activation", arch.activation)
        ac3.metric("Epochs", f"{arch.epochs:,}")
        ac4.metric("Collocation", f"{arch.n_collocation:,}")

        ac5, ac6, ac7 = st.columns(3)
        ac5.metric("Learning Rate", f"{arch.learning_rate}")
        ac6.metric("Ansatz", arch.ansatz_type or "None")
        ac7.metric("Loss Weights", str(arch.loss_weights))

        if arch.reasoning:
            st.caption(f"**Reasoning:** {arch.reasoning}")

    # Generated Code
    st.subheader("Generated Code")
    st.code(result.code, language="python", line_numbers=True)

    # Download button
    st.download_button(
        ":material/download: Download Code",
        data=result.code,
        file_name="generated_experiment.py",
        mime="text/x-python",
    )


_KB_STORE_DIR = Path("data/pinn-knowledge/store")
_KB_SOURCES_DIR = Path("data/pinn-knowledge/sources")
_KB_REGISTRY_DB = Path("data/pinn-knowledge/registry.db")
_KB_PAGE_SIZE = 10


def _load_kb():
    """Load the knowledge store and search engine (cached per session)."""
    from rag import KnowledgeStore, SearchEngine

    store = KnowledgeStore.load(_KB_STORE_DIR)
    engine = SearchEngine.from_store(store)
    return store, engine


@st.dialog("Confirm Delete", width="small")
def _confirm_delete_dialog(doc_id: str, doc_name: str):
    """Confirmation dialog for deleting a knowledge entry."""
    st.markdown(f"Delete **{format_experiment_name(doc_name)}** (`{doc_id}`)?")
    st.caption("This removes the document from the knowledge store permanently.")
    col_yes, col_no = st.columns(2)
    if col_yes.button(
        ":material/delete: Delete", type="primary", use_container_width=True
    ):
        from rag import KnowledgeStore

        store = KnowledgeStore.load(_KB_STORE_DIR)
        store.remove_document(doc_id)
        store.save()
        st.session_state["kb_deleted"] = doc_id
        st.rerun()
    if col_no.button("Cancel", use_container_width=True):
        st.rerun()


def page_knowledge_base():
    """Browse, search, upload, and manage the PINN knowledge base."""
    if not (_KB_STORE_DIR / "manifest.json").exists():
        st.warning(
            "Knowledge store not found. Build it first:\n\n"
            "```bash\nuv run python data/pinn-knowledge/build_index.py\n```"
        )
        return

    # Show delete success message
    if "kb_deleted" in st.session_state:
        st.success(f"Deleted document `{st.session_state.pop('kb_deleted')}`")

    store, engine = _load_kb()
    docs = store.list_documents()

    # --- Overview metrics ---
    total_nodes = sum(d.node_count for d in docs)
    total_tokens = sum(d.total_tokens for d in docs)
    all_techniques = sorted({t for d in docs for t in d.techniques})
    all_keywords = sorted({k for d in docs for k in d.keywords})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(docs))
    c2.metric("Total Nodes", total_nodes)
    c3.metric("Total Tokens", f"{total_tokens:,}")
    c4.metric("Techniques", len(all_techniques))

    # --- Tabs: Browse / Upload ---
    tab_browse, tab_upload = st.tabs([
        ":material/menu_book: Browse & Search",
        ":material/upload_file: Upload Document",
    ])

    with tab_upload:
        _kb_upload_section(store)

    with tab_browse:
        _kb_browse_section(store, engine, docs, all_keywords)


def _kb_upload_section(store):
    """File upload with progress tracking."""
    from rag import FileRegistry, ingest_file

    st.markdown(
        "Upload a **Markdown** (`.md`) or **PDF** (`.pdf`) file to add it "
        "to the knowledge base. Files are deduplicated by content hash "
        "and saved to `data/pinn-knowledge/sources/`."
    )

    uploaded = st.file_uploader(
        "Choose a file",
        type=["md", "markdown", "txt", "pdf"],
        help="Supported formats: Markdown (.md), Text (.txt), PDF (.pdf)",
    )

    if uploaded is None:
        return

    # Save uploaded file to sources directory
    _KB_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    source_path = _KB_SOURCES_DIR / uploaded.name
    source_path.write_bytes(uploaded.getvalue())

    suffix = source_path.suffix.lower()
    is_pdf = suffix == ".pdf"

    # Options row
    col_ingest, col_info, col_hybrid = st.columns([1, 1, 1])
    ingest_btn = col_ingest.button(
        ":material/play_arrow: Ingest",
        type="primary",
        use_container_width=True,
    )
    col_info.caption(f"**{uploaded.name}** — {uploaded.size / 1024:.1f} KB")

    use_hybrid = False
    if is_pdf:
        use_hybrid = col_hybrid.checkbox(
            "Hybrid PDF (LLM for complex pages)",
            help="Uses LLM vision for pages with diagrams/graphs. Requires LLM API key.",
        )

    if not ingest_btn:
        return

    # Progress tracking
    progress = st.progress(0, text="Checking for duplicates...")
    status_container = st.container()

    with FileRegistry(_KB_REGISTRY_DB) as registry:
        # Step 1: Dedup check
        file_hash = FileRegistry.compute_hash(source_path)
        existing = registry.lookup(file_hash)

        if existing is not None:
            progress.progress(100, text="Already indexed!")
            status_container.info(
                f"This file is already indexed as `{existing.doc_id}` "
                f"(hash: `{file_hash[:16]}...`). Content unchanged."
            )
            return

        # Step 2: Detect format and classify PDF pages
        progress.progress(20, text="Detecting format...")
        llm_client = None

        if is_pdf:
            from rag.indexing.pdf import classify_pages, get_page_count

            page_count = get_page_count(source_path)
            progress.progress(30, text=f"PDF detected: {page_count} pages. Classifying...")

            classifications = classify_pages(source_path)
            complex_pages = [c for c in classifications if c["complexity"] == "complex"]
            simple_pages = [c for c in classifications if c["complexity"] == "simple"]

            progress.progress(40, text=(
                f"Classification: {len(simple_pages)} simple, "
                f"{len(complex_pages)} complex pages"
            ))

            if complex_pages:
                page_nums = ", ".join(str(c["page_num"] + 1) for c in complex_pages)
                if use_hybrid:
                    status_container.caption(
                        f"Complex pages (LLM vision): {page_nums}"
                    )
                else:
                    status_container.caption(
                        f"Complex pages detected ({page_nums}) — "
                        f"using pymupdf4llm for all. Enable hybrid mode for LLM vision."
                    )

            # Create LLM client for hybrid mode
            if use_hybrid and complex_pages:
                try:
                    from llm_provider import LLMClient
                    llm_client = LLMClient()
                    progress.progress(45, text="LLM client ready for complex pages...")
                except Exception as exc:
                    status_container.warning(
                        f"Could not initialize LLM client: {exc}. "
                        f"Falling back to pymupdf4llm for all pages."
                    )
                    use_hybrid = False
        else:
            progress.progress(40, text="Markdown/text file detected")

        # Step 3: Index
        progress.progress(50, text="Converting and indexing...")

        result = ingest_file(
            source_path,
            store=store,
            registry=registry,
            llm_client=llm_client,
            hybrid_pdf=use_hybrid,
        )

        # Step 4: Results
        if result.status == "indexed":
            progress.progress(100, text="Indexing complete!")

            doc_meta = store.get_metadata(result.doc_id)

            with status_container.container(border=True):
                st.markdown(
                    f"### :material/check_circle: Indexed: "
                    f"{format_experiment_name(doc_meta.doc_name)}"
                )

                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Document ID", result.doc_id)
                rc2.metric("Nodes", doc_meta.node_count)
                rc3.metric("Tokens", doc_meta.total_tokens)

                st.caption(f"**Hash:** `{result.file_hash[:24]}...`")
                st.caption(f"**Source:** `{source_path}`")

                if is_pdf:
                    # Show converter details from registry
                    rec = registry.lookup(result.file_hash)
                    if rec and rec.llm_pages:
                        llm_pg = rec.llm_pages
                        st.caption(
                            f"**Converter:** hybrid (LLM pages: {llm_pg}, "
                            f"rest: pymupdf4llm)"
                        )
                    else:
                        st.caption("**Converter:** pymupdf4llm (all pages)")

                if doc_meta.keywords:
                    st.markdown(
                        f"**Keywords:** {', '.join(f'`{k}`' for k in doc_meta.keywords)}"
                    )

        elif result.status == "error":
            progress.progress(100, text="Error!")
            status_container.error(f"Ingestion failed: {result.message}")


def _kb_browse_section(store, engine, docs, all_keywords):
    """Browse and search with pagination."""
    st.markdown("---")

    # --- Search and filter ---
    col_search, col_filter = st.columns([2, 1])
    query = col_search.text_input(
        ":material/search: Search knowledge base",
        placeholder="e.g. burgers shock viscosity, spectral bias, loss weighting...",
    )
    keyword_filter = col_filter.selectbox(
        "Filter by keyword",
        ["All"] + all_keywords,
    )

    # Determine which docs to show
    if query.strip():
        matched_ids = engine.search(query, top_k=50)
        if not matched_ids:
            st.info("No results found. Try different search terms.")
            return
        filtered = [d for d in docs if d.doc_id in matched_ids]
        id_order = {doc_id: i for i, doc_id in enumerate(matched_ids)}
        filtered.sort(key=lambda d: id_order.get(d.doc_id, 999))
    else:
        filtered = docs

    if keyword_filter != "All":
        filtered = [d for d in filtered if keyword_filter in d.keywords]

    total = len(filtered)
    if total == 0:
        st.info("No documents match the current filters.")
        return

    # --- Pagination ---
    total_pages = max(1, (total + _KB_PAGE_SIZE - 1) // _KB_PAGE_SIZE)

    if total > _KB_PAGE_SIZE:
        col_info, col_pager = st.columns([2, 1])
        col_info.caption(f"Showing **{total}** documents")
        current_page = col_pager.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            key="kb_page",
        )
    else:
        current_page = 1
        st.caption(f"Showing **{total}** documents")

    start = (current_page - 1) * _KB_PAGE_SIZE
    end = min(start + _KB_PAGE_SIZE, total)
    page_docs = filtered[start:end]

    if total > _KB_PAGE_SIZE:
        st.caption(f"Page {current_page} of {total_pages} ({start + 1}–{end} of {total})")

    # --- Document cards ---
    for meta in page_docs:
        pde_label = meta.pde_type.split(":")[0].strip() if meta.pde_type else "General"
        title = format_experiment_name(meta.doc_name)

        with st.expander(
            f":material/article: **{title}** — {pde_label}",
            expanded=bool(query.strip()),
        ):
            # Metadata row
            mc1, mc2, mc3, mc4 = st.columns([3, 2, 2, 1])
            mc1.caption(f"**Nodes:** {meta.node_count}  |  **Tokens:** {meta.total_tokens}")
            mc2.caption(f"**ID:** `{meta.doc_id}`")
            indexed_date = meta.indexed_at.split("T")[0] if meta.indexed_at else "?"
            mc3.caption(f"**Indexed:** {indexed_date}")

            if mc4.button(
                ":material/delete:",
                key=f"del_{meta.doc_id}",
                help="Delete this entry",
            ):
                _confirm_delete_dialog(meta.doc_id, meta.doc_name)

            # PDE type
            if meta.pde_type:
                st.markdown(f"**Equation Type:** `{meta.pde_type}`")

            # Techniques
            if meta.techniques:
                tech_chips = "  ".join(f"`{t}`" for t in meta.techniques)
                st.markdown(f"**Techniques:** {tech_chips}")

            # Known issues
            if meta.known_issues:
                issue_chips = "  ".join(f"`{i}`" for i in meta.known_issues)
                st.markdown(f"**Known Issues:** {issue_chips}")

            # Keywords
            if meta.keywords:
                kw_chips = "  ".join(f"`{k}`" for k in meta.keywords)
                st.markdown(f"**Keywords:** {kw_chips}")

            # Full content toggle
            if st.toggle("Show full content", key=f"content_{meta.doc_id}"):
                tree = store.get_document(meta.doc_id)
                _render_tree_nodes(tree.root_nodes, level=0)


def _render_tree_nodes(nodes, level: int) -> None:
    """Recursively render tree nodes as nested markdown."""
    for node in nodes:
        prefix = "#" * min(node.level + 1, 5)
        st.markdown(f"{prefix} {node.title}")
        if node.text:
            st.markdown(node.text)
        if node.children:
            _render_tree_nodes(node.children, level + 1)


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
            ":material/smart_toy: Lang-PINN",
            ":material/menu_book: Knowledge Base",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

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
elif "Lang-PINN" in page:
    page_lang_pinn()
elif "Knowledge" in page:
    page_knowledge_base()
