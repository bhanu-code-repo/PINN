import numpy as np
from pinn import plot_comparison_1d, plot_contour, plot_loss_comparison


def test_plot_contour_saves_file(tmp_path):
    x = np.linspace(0, 1, 20)
    t = np.linspace(0, 1, 30)
    T, X = np.meshgrid(t, x)
    out = tmp_path / "contour.png"
    plot_contour(T, X, np.sin(X + T), save_path=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0


def test_plot_comparison_1d_saves_file(tmp_path):
    x = np.linspace(0, 1, 50)
    out = tmp_path / "comparison.png"
    plot_comparison_1d(x, np.sin(x), np.sin(x) + 0.01, save_path=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0


def test_plot_loss_comparison_saves_file(tmp_path):
    out = tmp_path / "losses.png"
    plot_loss_comparison({"run-a": [1.0, 0.1, 0.01], "run-b": [2.0, 0.5, 0.1]},
                         save_path=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0
