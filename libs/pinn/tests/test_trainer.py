import pytest
import torch
from pinn import PINN, PINNTrainer

N_EPOCHS = 5


@pytest.fixture()
def data():
    return torch.rand(20, 1)


@pytest.fixture()
def losses(data):
    return {
        "a": lambda m: m(data).pow(2).mean(),
        "b": lambda m: (m(data) - 1.0).pow(2).mean(),
    }


def make_trainer(tiny_model, cpu):
    trainer = PINNTrainer(tiny_model, device=cpu)
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    return trainer, optimizer


def test_history_length_and_keys(tiny_model, cpu, losses):
    trainer, opt = make_trainer(tiny_model, cpu)
    history = trainer.train(N_EPOCHS, opt, losses, verbose=False, log_every=0)
    assert history is trainer.loss_history
    assert len(history) == N_EPOCHS
    assert set(history[0]) == {"a", "b", "total"}


def test_total_is_weighted_sum(tiny_model, cpu, losses):
    trainer, opt = make_trainer(tiny_model, cpu)
    weights = {"a": 0.5, "b": 3.0}
    history = trainer.train(N_EPOCHS, opt, losses, weights=weights, verbose=False, log_every=0)
    for entry in history:
        expected = weights["a"] * entry["a"] + weights["b"] * entry["b"]
        assert entry["total"] == pytest.approx(expected, rel=1e-6)


def test_missing_weight_defaults_to_one(tiny_model, cpu, losses):
    trainer, opt = make_trainer(tiny_model, cpu)
    history = trainer.train(N_EPOCHS, opt, losses, weights={"a": 2.0}, verbose=False, log_every=0)
    entry = history[0]
    assert entry["total"] == pytest.approx(2.0 * entry["a"] + 1.0 * entry["b"], rel=1e-6)


def test_early_stopping_triggers(tiny_model, cpu, losses):
    # lr=0 -> loss can never improve -> patience exhausts immediately
    trainer = PINNTrainer(tiny_model, device=cpu)
    frozen_opt = torch.optim.SGD(tiny_model.parameters(), lr=0.0)
    patience = 3
    history = trainer.train(
        100, frozen_opt, losses, verbose=False, log_every=0, early_stop_patience=patience
    )
    # epoch 0 improves over inf; the next `patience` epochs do not
    assert len(history) == 1 + patience


def test_grad_clip_caps_gradient_norm(cpu):
    clip = 1e-3

    class RecordingSGD(torch.optim.SGD):
        norms: list[float] = []

        def step(self, closure=None):
            total = sum(
                p.grad.norm().item() ** 2
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            )
            RecordingSGD.norms.append(total**0.5)
            return super().step(closure)

    model = PINN(1, 2, 8)
    trainer = PINNTrainer(model, device=cpu)
    data = torch.rand(20, 1)
    # large-scale loss to guarantee raw gradients far exceed the clip value
    big_loss = {"big": lambda m: 1e6 * (m(data) - 5.0).pow(2).mean()}
    RecordingSGD.norms = []
    opt = RecordingSGD(model.parameters(), lr=1e-3)
    trainer.train(N_EPOCHS, opt, big_loss, verbose=False, log_every=0, grad_clip=clip)
    assert RecordingSGD.norms  # sanity: step() was reached
    assert all(n <= clip * 1.01 for n in RecordingSGD.norms)


def test_callbacks_fire_every_epoch(tiny_model, cpu, losses):
    trainer, opt = make_trainer(tiny_model, cpu)
    calls: list[tuple[int, dict]] = []
    trainer.train(
        N_EPOCHS, opt, losses, verbose=False, log_every=0,
        callbacks=[lambda epoch, entry: calls.append((epoch, entry))],
    )
    assert [epoch for epoch, _ in calls] == list(range(N_EPOCHS))
    assert all("total" in entry for _, entry in calls)


def test_checkpoint_round_trip(tiny_model, cpu, losses, tmp_path):
    trainer, opt = make_trainer(tiny_model, cpu)
    trainer.train(N_EPOCHS, opt, losses, verbose=False, log_every=0)
    metadata = {"seed": 0, "note": "round-trip"}
    path = trainer.save_checkpoint(tmp_path / "ckpt.pt", optimizer=opt, metadata=metadata)

    fresh_model = PINN(input_dim=1, hidden_layers=2, hidden_neurons=8)
    fresh_trainer = PINNTrainer(fresh_model, device=cpu)
    fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
    loaded_meta = fresh_trainer.load_checkpoint(path, optimizer=fresh_opt)

    assert loaded_meta == metadata
    assert fresh_trainer.loss_history == trainer.loss_history
    x = torch.rand(5, 1)
    assert torch.allclose(tiny_model(x), fresh_model(x))
    assert fresh_opt.state_dict()["param_groups"] == opt.state_dict()["param_groups"]


def test_plot_loss_history_headless(tiny_model, cpu, losses, tmp_path):
    trainer, opt = make_trainer(tiny_model, cpu)
    trainer.train(N_EPOCHS, opt, losses, verbose=False, log_every=0)
    out = tmp_path / "loss.png"
    trainer.plot_loss_history(show_total=True, save_path=out, show=False)
    assert out.exists() and out.stat().st_size > 0
