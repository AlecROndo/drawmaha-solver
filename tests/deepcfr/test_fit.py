"""The from-scratch refit: t-weighted squared error over the legal columns.

What is pinned here, and why:

- A small memory of real rows is memorised: the fit drives the loss far
  below a fresh net's, so the optimiser, the features and the gather are
  wired to each other.
- Later iterations weigh more: one row offered at t = 3 with target +1
  and at t = 1 with target −1 fits to the weighted answer, +0.5.
- A target's padding never reaches the loss or the gradient: PR 3 may pad
  with anything, so the padding here is NaN and the loss and every
  gradient stay finite and equal to the zero-padded case.
- The same seed gives identical weights (resume and the exactness pin need
  it) and a different seed different weights.
- Mismatched lengths, an empty memory, t below 1 and rows outside the
  table are refused by name before any training starts.
- Huge targets do not blow the net up: the gradient norm is clipped to the
  config's value, and every parameter stays finite.
- The process-global thread count is put back after a fit, so one fit's
  config never changes another test's arithmetic.

Every test skips without torch.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drawmaha_solver.deepcfr import fit as fit_module  # noqa: E402
from drawmaha_solver.deepcfr.fit import FitConfig, batch_loss, fit_from_scratch  # noqa: E402
from drawmaha_solver.deepcfr.net import Net, predict  # noqa: E402
from drawmaha_solver.minidrawmaha.features import (  # noqa: E402
    MAX_WIDTH,
    NO_ACTION,
    layout,
    legal_gather,
)

SMALL = FitConfig(steps=300, batch=256, threads=1)

# ---------------------------------------------------------------------------
# Helpers: real rows with random targets, and a row's legal predictions
# ---------------------------------------------------------------------------

def real_rows(n: int, seed: int, widths: set[int] | None = None):
    """`n` row numbers of the real table, t in 1..100, targets in [−1, 1] padded with zeros."""
    built = layout()
    rng = np.random.default_rng(seed)
    rows = np.empty(0, dtype=np.int64)
    while len(rows) < n:
        drawn = rng.integers(0, built.row_offset[-1], size=n)
        if widths is not None:
            point, _ = built.locate_rows(drawn)
            drawn = drawn[np.isin(built.width[point], list(widths))]
        rows = np.concatenate([rows, drawn])[:n]
    t = rng.integers(1, 101, size=n)
    targets = rng.uniform(-1.0, 1.0, size=(n, MAX_WIDTH)).astype(np.float32)
    point, _ = built.locate_rows(rows)
    targets[legal_gather()[point] == NO_ACTION] = 0.0
    return rows, t, targets

def legal_prediction(net: Net, row: int) -> np.ndarray:
    built = layout()
    point, _ = built.locate_rows(np.array([row]))
    gather = legal_gather()[point[0]]
    return predict(net, built.features(np.array([row])))[0, gather[gather != NO_ACTION]]

# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def test_fit_memorises_a_small_memory():
    rows, t, targets = real_rows(200, seed=0)
    net = fit_from_scratch(
        rows, t, targets, seed=(0, 1, 0), config=FitConfig(steps=1_500, batch=200, threads=1)
    )
    torch.manual_seed(0)
    fresh = Net()
    assert batch_loss(net, rows, t, targets).item() < 0.05 * batch_loss(fresh, rows, t, targets).item()

def test_fit_returns_the_net_in_eval_mode():
    net = fit_from_scratch(*real_rows(16, seed=5), seed=(0,), config=SMALL)
    assert not net.training

def test_later_iterations_weigh_more():
    # One row offered twice: +1 at t = 3 and −1 at t = 1. The weighted
    # least-squares answer is (3·1 + 1·(−1)) / (3 + 1) = +0.5.
    (row,), _, _ = real_rows(1, seed=3)
    rows = np.array([row, row])
    t = np.array([3, 1])
    targets = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]], dtype=np.float32)
    net = fit_from_scratch(
        rows, t, targets, seed=(0, 1, 0), config=FitConfig(steps=1_500, batch=64, threads=1)
    )
    np.testing.assert_allclose(legal_prediction(net, row), 0.5, atol=0.05)

def test_loss_is_over_the_legal_columns_only():
    # A 2-wide betting row: its loss is the sum over its two columns, so a
    # target met exactly at both leaves zero whatever the padding says.
    rows, t, _ = real_rows(1, seed=7, widths={2})
    torch.manual_seed(0)
    net = Net()
    point, _ = layout().locate_rows(rows)
    gather = legal_gather()[point[0]]
    outputs = predict(net, layout().features(rows))[0]
    targets = np.full((1, MAX_WIDTH), 9.0, dtype=np.float32)
    targets[0, :2] = outputs[gather[:2]]
    assert batch_loss(net, rows, t, targets).item() == pytest.approx(0.0, abs=1e-10)

def test_padding_never_reaches_the_loss():
    rows, t, targets = real_rows(64, seed=1, widths={2, 3})  # betting rows only
    point, _ = layout().locate_rows(rows)
    padding = legal_gather()[point] == NO_ACTION
    nan_padded = targets.copy()
    nan_padded[padding] = np.nan
    torch.manual_seed(0)
    net = Net()
    loss = batch_loss(net, rows, t, nan_padded)
    assert torch.isfinite(loss)
    assert loss.item() == batch_loss(net, rows, t, targets).item()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())

def test_same_seed_same_weights():
    a = fit_from_scratch(*real_rows(64, seed=2), seed=(7, 3, 1), config=SMALL)
    b = fit_from_scratch(*real_rows(64, seed=2), seed=(7, 3, 1), config=SMALL)
    c = fit_from_scratch(*real_rows(64, seed=2), seed=(7, 3, 0), config=SMALL)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not all(
        torch.equal(x, y) for x, y in zip(a.state_dict().values(), c.state_dict().values())
    )

def test_seed_tuples_are_folded_not_summed():
    # (1, 2) and (2, 1) sum alike; the fold must still tell them apart.
    a = fit_from_scratch(*real_rows(16, seed=2), seed=(1, 2), config=SMALL)
    b = fit_from_scratch(*real_rows(16, seed=2), seed=(2, 1), config=SMALL)
    assert not all(
        torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values())
    )

def test_config_sizes_the_net():
    net = fit_from_scratch(
        *real_rows(16, seed=2), seed=(0,), config=FitConfig(width=16, depth=2, steps=5, threads=1)
    )
    assert (net.width, net.depth) == (16, 2)

# ---------------------------------------------------------------------------
# Clipping and threads
# ---------------------------------------------------------------------------

def test_gradient_norm_is_clipped(monkeypatch):
    rows, t, targets = real_rows(32, seed=4)
    targets *= 1e6
    seen = []
    real_clip = fit_module.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        total = real_clip(parameters, max_norm, *args, **kwargs)
        seen.append((float(max_norm), float(total)))
        return total

    monkeypatch.setattr(fit_module, "clip_grad_norm_", spy)
    config = FitConfig(steps=5, batch=32, threads=1, clip=0.5)
    net = fit_from_scratch(rows, t, targets, seed=(0,), config=config)
    assert len(seen) == 5
    assert all(max_norm == 0.5 for max_norm, _ in seen)
    assert all(total > 0.5 for _, total in seen)  # 1e6 targets: every step was clipped
    assert all(torch.isfinite(p).all() for p in net.parameters())

def test_thread_count_is_put_back():
    before = torch.get_num_threads()
    fit_from_scratch(*real_rows(16, seed=2), seed=(0,), config=FitConfig(steps=2, threads=1))
    assert torch.get_num_threads() == before

# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_fit_refuses_mismatched_or_empty_rows():
    rows, t, targets = real_rows(10, seed=3)
    with pytest.raises(ValueError, match="same length"):
        fit_from_scratch(rows, t[:-1], targets, seed=(0,), config=SMALL)
    with pytest.raises(ValueError, match="empty"):
        fit_from_scratch(rows[:0], t[:0], targets[:0], seed=(0,), config=SMALL)
    with pytest.raises(ValueError, match=">= 1"):
        fit_from_scratch(rows, np.zeros_like(t), targets, seed=(0,), config=SMALL)

def test_fit_refuses_targets_of_the_wrong_width():
    rows, t, targets = real_rows(10, seed=3)
    with pytest.raises(ValueError, match=str(MAX_WIDTH)):
        fit_from_scratch(rows, t, targets[:, :3], seed=(0,), config=SMALL)

def test_fit_refuses_rows_outside_the_table():
    rows, t, targets = real_rows(10, seed=4)
    rows[0] = layout().row_offset[-1]
    with pytest.raises(IndexError):
        fit_from_scratch(rows, t, targets, seed=(0,), config=SMALL)

def test_config_refuses_nonpositive_sizes():
    for field in ("steps", "batch", "threads"):
        with pytest.raises(ValueError, match=field):
            FitConfig(**{field: 0})

def test_fit_refuses_a_non_finite_memory():
    rows, t, targets = real_rows(10, seed=3)
    bad_t = t.astype(np.float64)
    bad_t[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_from_scratch(rows, bad_t, targets, seed=(0,), config=SMALL)
    point, _ = layout().locate_rows(rows)
    legal = legal_gather()[point] != NO_ACTION
    bad_targets = targets.copy()
    bad_targets[np.argwhere(legal)[0][0], np.argwhere(legal)[0][1]] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_from_scratch(rows, t, bad_targets, seed=(0,), config=SMALL)

def test_fit_refuses_a_net_that_diverged():
    # An absurd learning rate sends the weights past float32 in two steps
    # (the clip bounds the gradient, not the step). That must be an error,
    # not a net that materialises to uniform play with nothing in the log.
    rows, t, targets = real_rows(10, seed=3)
    with pytest.raises(ValueError, match="finite"):
        fit_from_scratch(rows, t, targets, seed=(0,), config=FitConfig(steps=3, threads=1, lr=1e10))

def test_fit_refuses_rows_or_t_that_are_not_one_dimensional_ints():
    rows, t, targets = real_rows(10, seed=3)
    with pytest.raises(ValueError, match="1-D"):
        fit_from_scratch(rows, t[:, None], targets, seed=(0,), config=SMALL)
    with pytest.raises(ValueError, match="1-D"):
        fit_from_scratch(rows[:, None], t, targets, seed=(0,), config=SMALL)
    with pytest.raises(ValueError, match="integer"):
        fit_from_scratch(rows.astype(np.float64), t, targets, seed=(0,), config=SMALL)
