"""Distillation, the torch half: the loss, the fit, the fraction arm, the freeze.

What is pinned here, and why:

- The masked cross-entropy is minimised, with zero gradient, at the
  teacher's own distribution, and an illegal column gets no gradient.
- A tiny fit memorises a slice of 64 rows: the sampler, the features, the
  gather and the loss are wired to each other, the curve has an entry per
  `eval_every` steps, and the same config gives the same loss.
- A fraction arm reports the error on the rows it never saw.
- The frozen student is the softmax at every key, equals the on-device
  read the curve is taken from, and survives the file.

Every test skips without torch; the dataset half is `test_distill.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drawmaha_solver.deepcfr.distill import (  # noqa: E402
    DistillConfig,
    distill,
    masked_cross_entropy,
    reach_weighted_error,
    student_probabilities,
    student_strategy,
    teacher_targets,
)
from drawmaha_solver.deepcfr.net import Net, predict  # noqa: E402
from drawmaha_solver.minidrawmaha.rules.enumeration import (  # noqa: E402
    PublicPoint,
    private_keys,
    public_decision_points,
)
from drawmaha_solver.minidrawmaha.rules.features import (  # noqa: E402
    NO_ACTION,
    encode,
    layout,
    legal_gather,
)
from drawmaha_solver.minidrawmaha.rules.game import InfoSet  # noqa: E402
from drawmaha_solver.minidrawmaha.solver.packed_table import PackedTable  # noqa: E402
from drawmaha_solver.minidrawmaha.answer.strategy import (  # noqa: E402
    FrozenStrategy,
    StrategyInfo,
    average_rows,
    load_strategy,
    save_strategy,
)

# ---------------------------------------------------------------------------
# Helpers (restated: test modules never import one another)
# ---------------------------------------------------------------------------

def key_on(point: PublicPoint, private) -> InfoSet:
    hole, thrown, board = private
    return InfoSet(
        player=point.player, hole=hole, discarded=thrown, board=board,
        draws=point.draws, betting=point.betting,
    )

def some_keys() -> list[InfoSet]:
    """Two keys on every public point: the first and the middle private key."""
    keys = []
    for point in public_decision_points():
        shape = private_keys(board_cards=point.board_cards, discards=point.discards)
        keys += [key_on(point, shape[0]), key_on(point, shape[len(shape) // 2])]
    return keys

def random_strategy(seed: int) -> FrozenStrategy:
    """Positive random sums at every column, normalised: no row uniform by accident."""
    widths = PackedTable.whole_game().widths()
    sums = np.random.default_rng(seed).uniform(0.05, 1.0, int(widths.sum()))
    info = StrategyInfo(rule="test", column="linear", iteration=1, seed=seed, workers=1)
    return FrozenStrategy(average_rows(sums, widths), widths, info)

# ---------------------------------------------------------------------------

TINY = DistillConfig(width=32, depth=2, steps=600, batch=64, lr=3e-3, device="cpu", eval_every=200)

def test_masked_cross_entropy_is_minimised_at_the_teacher():
    # Outputs equal to log of the target at the legal columns: the softmax
    # over them IS the target, the loss is the rows' mean entropy, and the
    # gradient with respect to the outputs is zero. Illegal columns hold
    # anything and get no gradient.
    teacher = random_strategy(11)
    targets = torch.from_numpy(teacher_targets(teacher)[:6])
    point, _ = layout().locate_rows(np.arange(6))
    gather = torch.from_numpy(legal_gather()[point].astype(np.int64))
    legal = gather != NO_ACTION
    outputs = torch.full((6, 7), 5.0, requires_grad=True)
    with torch.no_grad():
        outputs.scatter_(1, gather.clamp(min=0), torch.where(legal, targets.log(), 5.0))
    loss = masked_cross_entropy(outputs, gather, targets)
    entropy = -(torch.where(legal, targets * targets.log(), 0.0)).sum(dim=1).mean()
    assert loss.item() == pytest.approx(entropy.item(), rel=1e-6)
    loss.backward()
    assert outputs.grad.abs().max().item() < 1e-6

def slice_weights(rows: int, picked: np.ndarray) -> np.ndarray:
    weights = np.zeros(rows)
    weights[picked] = 1.0 / len(picked)
    return weights

def test_a_tiny_fit_memorises_a_slice():
    teacher = random_strategy(13)
    targets = teacher_targets(teacher)
    built = layout()
    # 64 rows spread over the game, every width among them.
    picked = np.linspace(0, built.row_offset[-1] - 1, 64).astype(np.int64)
    widths = np.repeat(built.width, np.diff(built.row_offset))
    assert set(widths[picked]) == {2, 3, 4}
    weights = slice_weights(len(widths), picked)
    lines = []
    net, curve = distill(targets, weights, TINY, log=lines.append)
    assert isinstance(net, Net) and (net.width, net.depth) == (32, 2)
    assert [c["step"] for c in curve] == [200, 400, 600]
    assert all(set(c) == {"step", "loss", "lr", "rw_error", "seconds"} for c in curve)
    assert curve[0]["loss"] > curve[-1]["loss"]
    # 600 Adam steps of a 32 × 2 net on 64 rows of random targets reach an
    # error of about 1e-5 per column on this seed; 0.01 leaves headroom.
    assert curve[-1]["rw_error"] < 0.01
    assert len(lines) == 3
    # The in-memory student's error equals the curve's last point.
    student = student_strategy(net, teacher=teacher.info)
    assert reach_weighted_error(student, targets, weights) == pytest.approx(curve[-1]["rw_error"], abs=1e-6)
    assert distill(targets, weights, TINY, log=lambda _: None)[1][-1]["loss"] == curve[-1]["loss"]

def test_a_fraction_arm_reports_the_unseen_error():
    teacher = random_strategy(13)
    targets = teacher_targets(teacher)
    built = layout()
    picked = np.linspace(0, built.row_offset[-1] - 1, 64).astype(np.int64)
    weights = slice_weights(int(built.row_offset[-1]), picked)
    config = DistillConfig(width=32, depth=2, steps=200, batch=64, device="cpu", eval_every=100, train_fraction=0.5)
    _, curve = distill(targets, weights, config, log=lambda _: None)
    assert all({"rw_error", "rw_error_seen", "rw_error_unseen"} <= set(c) for c in curve)
    assert all(np.isfinite(c["rw_error_unseen"]) for c in curve)

def test_student_strategy_is_the_softmax_at_every_key_and_survives_the_file(tmp_path):
    teacher = random_strategy(13)
    net = Net(width=16, depth=2)
    student = student_strategy(net, teacher=teacher.info)
    assert student.info == StrategyInfo(rule="distill", column="linear", iteration=1, seed=13, workers=1)
    for key in some_keys():
        out = predict(net, encode(key)[None])[0][[int(a) for a in key.legal_actions()]].astype(np.float64)
        expected = np.exp(out - out.max())
        np.testing.assert_allclose(student[key], expected / expected.sum(), rtol=1e-5, atol=1e-6)
    save_strategy(student, tmp_path / "student.npz")
    back = load_strategy(tmp_path / "student.npz")
    np.testing.assert_array_equal(back.probabilities, student.probabilities)
    assert back.info == student.info
    # The on-device probabilities the curve is computed from are the freeze.
    on_device = student_probabilities(net, device="cpu")
    np.testing.assert_allclose(on_device, teacher_targets(student), atol=2e-6)

def test_distill_refuses_a_bad_config():
    with pytest.raises(ValueError, match="train_fraction"):
        DistillConfig(train_fraction=0.0)
    with pytest.raises(ValueError, match="steps"):
        DistillConfig(steps=0)
    with pytest.raises(ValueError, match="device"):
        DistillConfig(device="cuda")
    with pytest.raises(ValueError, match="lr"):
        DistillConfig(lr=0.0)
