"""The net: a 136 -> 7 MLP, its chunked prediction, and its save file.

What is pinned here, and why:

- The default is the spec's `136 -> 64 -> 64 -> 64 -> 7` with 17,543
  parameters, and width and depth are run parameters — one trainer must
  serve the size sweep.
- `predict` equals a plain forward pass, chunk by chunk, and returns float32
  NumPy — the shape `materialise` writes from.
- The wrong feature width is refused by name, not left to a matmul error.
- A saved net comes back with the same size and the same outputs.

Every test here skips without torch: the default suite runs without the
`deep` extra.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drawmaha_solver.deepcfr import net as net_module  # noqa: E402
from drawmaha_solver.deepcfr.net import Net, load_net, predict, save_net  # noqa: E402

# ---------------------------------------------------------------------------
# The shape
# ---------------------------------------------------------------------------

def test_default_net_is_three_by_sixty_four():
    net = Net()
    linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [
        (136, 64),
        (64, 64),
        (64, 64),
        (64, 7),
    ]
    # 136·64+64 + 2·(64·64+64) + 64·7+7 = 8,768 + 8,320 + 455.
    assert sum(p.numel() for p in net.parameters()) == 17_543
    assert (net.width, net.depth) == (64, 3)

def test_width_and_depth_are_parameters():
    net = Net(width=128, depth=4)
    linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 5
    assert linears[0].out_features == 128
    assert linears[-1].out_features == 7

def test_net_refuses_a_nonpositive_size():
    with pytest.raises(ValueError, match="width"):
        Net(width=0)
    with pytest.raises(ValueError, match="depth"):
        Net(depth=0)

def test_outputs_are_signed():
    # Regrets are signed, so the head has no activation: over random inputs
    # some outputs must come out negative.
    torch.manual_seed(0)
    x = torch.randn(256, 136)
    assert (Net()(x) < 0).any()

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_matches_forward_and_chunks(monkeypatch):
    net = Net()
    x = np.random.default_rng(0).standard_normal((1_000, 136)).astype(np.float32)
    monkeypatch.setattr(net_module, "PREDICT_CHUNK", 300)  # four chunks, the last short
    out = predict(net, x)
    assert out.shape == (1_000, 7) and out.dtype == np.float32
    np.testing.assert_allclose(out, net(torch.from_numpy(x)).detach().numpy(), rtol=1e-6)

def test_predict_of_nothing_is_an_empty_batch():
    assert predict(Net(), np.zeros((0, 136), dtype=np.float32)).shape == (0, 7)

def test_predict_refuses_the_wrong_width():
    with pytest.raises(ValueError, match="136"):
        predict(Net(), np.zeros((3, 92), dtype=np.float32))
    with pytest.raises(ValueError, match="136"):
        predict(Net(), np.zeros(136, dtype=np.float32))

# ---------------------------------------------------------------------------
# The save file
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    net = Net(width=32, depth=2)
    save_net(net, tmp_path / "net.pt")
    back = load_net(tmp_path / "net.pt")
    assert (back.width, back.depth) == (32, 2)
    x = np.ones((5, 136), dtype=np.float32)
    np.testing.assert_array_equal(predict(back, x), predict(net, x))
