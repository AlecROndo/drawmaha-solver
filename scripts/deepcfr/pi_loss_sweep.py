"""The policy-net loss sweep: refit Π on one stored policy memory under other losses, and grade each.

    uv run --extra deep python scripts/deepcfr/pi_loss_sweep.py \
        runs/sweep-256x4-s16000-lr5e-4 figures/rung4/pi_sweep/results.json

That is the command that wrote `figures/rung4/pi_sweep/results.json` on
2026-10-26 (the arms ran in three batches over the day; a third argument
`arm,arm,...` runs a subset and an arm already in the file is skipped).

**The question.** The 4 × 256 arm's exact average kept falling from 2M to
5M hands (0.115 → 0.108) while its policy net Π did not (0.135 → 0.136).
Π is a fresh net fit on the policy memory at each mark, so the stall is
either in the memory or in the fit. This sweep holds the memory fixed —
the policy memory saved at mark t = 500 of that run, and at t = 100 and
t = 200 for the arms suffixed `@t100` / `@t200` — and changes only the fit:
the loss, the head, the step count, the learning-rate schedule, the width
and the depth. Each refit is frozen the way `policy.policy_strategy`
freezes Π and graded by the exact best-response walk, so a lower number
means the fit alone moved exploitability.

**The arms.** `mse-linear-16k` is the run's own Π refit (squared error on a
linear head, 16,000 steps at a constant 5e-4), the control. `ce` is
cross-entropy on a softmax head (the targets are distributions); `kl` is
the KL divergence, which differs from `ce` by the targets' entropy and so
has the same gradient (it reproduces `ce` to the digit); `is` is the
Itakura–Saito divergence on (p + ε, q + ε), which weights a miss by about
1/p² and sits between the two. `mse-softmax` is squared error on softmax
outputs, to separate the head from the loss. `64k-cos` and up use cosine
decay from 1e-3.

**What is recorded per arm.** The grade (`exploitability`, `br0`, `br1`,
`value_p0`), the mean absolute difference from the exact average over every
column, the same over the "small" columns (exact probability in [0.01,
0.10]) and the share of those the refit zeroes (below 0.005), the fit and
grade seconds, and the arm's settings. `_reference` holds the exact
average's own grade at t = 500 and the run's Π grade, read from
`figures/rung4/sweep_256x4-s16000-lr5e-4.json`. The reach-weighted error,
which is the diagnostic that tracks exploitability, is
`scripts/deepcfr/pi_reach_weighted.py`.

Runs on Apple's GPU (`mps`): the fit is `fit.py`'s loop restated with a
loss argument, not `fit_from_scratch`, because the production fit knows one
loss. torch is imported at the top; this is a `--extra deep` script.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from drawmaha_solver.deepcfr.fit import _fold_seed
from drawmaha_solver.deepcfr.net import Net, predict
from drawmaha_solver.deepcfr.policy import policy_seed
from drawmaha_solver.deepcfr.reservoir import Reservoir
from drawmaha_solver.minidrawmaha.answer import analysis
from drawmaha_solver.minidrawmaha.answer.strategy import (
    FrozenStrategy,
    StrategyInfo,
    average_rows,
    load_strategy,
    save_strategy,
)
from drawmaha_solver.minidrawmaha.rules.features import NO_ACTION, layout, legal_gather

T = 500
RUN_GRADES = Path("figures/rung4/sweep_256x4-s16000-lr5e-4.json")

# name: (head, loss, steps, lr, schedule, width, depth[, mark])
ARMS = {
    "mse-linear-16k": ("linear", "mse", 16_000, 5e-4, "const", 256, 4),  # the run's Π, refit
    "ce-softmax-16k": ("softmax", "ce", 16_000, 5e-4, "const", 256, 4),
    "mse-softmax-16k": ("softmax", "mse", 16_000, 5e-4, "const", 256, 4),
    "mse-linear-64k-cos": ("linear", "mse", 64_000, 1e-3, "cosine", 256, 4),
    "ce-softmax-64k-cos": ("softmax", "ce", 64_000, 1e-3, "cosine", 256, 4),
    "ce-softmax-128k-cos": ("softmax", "ce", 128_000, 1e-3, "cosine", 256, 4),
    "mse-linear-128k-cos": ("linear", "mse", 128_000, 1e-3, "cosine", 256, 4),
    "ce-softmax-512x4-64k-cos": ("softmax", "ce", 64_000, 1e-3, "cosine", 512, 4),
    "mse-linear-512x4-64k-cos": ("linear", "mse", 64_000, 1e-3, "cosine", 512, 4),
    "ce-softmax-256x6-64k-cos": ("softmax", "ce", 64_000, 1e-3, "cosine", 256, 6),
    "ce-softmax-256k-cos": ("softmax", "ce", 256_000, 1e-3, "cosine", 256, 4),
    "ce-softmax-1024x4-64k-cos": ("softmax", "ce", 64_000, 1e-3, "cosine", 1024, 4),
    "ce-softmax-64k-cos@t200": ("softmax", "ce", 64_000, 1e-3, "cosine", 256, 4, 200),
    "ce-softmax-64k-cos@t100": ("softmax", "ce", 64_000, 1e-3, "cosine", 256, 4, 100),
    "mse-linear-64k-cos@t200": ("linear", "mse", 64_000, 1e-3, "cosine", 256, 4, 200),
    "kl-softmax-16k": ("softmax", "kl", 16_000, 5e-4, "const", 256, 4),
    "kl-softmax-64k-cos": ("softmax", "kl", 64_000, 1e-3, "cosine", 256, 4),
    "is-softmax-16k": ("softmax", "is", 16_000, 5e-4, "const", 256, 4),
    "is-softmax-64k-cos": ("softmax", "is", 64_000, 1e-3, "cosine", 256, 4),
    "is-softmax-128k-cos": ("softmax", "is", 128_000, 1e-3, "cosine", 256, 4),
    "is-softmax-256x6-64k-cos": ("softmax", "is", 64_000, 1e-3, "cosine", 256, 6),
    "is-softmax-512x4-64k-cos": ("softmax", "is", 64_000, 1e-3, "cosine", 512, 4),
    "is-softmax-256k-cos": ("softmax", "is", 256_000, 1e-3, "cosine", 256, 4),
    "is-softmax-1024x4-64k-cos": ("softmax", "is", 64_000, 1e-3, "cosine", 1024, 4),
    "is-softmax-64k-cos@t200": ("softmax", "is", 64_000, 1e-3, "cosine", 256, 4, 200),
    "is-softmax-64k-cos@t100": ("softmax", "is", 64_000, 1e-3, "cosine", 256, 4, 100),
}
IS_EPS = 1e-3  # Itakura–Saito on (p + ε, q + ε): finite at p = 0, still minimised by the mean


def loss_fn(outputs, gather, targets, t, *, head: str, loss: str):
    """One batch's loss: per row over its legal columns, weighted by t ÷ mean t, averaged over rows."""
    legal = gather != NO_ACTION
    picked = outputs.gather(1, gather.clamp(min=0))
    target = torch.where(legal, targets, 0.0)
    if head == "softmax":
        logp = torch.log_softmax(torch.where(legal, picked, -1e9), dim=1)
        if loss == "ce":
            per = -torch.where(legal, target * logp, 0.0).sum(dim=1)
        elif loss == "kl":
            per = torch.where(legal, torch.xlogy(target, target) - target * logp, 0.0).sum(dim=1)
        elif loss == "is":
            ratio = (target + IS_EPS) / (logp.exp() + IS_EPS)
            per = torch.where(legal, ratio - torch.log(ratio) - 1.0, 0.0).sum(dim=1)
        else:
            per = torch.where(legal, (logp.exp() - target) ** 2, 0.0).sum(dim=1)
    else:
        per = torch.where(legal, (picked - target) ** 2, 0.0).sum(dim=1)
    return ((t / t.mean()) * per).mean()


def mark_of(arm: str) -> int:
    return ARMS[arm][7] if len(ARMS[arm]) > 7 else T


def fit(rows, t, targets, arm: str) -> Net:
    """A fresh net fit to the memory under the arm's loss, head, schedule and size, on the GPU."""
    head, loss, steps, lr, schedule, width, depth = ARMS[arm][:7]
    device = torch.device("mps")
    built = layout()
    row_offset = torch.from_numpy(built.row_offset).to(device)
    private_offset = torch.from_numpy(built.private_offset).to(device)
    private = torch.from_numpy(np.array(built.private)).to(device)
    public = torch.from_numpy(np.array(built.public)).to(device)
    gather_all = torch.from_numpy(legal_gather().astype(np.int64)).to(device)
    m_rows = torch.from_numpy(rows.astype(np.int64)).to(device)
    m_t = torch.from_numpy(t.astype(np.float32)).to(device)
    m_targets = torch.from_numpy(np.require(targets, dtype=np.float32, requirements="W")).to(device)
    folded = _fold_seed(policy_seed(0, mark_of(arm)))
    torch.manual_seed(folded)
    net = Net(width=width, depth=depth).to(device)
    draw = torch.Generator().manual_seed(folded)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for step in range(steps):
        if schedule == "cosine":
            for g in opt.param_groups:
                g["lr"] = 0.5 * lr * (1 + math.cos(math.pi * step / steps))
        picked = torch.randint(len(rows), (10_000,), generator=draw).to(device)
        br = m_rows[picked]
        point = torch.searchsorted(row_offset, br, right=True) - 1
        position = br - row_offset[point]
        features = torch.cat([private[private_offset[point] + position], public[point]], dim=1)
        value = loss_fn(net(features), gather_all[point], m_targets[picked], m_t[picked], head=head, loss=loss)
        opt.zero_grad()
        value.backward()
        clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 4000 == 0:
            print(f"  {arm} step {step} loss {value.item():.5f}", flush=True)
    return net.to("cpu").eval()


def freeze(net: Net, arm: str) -> FrozenStrategy:
    """The net over every key as a strategy: softmax outputs as they are, linear outputs clipped and renormalised."""
    head = ARMS[arm][0]
    built = layout()
    gather = legal_gather()
    flat = np.empty(int(built.column_offset[-1]))
    for p in range(len(built.points)):
        rows = built.row_offset[p + 1] - built.row_offset[p]
        width = built.width[p]
        private = built.private[built.private_offset[p] : built.private_offset[p] + rows]
        public = np.broadcast_to(built.public[p], (rows, built.public.shape[1]))
        out = predict(net, np.concatenate([private, public], axis=1))[:, gather[p, :width]].astype(np.float64)
        if head == "softmax":
            out = np.exp(out - out.max(axis=1, keepdims=True))
            out /= out.sum(axis=1, keepdims=True)
        else:
            out = np.maximum(out, 0.0)
        flat[built.column_offset[p] : built.column_offset[p + 1]] = out.ravel()
    widths = np.repeat(built.width, np.diff(built.row_offset))
    info = StrategyInfo(rule="deep", column="pi", iteration=mark_of(arm), seed=0, workers=0)
    return FrozenStrategy(average_rows(flat, widths), widths, info)


def small_prob_stats(pi: FrozenStrategy, exact: FrozenStrategy) -> dict:
    """How far a refit sits from the exact average, over every column and over the small ones."""
    a, b = pi.probabilities.astype(np.float64), exact.probabilities.astype(np.float64)
    small = (b >= 0.01) & (b <= 0.10)
    return {
        "mean_abs_diff_vs_exact": float(np.abs(a - b).mean()),
        "small_cols": int(small.sum()),
        "small_mean_abs_diff": float(np.abs(a - b)[small].mean()),
        "small_zeroed_frac": float((a[small] < 0.005).mean()),
    }


def reference(run_grades: Path) -> dict:
    """The exact average's and the run's Π's grades at t = 500, from the committed grades file."""
    grades = json.loads(run_grades.read_text())
    exact = grades[f"deep/{T:05d}/exact"]["exploitability"]
    pi = grades[f"deep/{T:05d}/pi"]["exploitability"]
    return {"exact": {"mean_abs_diff_vs_exact": 0.0, "small_mean_abs_diff": 0.0, "small_zeroed_frac": 0.0,
                      "exploitability": round(exact, 4)},
            "run_pi_exploitability": round(pi, 4)}


def main() -> None:
    run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    arms = sys.argv[3].split(",") if len(sys.argv) > 3 else list(ARMS)
    out.parent.mkdir(parents=True, exist_ok=True)
    held = json.loads(out.read_text()) if out.exists() else {}
    loaded = None
    print(f"arms {arms}", flush=True)
    for arm in sorted(arms, key=mark_of, reverse=True):
        if arm in held:
            continue
        mark = mark_of(arm)
        if loaded is None or loaded[0] != mark:
            rows, t, targets = Reservoir.load(run_dir / "marks" / f"t{mark:05d}" / f"policy-t{mark:05d}.npz").contents()
            exact = load_strategy(run_dir / "frozen" / f"t{mark:05d}-exact.npz")
            loaded = (mark,)
            print(f"mark t={mark}: memory {len(rows):,} rows", flush=True)
        s = time.time()
        net = fit(rows, t, targets, arm)
        fit_s = time.time() - s
        strategy = freeze(net, arm)
        save_strategy(strategy, run_dir / "pi-sweep" / f"pi-{arm}.npz")
        s = time.time()
        g = analysis.grade(strategy)
        held[arm] = {**g, **small_prob_stats(strategy, exact), "fit_s": round(fit_s, 1),
                     "grade_s": round(time.time() - s, 1), "arm": dict(zip(
                         ("head", "loss", "steps", "lr", "schedule", "width", "depth", "t"), (*ARMS[arm][:7], mark)))}
        out.write_text(json.dumps(held, indent=2) + "\n")
        print(f"{arm}: exploitability {g['exploitability']:.4f}  fit {fit_s:.0f}s", flush=True)
    held.setdefault("_reference", reference(RUN_GRADES))
    out.write_text(json.dumps(held, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
