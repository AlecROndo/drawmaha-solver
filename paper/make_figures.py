"""Figures for the drawmaha ML survey. Real simulations where possible.

Run:  uv run --with matplotlib --with numpy --no-project python3 make_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ---- house palette (lecture-report.sty) --------------------------------------
INK = "#1A202C"; SLATE = "#52616F"; TEAL = "#0F7B6C"; BLUE = "#2B6CB0"
RED = "#C53030"; HAIR = "#C9CFD6"; TEALPALE = "#EDF7F3"; PALEBLUE = "#EEF2F7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.edgecolor": HAIR, "axes.labelcolor": INK, "axes.titlesize": 10,
    "xtick.color": SLATE, "ytick.color": SLATE, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

FIG = "figures/"

# =============================================================================
# 1. RPS regret matching: current strategy swings, average converges (REAL sim)
# =============================================================================
def fig_rps():
    rng = np.random.default_rng(7)
    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])  # row payoff R,P,S
    T = 3000
    reg = [np.zeros(3), np.zeros(3)]
    cur_hist, avg_hist = [], []
    strat_sum = [np.zeros(3), np.zeros(3)]

    def rm(r):
        p = np.maximum(r, 0)
        return p / p.sum() if p.sum() > 0 else np.ones(3) / 3

    for t in range(T):
        s = [rm(reg[0]), rm(reg[1])]
        strat_sum[0] += s[0]; strat_sum[1] += s[1]
        a0 = rng.choice(3, p=s[0]); a1 = rng.choice(3, p=s[1])
        u0 = A[:, a1]          # what each of my actions pays vs their sampled action
        u1 = -A[a0, :]
        reg[0] += u0 - u0[a0]
        reg[1] += u1 - u1[a1]
        cur_hist.append(s[0].copy())
        avg_hist.append(strat_sum[0] / (t + 1))

    cur = np.array(cur_hist); avg = np.array(avg_hist)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    names = ["Rock", "Paper", "Scissors"]; cols = [BLUE, TEAL, RED]
    for i in range(3):
        axes[0].plot(cur[:, i], color=cols[i], lw=0.7, alpha=0.85)
        axes[1].plot(avg[:, i], color=cols[i], lw=1.6, label=names[i])
    for ax in axes:
        ax.axhline(1/3, color=HAIR, lw=0.8, ls="--")
        ax.set_xlabel("iteration"); ax.set_ylim(-0.03, 1.03)
    axes[0].set_title("current strategy: swings forever", color=INK)
    axes[1].set_title("running average: settles at 1/3 each", color=INK)
    axes[0].set_ylabel("P(action), player 1")
    axes[1].annotate("Nash: 1/3", xy=(T * 0.72, 1/3), xytext=(T * 0.72, 0.52),
                     color=SLATE, fontsize=8.5,
                     arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.7))
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(FIG + "rps.png"); plt.close(fig)

# =============================================================================
# 2. Kuhn poker: REAL vanilla CFR vs CFR+ exploitability curves
# =============================================================================
CARDS = [0, 1, 2]  # J,Q,K
DEALS = [(a, b) for a in CARDS for b in CARDS if a != b]  # prob 1/6 each

def terminal_value(h, c1, c2):
    """Value to player 1 at terminal history h ('' actions p=pass, b=bet)."""
    if h == "pp":  return 1 if c1 > c2 else -1
    if h == "bp":  return 1            # P1 bet, P2 folded
    if h == "pbp": return -1           # P2 bet, P1 folded
    if h in ("bb", "pbb"): return 2 if c1 > c2 else -2
    return None

class Node:
    def __init__(self):
        self.regret = np.zeros(2)
        self.strat_sum = np.zeros(2)
    def strategy(self, plus):
        r = np.maximum(self.regret, 0)
        return r / r.sum() if r.sum() > 0 else np.ones(2) / 2

def cfr(nodes, h, c1, c2, p1, p2, plus, t, update_player=None):
    tv = terminal_value(h, c1, c2)
    if tv is not None:
        return tv
    player = len(h) % 2
    card = c1 if player == 0 else c2
    key = (card, h)
    node = nodes.setdefault(key, Node())
    sigma = node.strategy(plus)
    vals = np.zeros(2)
    node_v = 0.0
    for i, a in enumerate("pb"):
        if player == 0:
            vals[i] = cfr(nodes, h + a, c1, c2, p1 * sigma[i], p2, plus, t, update_player)
        else:
            vals[i] = cfr(nodes, h + a, c1, c2, p1, p2 * sigma[i], plus, t, update_player)
        node_v += sigma[i] * vals[i]
    my_reach = p1 if player == 0 else p2
    opp_reach = p2 if player == 0 else p1
    sign = 1 if player == 0 else -1
    if update_player is None or update_player == player:
        node.regret += opp_reach * sign * (vals - node_v)
        if plus:
            node.strat_sum += t * my_reach * sigma        # linear averaging
        else:
            node.strat_sum += my_reach * sigma
    return node_v

def avg_policy(nodes):
    pol = {}
    for k, n in nodes.items():
        s = n.strat_sum
        pol[k] = s / s.sum() if s.sum() > 0 else np.ones(2) / 2
    return pol

def _br(pol, h, my_card, reach, me):
    """Value to best-responder `me` holding my_card at public history h,
    aggregated over opponent cards weighted by `reach` (chance x their policy).
    Maximizing per INFOSET (my card + history), never per deal — the best
    responder cannot see the opponent's card."""
    probe = next(iter(reach))
    c1, c2 = (my_card, probe) if me == 0 else (probe, my_card)
    if terminal_value(h, c1, c2) is not None:
        out = 0.0
        for o, w in reach.items():
            c1, c2 = (my_card, o) if me == 0 else (o, my_card)
            tv = terminal_value(h, c1, c2)
            out += w * (tv if me == 0 else -tv)
        return out
    player = len(h) % 2
    if player == me:
        return max(_br(pol, h + a, my_card, reach, me) for a in "pb")
    out = 0.0
    for i, a in enumerate("pb"):
        new_reach = {o: w * pol.get((o, h), np.ones(2) / 2)[i]
                     for o, w in reach.items()}
        out += _br(pol, h + a, my_card, new_reach, me)
    return out

def exploitability(pol):
    v = 0.0
    for me in (0, 1):
        for my_card in CARDS:
            opp = [c for c in CARDS if c != my_card]
            reach = {o: 1.0 / len(DEALS) for o in opp}
            v += _br(pol, "", my_card, reach, me)
    return v / 2  # per-player exploitability; 0 at Nash

def game_value(pol):
    """Expected value to P1 when both play pol (sanity: Kuhn Nash = -1/18)."""
    def walk(h, c1, c2):
        tv = terminal_value(h, c1, c2)
        if tv is not None:
            return tv
        player = len(h) % 2
        card = c1 if player == 0 else c2
        sigma = pol.get((card, h), np.ones(2) / 2)
        return sum(sigma[i] * walk(h + a, c1, c2) for i, a in enumerate("pb"))
    return sum(walk("", c1, c2) for c1, c2 in DEALS) / len(DEALS)

def run_kuhn(plus, T, checkpoints):
    nodes = {}
    xs, ys_avg, ys_cur = [], [], []
    for t in range(1, T + 1):
        if plus:  # alternating updates + once-per-iteration clip (regret-matching+)
            for up in (0, 1):
                for c1, c2 in DEALS:
                    cfr(nodes, "", c1, c2, 1.0, 1.0, True, t, update_player=up)
                for (card, h), node in nodes.items():
                    if len(h) % 2 == up:
                        node.regret = np.maximum(node.regret, 0)
        else:
            for c1, c2 in DEALS:
                cfr(nodes, "", c1, c2, 1.0, 1.0, False, t)
        if t in checkpoints:
            xs.append(t)
            ys_avg.append(exploitability(avg_policy(nodes)))
            ys_cur.append(exploitability({k: n.strategy(plus)
                                          for k, n in nodes.items()}))
    return xs, ys_avg, ys_cur, nodes

def fig_kuhn():
    T = 10000
    checkpoints = sorted(set(int(x) for x in np.logspace(0, 4, 40)))
    xs, av_v, cu_v, nodes_v = run_kuhn(False, T, checkpoints)
    _, av_p, cu_p, nodes_p = run_kuhn(True, T, checkpoints)
    gv = game_value(avg_policy(nodes_v))
    print(f"[kuhn] vanilla avg-policy game value = {gv:.5f}  (Nash: {-1/18:.5f})")
    print(f"[kuhn] final expl: van-avg={av_v[-1]:.1e} van-cur={cu_v[-1]:.1e} "
          f"plus-avg={av_p[-1]:.1e} plus-cur={cu_p[-1]:.1e}")
    assert abs(gv - (-1/18)) < 5e-3, "Kuhn CFR sanity check failed"

    fig, ax = plt.subplots(figsize=(4.85, 3.2))
    ax.loglog(xs, cu_v, color=BLUE, lw=1.1, ls=":",
              label="vanilla CFR -- current strategy")
    ax.loglog(xs, av_v, color=BLUE, lw=1.8, label="vanilla CFR -- average")
    ax.loglog(xs, av_p, color=TEAL, lw=1.8, label="CFR+ -- average")
    ax.loglog(xs, cu_p, color=TEAL, lw=1.1, ls=":",
              label="CFR+ -- current strategy")
    guide = av_v[6] * np.sqrt(xs[6]) / np.sqrt(np.array(xs, float))
    ax.loglog(xs, guide, color=HAIR, lw=1.0, ls="--")
    ax.annotate(r"$1/\sqrt{T}$ guide", xy=(xs[-8], guide[-8]), xytext=(-4, 7),
                textcoords="offset points", color=SLATE, fontsize=8.5)
    ax.annotate("cycles forever:\nnever converges", xy=(xs[-12], cu_v[-12]),
                xytext=(-78, -6), textcoords="offset points",
                color=BLUE, fontsize=8)
    ax.set_xlabel("CFR iterations $T$")
    ax.set_ylabel("exploitability (chips/hand, antes of 1)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Kuhn poker, solved live for this figure", color=INK)
    fig.savefig(FIG + "kuhn_cfr.png"); plt.close(fig)

# =============================================================================
# 3. Self-play dynamics: orbits vs regularized contraction (real vector fields)
# =============================================================================
def fig_cycle():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    fig.subplots_adjust(wspace=0.18)
    lim = 1.15
    Y, X = np.mgrid[-lim:lim:24j, -lim:lim:24j]
    for ax, lam, title in [
        (axes[0], 0.0, "plain self-play: orbits forever"),
        (axes[1], 0.35, "with a magnet term: contracts"),
    ]:
        U = Y - lam * X
        V = -X - lam * Y
        ax.streamplot(X, Y, U, V, color=SLATE, linewidth=0.7, density=0.85,
                      arrowsize=0.8)
        ax.plot(0, 0, "o", color=RED, ms=6, zorder=5)
        ax.annotate("equilibrium", xy=(0, 0), xytext=(0.15, -0.9),
                    color=RED, fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
        # one actual trajectory
        x, y = 0.9, 0.0
        tr = [(x, y)]
        for _ in range(2600):
            dt = 0.01
            x, y = x + dt * (y - lam * x), y + dt * (-x - lam * y)
            tr.append((x, y))
        tr = np.array(tr)
        ax.plot(tr[:, 0], tr[:, 1], color=TEAL, lw=1.6)
        ax.set_title(title, color=INK)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("player 1's strategy deviation")
    axes[0].set_ylabel("player 2's strategy deviation")
    fig.savefig(FIG + "cycle.png"); plt.close(fig)

# =============================================================================
# 4. Vanilla vs external vs outcome sampling: what each traversal touches
# =============================================================================
def draw_tree(ax, mode, rng):
    """Depth-4 alternating tree. mode: 'full' | 'external' | 'outcome'."""
    levels = 4
    def recurse(x, y, depth, w, visited):
        if depth == levels:
            return
        player = depth % 2  # 0 = traverser, 1 = opponent/chance
        for i, dx in enumerate([-w, w]):
            child_visited = visited
            if visited:
                if mode == "full":
                    child_visited = True
                elif mode == "external":
                    child_visited = (player == 0) or (i == rng.integers(2))
                else:  # outcome
                    child_visited = (i == rng.integers(2))
            cx, cy = x + dx, y - 1
            ax.plot([x, cx], [y, cy],
                    color=TEAL if child_visited else HAIR,
                    lw=1.9 if child_visited else 0.9, zorder=1)
            marker = "o" if (depth + 1) % 2 == 1 else "s"
            ax.plot(cx, cy, marker,
                    color=TEAL if child_visited else HAIR, ms=3.4, zorder=2)
            recurse(cx, cy, depth + 1, w / 2, child_visited)
    ax.plot(0, 0, "s", color=TEAL, ms=4.5, zorder=2)
    recurse(0, 0, 0, 2.0, True)
    ax.set_xlim(-4.4, 4.4); ax.set_ylim(-4.4, 0.5)
    ax.axis("off")

def fig_sampling():
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2))
    titles = ["vanilla CFR:\nwalk everything",
              "external sampling:\nall my actions, sample theirs",
              "outcome sampling:\none playout"]
    for ax, mode, title in zip(axes, ["full", "external", "outcome"], titles):
        draw_tree(ax, mode, rng)
        ax.set_title(title, color=INK, fontsize=9)
    handles = [plt.Line2D([], [], marker="s", ls="", color=TEAL,
                          label="my decision node"),
               plt.Line2D([], [], marker="o", ls="", color=TEAL,
                          label="opponent / chance node")]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.savefig(FIG + "sampling.png"); plt.close(fig)

# =============================================================================
# 5. Same average equity, opposite shapes (illustrative histograms)
# =============================================================================
def fig_equity():
    rng = np.random.default_rng(11)
    made = np.clip(rng.normal(0.54, 0.07, 60000), 0, 1)
    hit = rng.random(60000) < 0.45
    draw = np.where(hit, rng.normal(0.88, 0.045, 60000),
                    rng.normal(0.26, 0.055, 60000))
    draw = np.clip(draw, 0, 1)
    draw += (0.54 - draw.mean())  # pin the means equal
    fig, ax = plt.subplots(figsize=(4.7, 2.6))
    bins = np.linspace(0, 1, 55)
    ax.hist(made, bins=bins, density=True, color=BLUE, alpha=0.55,
            label="made hand (two pair)")
    ax.hist(draw, bins=bins, density=True, color=RED, alpha=0.5,
            label="big draw (4-flush + overcards)")
    ax.axvline(0.54, color=INK, lw=1.1, ls="--")
    ax.annotate("both average 0.54", xy=(0.54, ax.get_ylim()[1] * 0.93),
                xytext=(8, 0), textcoords="offset points", color=INK, fontsize=8.5)
    ax.set_xlabel("equity at showdown vs a random hand")
    ax.set_ylabel("density"); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(FIG + "equity.png"); plt.close(fig)

# =============================================================================
# 6. Compute scale of landmark systems (log axis, from reported hardware)
# =============================================================================
def fig_compute():
    systems = [
        ("Kuhn CFR (solved for this paper)", 1e-4, "seconds on a laptop"),
        ("Exact best response, limit hold'em\n(Johanson 2011)", 76, "76 CPU-days"),
        ("Ostroumov 2-7 draw solver, one run\n(2013--15)", 3000, "~1{,}000 cores x 3 days"),
        ("Deep CFR on Flop Hold'em\n(Brown 2019)", 10, "~single GPU-class"),
        ("Cepheus: limit hold'em solved\n(Bowling 2015)", 330000, "4{,}800 CPUs x 68 days"),
        ("ReBeL data generation\n(Brown 2020)", 1e6, "720 V100 GPUs"),
    ]
    systems = sorted(systems, key=lambda s: s[1])
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    names = [s[0] for s in systems]; vals = [s[1] for s in systems]
    cols = [TEAL if "Deep CFR" in n or "Kuhn" in n else BLUE for n in names]
    y = np.arange(len(systems))
    ax.barh(y, vals, color=cols, height=0.62, log=True)
    ax.set_yticks(y, [n.replace("\\n", "\n") for n in names], fontsize=8)
    for i, (n, v, lab) in enumerate(systems):
        ax.text(v * 1.6, i, lab.replace("{,}", ","), va="center",
                fontsize=8, color=SLATE)
    ax.set_xlim(3e-5, 3e8)
    ax.set_xlabel("approximate compute (CPU-core-days, log scale; GPU entries approximate)")
    ax.invert_yaxis()
    fig.savefig(FIG + "compute.png"); plt.close(fig)


# =============================================================================
# 7. Drawmaha street structure, two rows (replaces too-wide dot version)
# =============================================================================
def fig_streets():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    boxes = [  # (label, color, textcolor) in play order
        ("Deal\n5 + 5", "#F4F5F3", INK), ("Bet\n(preflop)", PALEBLUE, INK),
        ("Flop\n3 cards", "#F4F5F3", INK), ("Bet", PALEBLUE, INK),
        ("DRAW\ndiscard 0–5\n(count public)", TEALPALE, TEAL),
        ("Turn", "#F4F5F3", INK), ("Bet", PALEBLUE, INK),
        ("River", "#F4F5F3", INK), ("Bet", PALEBLUE, INK),
        ("Showdown\nsplit pot", "#FBEAE8", RED),
    ]
    W, H, GX = 1.7, 0.9, 0.5
    pos = []
    for i in range(5):
        pos.append((i * (W + GX), 1.6))       # top row, left to right
    for i in range(5):
        pos.append(((4 - i) * (W + GX), 0.0)) # bottom row, right to left
    for (label, fc, tc), (x, y) in zip(boxes, pos):
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.06",
                                    fc=fc, ec=TEAL if fc == TEALPALE else HAIR,
                                    lw=1.2))
        ax.text(x + W / 2, y + H / 2, label, ha="center", va="center",
                fontsize=8.2, color=tc)
    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", color=SLATE,
                                     lw=1.1, mutation_scale=11, zorder=1))
    for i in range(4):  # top row arrows ->
        arrow((pos[i][0] + W + 0.06, pos[i][1] + H / 2),
              (pos[i + 1][0] - 0.06, pos[i + 1][1] + H / 2))
    arrow((pos[4][0] + W / 2, pos[4][1] - 0.08),   # wrap down
          (pos[5][0] + W / 2, pos[5][1] + H + 0.08))
    for i in range(5, 9):  # bottom row arrows <- (leftward in x)
        arrow((pos[i][0] - 0.06, pos[i][1] + H / 2),
              (pos[i + 1][0] + W + 0.06, pos[i + 1][1] + H / 2))
    ax.set_xlim(-0.3, 5 * (W + GX)); ax.set_ylim(-0.35, 2.85)
    ax.axis("off")
    fig.savefig(FIG + "streets.png"); plt.close(fig)

# =============================================================================
if __name__ == "__main__":
    import os
    os.makedirs(FIG, exist_ok=True)
    fig_rps(); print("rps done")
    fig_kuhn(); print("kuhn done")
    fig_cycle(); print("cycle done")
    fig_sampling(); print("sampling done")
    fig_equity(); print("equity done")
    fig_compute(); print("compute done")
    fig_streets(); print("streets done")
