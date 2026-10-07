"""Range composition of full Drawmaha on one flop, for the range-UI studies.

Samples N five-card holdings from the 49 unseen cards on the flop Ks 9s 4d and
classifies each twice: INNER (the five hole cards as a poker hand, with
4-flush / 4-straight draw flags) and OUTER (best Omaha hand, exactly 2 hole +
the 3 board cards, with flush-draw / straight-draw flags). The composition is
real (sampling error ~0.1%). The action mix is NOT a solver output: rung 4 is
not trained, so a smooth illustrative policy of each hand's two showdown-now
equities stands in, and is labelled as such everywhere it is drawn.
"""
import json, random, itertools, math, sys
from collections import Counter, defaultdict

random.seed(7)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000

RANKS = '23456789TJQKA'
SUITS = 'shdc'
SUIT_GLYPH = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
FLOP = [(11, 0), (7, 0), (2, 2)]  # Ks 9s 4d  (rank index, suit index)
deck = [(r, s) for r in range(13) for s in range(4) if (r, s) not in FLOP]

CATS = ['high card', 'pair', 'two pair', 'trips', 'straight', 'flush', 'full house', 'quads', 'straight flush']

def eval5(cards):
    """Return (category 0..8, tiebreak tuple) for exactly five cards."""
    ranks = sorted((c[0] for c in cards), reverse=True)
    suits = {c[1] for c in cards}
    cnt = Counter(ranks)
    groups = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    flush = len(suits) == 1
    distinct = sorted(set(ranks), reverse=True)
    straight_hi = None
    if len(distinct) == 5:
        if distinct[0] - distinct[4] == 4:
            straight_hi = distinct[0]
        elif distinct == [12, 3, 2, 1, 0]:
            straight_hi = 3
    if straight_hi is not None and flush:
        return 8, (straight_hi,)
    if groups[0][1] == 4:
        return 7, (groups[0][0], groups[1][0])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return 6, (groups[0][0], groups[1][0])
    if flush:
        return 5, tuple(ranks)
    if straight_hi is not None:
        return 4, (straight_hi,)
    if groups[0][1] == 3:
        return 3, (groups[0][0],) + tuple(g[0] for g in groups[1:])
    if groups[0][1] == 2 and groups[1][1] == 2:
        return 2, (groups[0][0], groups[1][0], groups[2][0])
    if groups[0][1] == 2:
        return 1, (groups[0][0],) + tuple(g[0] for g in groups[1:])
    return 0, tuple(ranks)

def score_int(cat, tb):
    v = cat
    for t in tb + (0,) * (5 - len(tb)):
        v = v * 13 + t
    return v

def straight_outs(rank_sets):
    """Rank-only: how many distinct ranks complete a straight given 4-card rank sets."""
    outs = set()
    windows = [set(range(lo, lo + 5)) for lo in range(0, 9)] + [{12, 0, 1, 2, 3}]
    for four in rank_sets:
        if len(four) != 4:
            continue
        for w in windows:
            if four <= w:
                outs |= (w - four)
    return outs

def inner_class(hole):
    cat, tb = eval5(hole)
    suitc = Counter(c[1] for c in hole)
    ranks = sorted({c[0] for c in hole})
    four_flush = cat < 5 and max(suitc.values()) == 4
    s_outs = set()
    if cat < 4:
        for four in itertools.combinations(ranks, 4):
            s_outs |= straight_outs([set(four)])
        # a 4-straight using 4 distinct ranks of the five
    four_straight = 'open' if len(s_outs) >= 2 else ('gut' if len(s_outs) == 1 else None)
    return cat, score_int(cat, tb), four_flush, four_straight

def outer_class(hole):
    best = (-1, None)
    for two in itertools.combinations(hole, 2):
        cat, tb = eval5(list(two) + FLOP)
        s = score_int(cat, tb)
        if s > best[0]:
            best = (s, cat)
    s, cat = best
    # flush draw: two hole cards of the board's two-suited suit (spades)
    board_suits = Counter(c[1] for c in FLOP)
    fd = False
    for suit, k in board_suits.items():
        if k == 2 and sum(1 for c in hole if c[1] == suit) >= 2:
            fd = True
    if cat >= 5:
        fd = False
    # straight draw: 2 hole ranks + 2 board ranks, 4 distinct in a window
    s_outs = set()
    if cat < 4:
        hr = [c[0] for c in hole]
        br = [c[0] for c in FLOP]
        for h2 in itertools.combinations(sorted(set(hr)), 2):
            for b2 in itertools.combinations(sorted(set(br)), 2):
                four = set(h2) | set(b2)
                if len(four) == 4:
                    s_outs |= straight_outs([four])
    sd = 'wrap' if len(s_outs) >= 3 else ('open' if len(s_outs) == 2 else ('gut' if len(s_outs) == 1 else None))
    return cat, s, fd, sd

def card_str(c):
    return RANKS[c[0]] + SUIT_GLYPH[SUITS[c[1]]]

# ---- sample
hands = []
for _ in range(N):
    hole = random.sample(deck, 5)
    ic, isc, i4f, i4s = inner_class(hole)
    oc, osc, ofd, osd = outer_class(hole)
    hands.append((hole, ic, isc, i4f, i4s, oc, osc, ofd, osd))

# showdown-now equity proxies: percentile of score among sampled holdings
def percentiles(scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    pct = [0.0] * len(scores)
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # P(beat random) + 0.5 P(tie)
        p = (i + 0.5 * (j - i + 1)) / n
        for k in range(i, j + 1):
            pct[order[k]] = p
        i = j + 1
    return pct

pI = percentiles([h[2] for h in hands])
pO = percentiles([h[6] for h in hands])

def sig(x):
    return 1 / (1 + math.exp(-x))

def policy(h, pi, po):
    """Illustrative mix facing a pot bet on the flop: (fold, call, pot)."""
    hole, ic, isc, i4f, i4s, oc, osc, ofd, osd = h
    pi2 = min(1.0, pi + (0.14 if i4f else 0) + (0.06 if i4s == 'open' else 0.03 if i4s == 'gut' else 0))
    po2 = min(1.0, po + (0.16 if ofd else 0) + (0.10 if osd == 'wrap' else 0.07 if osd == 'open' else 0.03 if osd == 'gut' else 0))
    eq = 0.5 * (pi2 + po2)
    scoop = pi2 * po2
    pot = 0.92 * sig(16 * (eq - 0.74)) + 0.55 * sig(18 * (scoop - 0.55))
    # semi-bluff raises with the big draws
    pot += 0.22 * (1 if (ofd and osd) else 0) * (1 - pi2)
    pot = min(0.98, pot)
    fold = 0.97 * sig(14 * (0.47 - eq)) * (1 - pot)
    call = max(0.0, 1 - pot - fold)
    return fold, call, pot, eq, pi2, po2

mix = [policy(h, pI[i], pO[i]) for i, h in enumerate(hands)]

# ---- aggregates
INNER_ROWS = CATS
OUTER_COLS = CATS
grid = defaultdict(lambda: [0, 0.0, 0.0, 0.0])  # n, f, c, p sums
rowtot = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
coltot = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
for h, m in zip(hands, mix):
    key = (h[1], h[5])
    for d, k in ((grid, key), (rowtot, h[1]), (coltot, h[5])):
        d[k][0] += 1; d[k][1] += m[0]; d[k][2] += m[1]; d[k][3] += m[2]

def pack(d):
    out = {}
    for k, (n, f, c, p) in d.items():
        kk = ','.join(map(str, k)) if isinstance(k, tuple) else str(k)
        out[kk] = {'share': n / N, 'f': f / n, 'c': c / n, 'p': p / n}
    return out

# draw flags marginals inside each inner row / outer col (for the ledger's sub-rows)
sub = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
for h, m in zip(hands, mix):
    iflag = '4-flush' if h[3] else ('4-straight' if h[4] else 'no draw')
    oflag = ('flush draw' if h[7] else '') + ('+' if h[7] and h[8] else '') + (f'{h[8]} straight draw' if h[8] else '')
    oflag = oflag or 'no draw'
    k = (h[1], iflag, h[5], oflag)
    sub[k][0] += 1; sub[k][1] += m[0]; sub[k][2] += m[1]; sub[k][3] += m[2]
subrows = [{'inner': CATS[k[0]], 'iflag': k[1], 'outer': CATS[k[2]], 'oflag': k[3], 'share': v[0] / N, 'f': v[1] / v[0], 'c': v[2] / v[0], 'p': v[3] / v[0]} for k, v in sub.items() if v[0] >= 20]

# ribbon: sort by eq, 240 columns
def ribbon(key):
    order = sorted(range(N), key=key)
    cols = 240
    out = []
    for c in range(cols):
        seg = order[c * N // cols:(c + 1) * N // cols]
        f = sum(mix[i][0] for i in seg) / len(seg)
        ca = sum(mix[i][1] for i in seg) / len(seg)
        p = sum(mix[i][2] for i in seg) / len(seg)
        eq = sum(mix[i][3] for i in seg) / len(seg)
        out.append([round(f, 3), round(ca, 3), round(p, 3), round(eq, 3)])
    return out
ribbons = {'total': ribbon(lambda i: mix[i][3]), 'inner': ribbon(lambda i: (mix[i][4], mix[i][3])), 'outer': ribbon(lambda i: (mix[i][5], mix[i][3]))}

# scatter sample
scat_idx = random.sample(range(N), 2600)
scatter = [[round(mix[i][5], 3), round(mix[i][4], 3), round(mix[i][0], 2), round(mix[i][1], 2), round(mix[i][2], 2), hands[i][1], hands[i][5]] for i in scat_idx]
# hexbin density (18x18) over (po2, pi2)
B = 18
dens = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
for i in range(N):
    x = min(B - 1, int(mix[i][5] * B)); y = min(B - 1, int(mix[i][4] * B))
    d = dens[(x, y)]; d[0] += 1; d[1] += mix[i][0]; d[2] += mix[i][1]; d[3] += mix[i][2]
density = [{'x': k[0], 'y': k[1], 'share': v[0] / N, 'f': v[1] / v[0], 'c': v[2] / v[0], 'p': v[3] / v[0]} for k, v in dens.items()]

# deck: how often each unseen card sits in the range (uniform → 5/49 each) and in the POT range
card_pot = defaultdict(float); card_fold = defaultdict(float); card_n = defaultdict(int)
tot_pot = sum(m[2] for m in mix); tot_fold = sum(m[0] for m in mix)
for h, m in zip(hands, mix):
    for c in h[0]:
        card_n[c] += 1; card_pot[c] += m[2]; card_fold[c] += m[0]
deckstats = {card_str(c): {'inRange': card_n[c] / N, 'potShare': card_pot[c] / tot_pot * 49 / 5, 'foldShare': card_fold[c] / tot_fold * 49 / 5} for c in deck}

# the draw node (after calling): illustrative discard-count distribution per inner class
draw = {}
for ic in range(9):
    idx = [i for i in range(N) if hands[i][1] == ic]
    if not idx: continue
    counts = Counter()
    for i in idx:
        h = hands[i]
        if ic >= 2: k = 0 if ic >= 4 else (1 if random.random() < 0.55 else 0)
        elif h[3]: k = 1
        elif h[4] == 'open': k = 1
        elif ic == 1: k = 3 if random.random() < 0.6 else 2
        else: k = 2 if h[7] else (3 if random.random() < 0.5 else 4)
        counts[k] += 1
    draw[CATS[ic]] = {'share': len(idx) / N, 'throw': [counts[k] / len(idx) for k in range(6)]}

# example hands for the leaves: top of each cell by share
examples = defaultdict(list)
for i, (h, m) in enumerate(zip(hands, mix)):
    key = f'{h[1]},{h[5]}'
    if len(examples[key]) < 80:
        examples[key].append({'cards': [card_str(c) for c in sorted(h[0], reverse=True)], 'f': round(m[0], 2), 'c': round(m[1], 2), 'p': round(m[2], 2), 'eqI': round(m[4], 2), 'eqO': round(m[5], 2), 'i4f': h[3], 'i4s': h[4], 'ofd': h[7], 'osd': h[8]})

# marginal mix
F = sum(m[0] for m in mix) / N; C = sum(m[1] for m in mix) / N; P = sum(m[2] for m in mix) / N

out = {
    'N': N, 'flop': [card_str(c) for c in FLOP], 'cats': CATS,
    'overall': {'f': F, 'c': C, 'p': P},
    'grid': pack(grid), 'rows': pack(rowtot), 'cols': pack(coltot), 'subrows': subrows,
    'ribbons': ribbons, 'scatter': scatter, 'density': density, 'deck': deckstats, 'draw': draw, 'examples': examples,
}
with open('research/range-ui-studies/data.js', 'w') as fh:
    fh.write('window.RANGE = ' + json.dumps(out) + ';\n')
print('rows', {CATS[k]: round(v[0] / N, 4) for k, v in sorted(rowtot.items())})
print('cols', {CATS[k]: round(v[0] / N, 4) for k, v in sorted(coltot.items())})
print('overall', round(F, 3), round(C, 3), round(P, 3))
