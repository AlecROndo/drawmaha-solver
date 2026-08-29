# Handoff breadcrumb — drawmaha-solver grill/clarification state (2026-08-29)

Facts from the prior conversation that are NOT rediscoverable from the repo.
Everything else (rules, ladder, research, paper) IS in the repo — read those
files rather than trusting summaries.

## Purpose (Alec's own framing)
CV/portfolio project + vehicle to learn applied ML/RL/DL. Getting better at
drawmaha HU is a side effect, not the goal. Alec is NEW to both CFR and
PyTorch — explain concepts intuitively as decisions come up (he is taking
CS672, ML theory; knows sklearn/boosting/basic RL: actor-critic, reward fns).

## Grill ledger (interview run under the grill-me skill, one question at a time)
Decided (also reflected in README.md):
- Q1 purpose: portfolio/learning (above).
- Q2 game: standard Drawmaha + face-up draw-1 house rule (draw exactly 1 →
  replacement dealt face up, drawer may keep it or reject for face-down 2nd).
- Q3 betting: v1 action set {fold, check/call, pot} ONLY; bet-size menus are
  config so 2-size/custom solves later = re-solve, not rewrite. 100bb default.
- Q4 product: GTOWizard-style dashboard FIRST; trainer mode later as a thin
  quiz layer over the same query API.
- Q5 method: CFR family via validation ladder (RPS → Kuhn → Leduc →
  mini-drawmaha MCCFR → full-game Deep CFR). PPO parked as extension chapter.
- Q6 dashboard data path: LIVE inference on node click (batch-evaluate policy
  net + reach probabilities), NO precomputed solve library. In-memory
  memoization of visited nodes is fine (invisible plumbing).
- Q7 range view: interpretable board-dependent category buckets, filterable by
  INNER hand (5 hole cards as poker hand) and/or OUTER hand (Omaha half) —
  Alec's terminology; plus exact-hand text input. Learned clustering demoted
  to an analysis chapter, not the UI.
- Q8 stack: Python 3.12 + NumPy + PyTorch + uv + pytest; FastAPI backend +
  React/TypeScript (Vite) dashboard. Compute default: Mac first, Modal GPU
  for rung 4 only if needed.

PENDING — the grill stopped here:
- Q9 evaluation story (three legs proposed, NOT locked: (1) rung-3-vs-rung-4
  truth check on mini-drawmaha; (2) scripted best-response/LBR-style probe as
  full-scale exploitability meter; (3) head-to-head baseline ladder with
  duplicate dealing; learned actor-critic exploiter parked in PPO extension).
  Alec paused decisions to first understand the field → the survey paper was
  produced. Resume by re-asking Q9 informed by paper Sections 11/14.
- Never-asked branches: working mode (how much Alec writes vs Claude writes —
  he wants to LEARN, so ask), timeline/pacing, mini-drawmaha exact design
  (deck/hand size — flagged as open item C-last in paper Section 14).

## Paper status
26-page survey at paper/drawmaha-ml-survey.pdf, compiled clean (tectonic),
built in lecture-report house style per Alec's request. Author-side visual
sweep done; two table overflows already fixed. TWO CRITIC AGENTS
(accuracy-critic, format-critic) were dispatched in the old session and went
silent ~22h; they were pinged for final reports at handoff time — check
.context/ for any critic-findings file saved after this one; if none exists,
the one-round critic pass (paper skill step 7) was never completed. Either
re-run two fresh critics on the PDF or ask Alec if he wants it.
Delivery also incomplete: PDF not yet copied to ~/Desktop/Claude/poker/.

## Other conversation facts
- Compute estimate given to Alec: ~$200–700 total done properly (min ~$100,
  ceiling ~$1.5k); engine speed and restart count dominate, not GPU tier;
  mid-tier GPU (L40S/A10G) suffices for 99K-param nets; Modal for bursts,
  Nebius only if single long runs dominate.
- Semantic Scholar API key now in ~/.config/personal/.env
  (SEMANTIC_SCHOLAR_API_KEY, verified live 2026-08-29); arXiv needs no key.
- Git identity: commits must NOT carry Claude co-author attribution
  (user-level settings.json attribution.commit/pr = "" — already set).
- Repo remote: https://github.com/AlecROndo/drawmaha-solver (private).
