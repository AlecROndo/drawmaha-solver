# Research lane: abstraction-bucketing

_Generated 2026-08-28 by a 9-agent literature sweep (Semantic Scholar / arXiv / web). Source material for the ML-survey paper._

## Lane narrative

THE STORY. CFR-family solvers need a table entry per information set, and real poker games have astronomically many (two-player no-limit hold'em: ~10^165 nodes; solvable games: ~10^12 — figures from the Ganzfried-Sandholm 2014 paper I read in full). Abstraction is the bridge: build a smaller game whose solution transfers back. The lane's history is a ratchet of better answers to one question — 'which hands should share a strategy?' Layer 0 is lossless: suit isomorphism costs nothing and is provably exact (Gilpin & Sandholm JACM 2007). For Drawmaha I brute-force-verified the payoff in-session: 2,598,960 five-card hands collapse to exactly 134,459 canonical classes (19.3x) — do this regardless of everything else. Layer 1 was scalar bucketing: rank hands by E[HS] (equity vs a random hand) and split into percentile buckets; E[HS^2] over-weights high-variance hands to partially credit draws. Layer 2 realized a scalar hides the shape: 6c6d and KcQc have near-identical E[HS] (0.634 vs 0.633) but opposite equity DISTRIBUTIONS — made hand vs live draw. Distribution-aware abstraction clusters full equity histograms with k-means under earth mover's distance (L2 provably worse — EMD respects how far probability mass moves). Johanson et al. 2013 showed distribution-aware beats expectation-based, and that imperfect recall (forgetting past streets to spend buckets on the present) beats perfect recall at equal size, at the cost of CFR's convergence guarantees. Layer 3, potential-aware (Ganzfried & Sandholm 2014, the champion-agent method): cluster on histograms over NEXT-round buckets, built bottom-up, so trajectory matters, not just destination. Action abstraction ran in parallel: discretize bet sizes, then interpret off-grid bets with the pseudo-harmonic mapping (Ganzfried & Sandholm 2013) — prior nearest/geometric mappings are provably exploitable. Waugh et al. 2009 hangs over everything: refining an abstraction can INCREASE real-game exploitability, so bucket sweeps must be validated by exploitability or head-to-head, never assumed.

THE MODERN QUESTION — does deep learning replace all this? Three answers coexist. (a) Libratus (Science 2018): keep abstraction for a blueprint, patch it with nested real-time subgame solving (which also supersedes action translation for off-tree bets). (b) DeepStack (Science 2017): drop whole-game abstraction; re-solve locally with a learned counterfactual value network. (c) Deep CFR (ICML 2019): run CFR on the full game with networks replacing tables — the abstract literally says it 'obviates the need for abstraction.' The honest synthesis for the survey: neural nets replace the CLUSTERING but not the REPRESENTATION problem. Deep CFR's input featurization (canonical hands, card embeddings, street/pot features) is abstraction moved into the input layer, and everything the potential-aware literature learned about what matters (trajectory, draws, distribution shape) is a checklist for those features.

DRAWMAHA SPECIFICS. (1) Split pot makes strength two-dimensional: a hand has draw-side and Omaha-side equity plus scoop probability, so scalar E[HS] is worse than usual — cluster joint equity vectors, or feed both equities to the network. (2) The single draw round is exactly what potential-aware distance was invented for: pre-draw buckets should be distributions over post-draw buckets (one recursive level). (3) The dashboard needs abstraction twice: pseudo-harmonic translation (or better, small nested re-solves) to interpret real bets, and a coarse display-time bucketing — an equity-histogram k-means into 8-20 named buckets over the 134,459 classes is ideal for a human-readable range view even if the solver itself is bucket-free.

VERIFICATION STATUS. All nine papers confirmed via fetches/API (titles, venues, citation counts from Semantic Scholar batch call 2026-08-28); I read Johanson 2013 and Ganzfried 2014 PDFs directly. From background knowledge, NOT verified this pass: the exact pseudo-harmonic formula f_{A,B}(x) = ((B-x)(1+A))/((B-A)(1+x)); Deep CFR's reservoir-buffer/retraining details and flop-hold'em experiment scale; DeepStack's internal 1,000-bucket input featurization. Not found: any prior published solver work on Drawmaha itself (expected — the variant appears unstudied), and essentially no literature on abstraction for split-pot or 5-card-draw-dimension games; the joint-equity clustering suggestion is my synthesis, not a cited result. Sandholm-2015 citation count not retrieved (rate limit).

## Papers

### Lossless Abstraction of Imperfect Information Games (2007)
_Andrew Gilpin, Tuomas Sandholm · Journal of the ACM 54(5) · 122 citations_
<https://dl.acm.org/doi/10.1145/1284320.1284324>

**Summary:** The founding paper of automated game abstraction. It defines the 'ordered game isomorphism': two game states that differ only in strategically irrelevant details (e.g., which suits the cards happen to be) can be merged into one, and the paper proves any Nash equilibrium of the merged game converts back to an exact equilibrium of the original game — zero loss. Their algorithm, GameShrink, applies these merges automatically and let them solve Rhode Island Hold'em (~3.1 billion nodes) exactly.

**Key Ideas:** Lossless abstraction via symmetry: group information sets that are equivalent under a relabeling (suit permutation being the canonical poker example), solve the smaller game, and lift the strategy back. GameShrink finds these merges in time sublinear in game-tree size.

**Tradeoffs:** Pro: mathematically free — no approximation error at all, so it should always be the first layer of any pipeline. Con: symmetry alone rarely shrinks a game enough (roughly one to two orders of magnitude in poker), so lossy abstraction or function approximation is still needed on top; GameShrink's lossy extension has no quality guarantee.

**Relevance:** Directly actionable: I brute-force-verified that Drawmaha's 2,598,960 five-card hands collapse to exactly 134,459 suit-isomorphic classes (a 19.3x reduction, computed by canonicalizing over all 4! suit permutations). That is the free first move before any bucketing, and the same canonicalization should key the tabular CFR dictionaries, the training-data dedup for Deep CFR, and the dashboard's hand lookup.

### Abstraction Pathologies in Extensive Games (2009)
_Kevin Waugh, David Schnizlein, Michael Bowling, Duane Szafron · AAMAS 2009 · 90 citations_
<https://webdocs.cs.ualberta.ca/~bowling/papers/09aamas-abstraction.pdf>

**Summary:** The cautionary tale of the field. The intuition 'a finer abstraction can only help' is false: the paper gives concrete examples where strictly refining an abstraction (adding more buckets, distinguishing more hands) produces an abstract-game equilibrium that is MORE exploitable in the real game. Abstraction quality is non-monotonic.

**Key Ideas:** Because the opponent in the abstract game is also abstracted, solving a finer abstraction can shift the equilibrium in ways that hurt real-game performance; there is no theoretical guarantee linking abstract-game equilibrium quality to real-game exploitability for lossy abstractions (bounds came only later, and only for special classes).

**Tradeoffs:** This is a negative result, not a method — its value is calibration: bucket-count sweeps must be validated by measuring real-game performance (head-to-head play or best-response exploitability), never assumed from size alone.

**Relevance:** For the shrunken-Drawmaha MCCFR stage: when sweeping bucket counts, evaluate each abstraction by exploitability in the (shrunken) real game or round-robin play, and expect occasional non-monotonic results rather than treating them as bugs. Also the reason the field drifted toward abstraction-free methods.

### Evaluating State-Space Abstractions in Extensive-Form Games (2013)
_Michael Johanson, Neil Burch, Richard Valenzano, Michael Bowling · AAMAS 2013 · 82 citations_
<https://www.ifaamas.org/Proceedings/aamas2013/docs/p271.pdf>

**Summary:** The best single tutorial-and-benchmark on card bucketing (I read the paper itself). It surveys the whole toolbox — E[HS] (expected hand strength = equity vs a random hand averaged over rollouts), E[HS^2], percentile bucketing, nested E[HS^2]/E[HS] (used by Hyperborean in the Annual Computer Poker Competition 2007-09), distribution-aware clustering of equity histograms, and imperfect recall — then evaluates them in two-player limit hold'em (3.2x10^14 information sets) using CFR-BR, which computes the least-exploitable strategy representable in a given abstraction. Findings: distribution-aware abstractions beat expectation-based ones, and imperfect-recall abstractions beat perfect-recall at equal size on both exploitability and head-to-head play.

**Key Ideas:** Represent each hand not by a scalar strength but by its full equity histogram over rollouts; cluster histograms with k-means under earth mover's distance; let the agent 'forget' earlier-round observations (imperfect recall) to spend the bucket budget on the current situation; evaluate abstractions by CFR-BR exploitability rather than proxy metrics.

**Tradeoffs:** E[HS]-style scalars are trivial to compute and fine for a first prototype, but conflate made hands with draws (a scalar cannot see that 6c6d and KcQc have near-identical E[HS] but opposite equity distributions). Distribution-aware clustering fixes this at the cost of computing per-hand histograms and an EMD k-means pass. Imperfect recall gives strictly better use of a fixed bucket budget but voids CFR's convergence guarantees (empirically it still works).

**Relevance:** This is the recipe paper for the shrunken-Drawmaha stage: compute each 5-card hand's equity histogram (vs random opponent hands, over draw outcomes and board runouts), k-means them with EMD. Crucially, Drawmaha is split-pot, so 'equity' is two-dimensional (high-hand share and Omaha-board share); cluster on the joint distribution or on (high-equity, low/board-equity, scoop-probability) features — a scalar E[HS] is even more misleading here than in hold'em.

### Action Translation in Extensive-Form Games with Large Action Spaces: Axioms, Paradoxes, and the Pseudo-Harmonic Mapping (2013)
_Sam Ganzfried, Tuomas Sandholm · IJCAI 2013 · 31 citations_
<https://dl.acm.org/doi/abs/10.5555/2540128.2540148>

**Summary:** The action-abstraction counterpart of card bucketing. A solver trained on a betting grid (e.g., half-pot, pot, all-in) must interpret an opponent bet that falls between grid points; the rule doing that is the action translation mapping. The paper shows the previously used mappings (deterministic nearest-bet, geometric/arithmetic randomized) violate natural axioms and are badly exploitable — an opponent can size bets to be systematically misread — and derives the pseudo-harmonic mapping, which satisfies the axioms and is provably far less exploitable, calibrated from analytical solutions of simplified poker games.

**Key Ideas:** Randomize between the two neighboring abstract bet sizes A < x < B with a probability derived from game-theoretic first principles (the mapping is exact in a clairvoyance toy game) rather than from geometric distance; desiderata include boundary consistency, monotonicity, and robustness to small perturbations.

**Tradeoffs:** Pro: drop-in, no retraining, the standard for every abstraction-based agent since. Con: any translation is still a patch — Libratus-style nested subgame solving (re-solving off-tree bets exactly in real time) dominates it when compute allows; translation also interacts with bet-size choice in the abstraction (put grid points where opponents actually bet).

**Relevance:** Pot-limit Drawmaha has a continuous-ish bet space capped at pot, so the trained tree will use a small bet grid (e.g., 33%/66%/pot). Both the live-inference dashboard (interpreting a real opponent's $37 bet into a $30-or-$45 grid) and any bot-vs-bot evaluation need this mapping; implement pseudo-harmonic, not nearest-neighbor. Discard sizes 0-5 are a second, naturally discrete action dimension that needs no translation.

### Potential-Aware Imperfect-Recall Abstraction with Earth Mover's Distance in Imperfect-Information Games (2014)
_Sam Ganzfried, Tuomas Sandholm · AAAI 2014 · 60 citations_
<https://ojs.aaai.org/index.php/AAAI/article/view/8816>

**Summary:** The state-of-the-art card abstraction before neural nets, and the one inside CMU's champion agents (I read the paper itself). Distribution-aware abstraction (Johanson et al.) clusters hands by their histogram of FINAL-round equity; this paper argues that is still myopic — two hands can share a final equity distribution but realize it along different trajectories (when the draw/board hits matters strategically). Potential-aware abstraction instead represents a hand by its histogram over NEXT-round buckets (built bottom-up from the last round), capturing the whole trajectory of strength, and clusters with a fast custom EMD approximation in this multi-dimensional space. Statistically significant head-to-head improvement over the previous best abstraction in no-limit hold'em (a game of ~10^165 nodes solved via ~10^12-node abstractions).

**Key Ideas:** Bottom-up recursive bucketing: cluster the final round on equity histograms, then represent each earlier-round hand as a probability vector over the next round's clusters and cluster those vectors with (approximate multi-dimensional) EMD; combine with imperfect recall so each round's budget is spent on the present.

**Tradeoffs:** Pro: best-quality fixed abstraction known; exactly targets games where hands transform (draws). Con: the most engineering-heavy option — recursive passes, custom EMD heuristics (exact multi-dimensional EMD is far too slow), and imperfect recall costs convergence guarantees; overkill for a first prototype vs plain equity-histogram k-means.

**Relevance:** Conceptually the best match for Drawmaha's draw round: a pat straight and a 4-flush with a redraw differ precisely in trajectory, which is what potential-aware distance measures. Pre-draw buckets should be built from distributions over post-draw buckets (the paper's exact recipe, one level deep since Drawmaha has one draw). Realistic scope: use it for the MCCFR-on-shrunken-game stage and as the feature design for what the Deep CFR network must implicitly learn.

### Abstraction for Solving Large Incomplete-Information Games (2015)
_Tuomas Sandholm · AAAI 2015 (Senior Member track survey)_
<https://ojs.aaai.org/index.php/AAAI/article/view/9757>

**Summary:** A compact survey of the whole abstract-solve-lift paradigm by the field's central figure: why abstraction is needed, lossless vs lossy information abstraction, action abstraction, the abstraction-pathology problem, and the (then-new) theoretical results bounding real-game strategy quality from abstract-game solutions.

**Key Ideas:** Frames the canonical three-step pipeline — abstract the game, compute an approximate equilibrium in the abstraction, map the strategy back — and organizes all the individual techniques (isomorphisms, bucketing, imperfect recall, bet discretization, translation) as choices within it.

**Tradeoffs:** A survey, so no new method; its value is the map. Its theory sections (bounded-loss abstraction) are honest that the bounds are too loose to guide practical poker-scale abstraction, which remains empirical.

**Relevance:** The single best background read for the survey's audience: it supplies the vocabulary connecting every other paper in this lane, and its pipeline framing is exactly the tabular-CFR-then-MCCFR-then-Deep-CFR plan, with abstraction as the dial that changes between stages.

### DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker (2017)
_Matej Moravčík, Martin Schmid, Neil Burch, Viliam Lisý, Dustin Morrill, Nolan Bard, Trevor Davis, Kevin Waugh, Michael Johanson, Michael Bowling · Science 356 (arXiv:1701.01724) · 1029 citations_
<https://arxiv.org/abs/1701.01724>

**Summary:** The first demonstration that a whole-game card abstraction can be replaced by local search plus a learned value function. Instead of pre-solving an abstracted game, DeepStack re-solves only the current public subtree at decision time (continual re-solving), truncating lookahead with deep 'counterfactual value networks' — trained on millions of randomly generated subgames — that act as learned intuition for how the rest of the game plays out. It beat professional players with statistical significance over 44,000 hands of heads-up no-limit hold'em and produced strategies harder to exploit than abstraction-based agents.

**Key Ideas:** Decomposition (solve only the subgame you are in, carrying opponent ranges and counterfactual values as the interface), depth-limited lookahead, and a neural network mapping (ranges, pot size, board) to per-hand counterfactual values in place of solving to the end of the game.

**Tradeoffs:** Pro: sidesteps whole-game bucketing and most abstraction pathologies; memory footprint tiny compared to blueprint tables. Con: needs substantial per-decision compute at play time (a re-solve per action) — heavy for a live dashboard; still uses a sparse bet menu in lookahead, and the value networks must be trained on well-distributed random subgames, itself a design problem. (Note: its networks internally still bucket hands for input featurization — from background knowledge, not verified in this pass.)

**Relevance:** The main architectural alternative to the planned Deep CFR route: rather than one average-policy network for all of Drawmaha, train a post-draw value network and re-solve the pre-draw/draw streets live. Probably a v2 idea — but its interface concept (opponent range over 134k canonical hands as first-class object) is exactly what the dashboard's range view must display anyway.

### Superhuman AI for Heads-Up No-Limit Poker: Libratus Beats Top Professionals (2018)
_Noam Brown, Tuomas Sandholm · Science 359 · 843 citations_
<https://www.science.org/doi/10.1126/science.aao1733>

**Summary:** The abstraction-based counterpoint to DeepStack, and proof the classical pipeline scales when combined with real-time search. Libratus beat four top professionals over 120,000 hands using three modules: a blueprint strategy solved by MCCFR on an abstracted game (card bucketing plus a bet-size grid), nested safe subgame solving that re-solves any subgame actually reached — including opponent bets outside the grid, replacing action translation with exact real-time response — and a self-improver that added the opponents' most-used off-tree bet sizes into the abstraction overnight.

**Key Ideas:** Coarse abstraction is fine for a blueprint if every strategically important situation gets refined by real-time re-solving; 'nested' means each off-tree opponent action spawns a fresh subgame solve whose solution is guaranteed (in the safe variant) not to increase exploitability. Notably it used unsafe subgame solving once on first reaching the third betting round, and safe solving thereafter.

**Tradeoffs:** Pro: keeps tabular CFR's reliability and interpretability; abstraction quality matters less because search patches it. Con: supercomputer-scale blueprint compute and nontrivial real-time solving infrastructure; card abstraction is still present, so trajectory-blind buckets can still leak in un-resolved portions.

**Relevance:** Defines the modern tradeoff the survey must present: abstraction + search (Libratus) vs learned values + search (DeepStack) vs learned CFR, no search (Deep CFR). For Drawmaha, the takeaway is that the MCCFR-on-abstraction stage is not just scaffolding — with even shallow subgame re-solving at dashboard time, it can be a legitimate final system, and nested re-solving is the principled upgrade over pseudo-harmonic translation for off-grid bets.

### Deep Counterfactual Regret Minimization (2019)
_Noam Brown, Adam Lerer, Sam Gross, Tuomas Sandholm · ICML 2019 (arXiv:1811.00164) · 263 citations_
<https://arxiv.org/abs/1811.00164>

**Summary:** The paper the Drawmaha plan is built on. Deep CFR runs CFR-style traversals of the FULL unabstracted game, but instead of tabular regrets it trains a neural network per player to predict counterfactual regret (an 'advantage' network) from sampled traversals, plus a final network that fits the time-averaged strategy. The abstract's explicit claim: it 'obviates the need for abstraction by instead using deep neural networks to approximate the behavior of CFR in the full game,' and it is the first non-tabular CFR variant to succeed in large games — the network generalizes across similar hands, which is bucketing learned end-to-end instead of hand-engineered.

**Key Ideas:** Alternate: sample traversals of the real game with the current networks defining strategies (regret matching on predicted advantages); store (infoset, sampled regret) targets in replay buffers; periodically retrain the advantage networks from scratch; fit the average-strategy network last — that average policy is the equilibrium approximation (and the thing the dashboard queries).

**Tradeoffs:** Pro: no bucketing pipeline to design, no pathology from a frozen abstraction, and the card-input featurization becomes an ML problem the author already has the skills for. Con: convergence is only approximate (network error compounds), needs careful buffer/retraining hygiene, and evaluating exploitability of a neural policy is itself hard; in head-to-head published comparisons it beat abstraction-based MCCFR baselines but tabular-on-good-abstraction remains competitive at moderate scale. (Details from background knowledge, flagged in notes: experiments were on large hold'em subgames/flop hold'em, with reservoir-sampled buffers and per-iteration advantage retraining; linear weighting of later iterations.)

**Relevance:** The load-bearing citation for the whole project plan. Two Drawmaha-specific notes: (1) 'no abstraction' is not literal — suit-isomorphic canonicalization of the 5-card hand (134,459 classes) plus a permutation-invariant card encoding IS the abstraction, moved into the input layer, and getting that featurization right (card embeddings, draw-count, split-pot board features) is the project's real modeling work; (2) the average-policy network the dashboard queries is exactly Deep CFR's final network, so the plan's inference story is native to this method.

