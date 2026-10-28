# Research lane: cfr-foundations

_Generated 2026-08-28 by a 9-agent literature sweep (Semantic Scholar / arXiv / web). Source material for the ML-survey paper._

## Lane narrative

THE STORY. This lane answers one question: why does iterating a dumb local update rule produce a Nash equilibrium in a game as adversarial as poker? The chain has three links. (1) Hart & Mas-Colell 2000: regret matching — play actions in proportion to accumulated positive regret — is a no-regret learner. (2) The folk theorem: in a TWO-PLAYER ZERO-SUM game, if both players' average regret over T iterations is ε, the pair of TIME-AVERAGED strategies is a 2ε-Nash equilibrium. Regret provably shrinks as O(1/√T), so average strategies converge to Nash — while the current strategies cycle forever and converge to nothing. This is the single most common conceptual bug: always output/query the average policy (Deep CFR's average-policy network is exactly this). (3) Zinkevich et al. 2007: 'counterfactual regret' decomposes regret across information sets — a local regret-matcher at every decision point, with values weighted by the probability that OPPONENT + CHANCE reach that infoset — and the sum of local regrets bounds global regret. That decomposition is what makes the tree tractable, and everything since 2007 is an efficiency layer on top of it.

THE OPTIONS, as a builder's decision tree. First fork: can you afford full tree traversals? If yes (toy games, aggressively shrunken Drawmaha): vanilla CFR to learn and debug, then CFR+ (regret-matching+ clipping, alternating updates, linear averaging — all three together) or, better, DCFR with the recommended (α, β, γ) = (1.5, 0, 2), which beat CFR+ in every game Brown & Sandholm tested. If no (real Drawmaha, whose deal space alone is C(52,5)² ≈ 6.7 × 10¹² before draws): MCCFR. Second fork, within MCCFR: external sampling (sample chance + opponent, traverse all own actions) is the practitioner default — low variance, simple weights; outcome sampling (one playout per iteration) is cheapest but much noisier, mainly for online settings. Third fork: if sampled convergence is noise-limited, add VR-MCCFR baselines (order-of-magnitude speedup, variance down 1000×, and the only known way to combine CFR+-style updates with sampling). Crucial interaction: CFR+/DCFR's negative-regret handling mixes badly with sampling noise, so with MCCFR use LINEAR weighting (LCFR) — the variant Brown & Sandholm explicitly flag as sampling-compatible and the one Deep CFR's data-weighting inherits. Convergence-rate summary: everything tabular here is O(1/√T) exploitability in theory; CFR+ is proven to share CFR's asymptotic rate (IJCAI 2015 'tracking regret' analysis) but empirically behaves like ~1/T; DCFR's bound is worse than CFR's by a constant factor yet it converges fastest in practice — the field trades bounds for empirics with open eyes.

WHAT PRACTITIONERS ACTUALLY USE: full-traversal solvers (PioSolver-class, Cepheus) run CFR+/discounted variants — Cepheus solved 10¹⁴-infoset limit hold'em with CFR+ on 4,800 CPUs over 68 days, which is both the tabular approach's triumph and its ceiling, and the clearest argument that full Drawmaha needs Deep CFR. Research code for large games defaults to external-sampling MCCFR with linear weighting.

PITFALLS: querying the current instead of average strategy; implementing regret-matching+ without alternating updates and weighted averaging (can be slower than vanilla); trusting noisy sampled exploitability — build an exact best-response calculator during the Kuhn/Leduc phase and keep it for every later phase; forgetting that Kuhn has a known closed-form equilibrium family to validate against. Reassurance for this project: heads-up split-pot Drawmaha is still two-player zero-sum (payoffs sum to the pot), so every guarantee in this lane applies unchanged; the draw round is just extra chance/action nodes — CFR is indifferent to their semantics, though the 32-subset discard action space fattens infosets.

VERIFICATION CAVEATS. All ten sources were confirmed by fetch or API (titles/authors/years/venues, and abstracts for the load-bearing ones). Semantic Scholar rate-limited heavily, so citation counts are included only where retrieved (Hart & Mas-Colell: 1,398; Bowling 2015: 559 — S2 undercounts vs Google Scholar). Stated from background knowledge and NOT independently verified this session: the exact CFR regret bound constants (Δ|Iᵢ|√(A)·√T), Cepheus's final exploitability figure (widely reported as <1 mbb/g — verify before citing a number), and the claim that Deep CFR uses linear weighting (belongs to the deep-cfr lane's verification). The DCFR(1.5, 0, 2) recommendation was confirmed via a 2025 AAAI paper quoting the original, not the original's own text. I found no published CFR work on Drawmaha or any split-pot draw variant — the project appears genuinely first there.

## Papers

### A Simple Adaptive Procedure Leading to Correlated Equilibrium (2000)
_Sergiu Hart, Andreu Mas-Colell · Econometrica · 1398 citations_
<https://doi.org/10.1111/1468-0262.00153>

**Summary:** Introduces regret matching, the tiny learning rule at the heart of every CFR variant. Each player tracks, for each action, how much better off they would have been had they always played that action instead of what they actually played (the 'regret'), and then plays each action with probability proportional to its positive regret. The paper proves that when all players follow this procedure, the empirical distribution of play converges to the set of correlated equilibria.

**Key Ideas:** Regret matching: strategy at time t+1 = normalize(positive parts of cumulative regrets). No gradients, no learning rate, no optimizer — just accumulate regrets and normalize. Its no-regret property (average regret shrinks toward zero) is what all of CFR's guarantees are built on.

**Tradeoffs:** Simpler and hyperparameter-free compared to alternatives like Hedge/multiplicative weights (which need a learning-rate schedule), at the cost of a slightly weaker theoretical constant. In two-player zero-sum games the no-regret property is enough to reach Nash, which is why poker work standardized on it.

**Relevance:** This is the ~10-line function you will write first. In the Drawmaha solver, every information set (a player's 5 cards + betting/draw history) stores a cumulative-regret vector per action, and regret matching turns it into a strategy. Understanding this one rule makes all CFR variants readable: they differ almost only in how regrets are accumulated and weighted.

### Regret Minimization in Games with Incomplete Information (2007)
_Martin Zinkevich, Michael Johanson, Michael Bowling, Carmelo Piccione · NIPS 2007_
<https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information>

**Summary:** The original CFR paper. The problem: regret matching works on one-shot matrix games, but poker is a tree of sequential decisions with hidden information; converting the tree to a matrix of full strategies is astronomically large. CFR's trick is to define 'counterfactual regret' locally at each information set and prove that the sum of these local regrets upper-bounds overall regret — so running independent little regret-matchers at every decision point minimizes global regret. Verified from the abstract: this let them solve limit hold'em abstractions with 10^12 states, two orders of magnitude larger than previous methods (LP solvers).

**Key Ideas:** Counterfactual value = expected value at an infoset weighted by the probability that OPPONENTS and chance reach it (your own reaching probability is counterfactually set to 1). Minimizing counterfactual regret at every infoset minimizes overall regret; by the folk theorem (average regret ε for both players in a zero-sum game ⇒ the pair of AVERAGE strategies is a 2ε-Nash equilibrium), the time-averaged strategy converges to Nash. The current strategy oscillates and does not converge — only the average does.

**Tradeoffs:** Memory scales with information sets × actions (two tables: cumulative regret and cumulative strategy), not with full strategy space — that is the decisive win over LP methods. Cost: each iteration is a full tree traversal, and convergence is O(1/√T) in exploitability, which is slow in wall-clock terms; every later variant attacks one of these two costs.

**Relevance:** This is the algorithm to implement and validate on Kuhn and Leduc. It also dictates the Drawmaha architecture: the dashboard must query the AVERAGE policy (or a network trained to imitate it), never the final iterate. Heads-up Drawmaha with a split pot is still two-player zero-sum (the players' payoffs sum to the pot), so the folk-theorem guarantee applies unchanged.

### An Introduction to Counterfactual Regret Minimization (2013)
_Todd W. Neller, Marc Lanctot · AAAI Model AI Assignments (tutorial)_
<https://modelai.gettysburg.edu/2013/cfr/cfr.pdf>

**Summary:** The standard pedagogical on-ramp: builds regret matching on rock-paper-scissors, then walks through a complete, worked CFR implementation that solves Kuhn poker (3 cards, one betting round), with runnable code. Widely used as the first thing practitioners implement before touching a real game.

**Key Ideas:** Concrete infoset keying (card + history string), the two-table layout (regret sum, strategy sum), recursive tree walk returning counterfactual values, and reading off the average strategy at the end — exactly the skeleton a real solver scales up.

**Tradeoffs:** A tutorial, not a research contribution: covers vanilla CFR and chance sampling only; no CFR+/discounting/variance reduction. Its Java-style pseudocode ports to Python/NumPy in an afternoon.

**Relevance:** Directly matches the planned first milestone (tabular CFR on Kuhn/Leduc). Kuhn poker's exact analytical equilibrium (the paper gives it, parameterized by alpha) provides the ground-truth check the project needs before trusting the implementation on Drawmaha; an exploitability (best-response) calculator built at this stage remains the validation tool for every later phase.

### Monte Carlo Sampling for Regret Minimization in Extensive Games (2009)
_Marc Lanctot, Kevin Waugh, Martin Zinkevich, Michael Bowling · NIPS 2009_
<https://papers.nips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games>

**Summary:** Introduces MCCFR: instead of traversing the entire game tree every iteration, sample parts of it and update regrets with importance-weighted estimates that equal the full-CFR update in expectation. Verified from the abstract: defines outcome sampling (sample a single playout) and external sampling (sample chance and opponent actions, but traverse all of the updating player's own actions), proves both keep overall regret bounded with high probability, tightens the original CFR bound, and shows the far cheaper iterations give dramatically faster wall-clock convergence in several games.

**Key Ideas:** Unbiased sampled counterfactual values via importance weighting over sampled 'blocks' of terminal histories. External sampling is the sweet spot: no importance-weight explosion on your own actions, iterations cost roughly one deal + opponent playout, and regret updates stay low-variance.

**Tradeoffs:** More iterations needed than vanilla CFR, but each is orders of magnitude cheaper — a large net win when the tree is dominated by chance branching (card deals). Outcome sampling is the cheapest per iteration and touches only one path (useful for online/bandit-style settings) but has much higher variance; external sampling is the practitioner default. Sampling noise also means exploitability estimates fluctuate, so evaluation needs separate best-response computation.

**Relevance:** The pivotal algorithm for this project. Drawmaha's chance nodes are brutal — C(52,5)^2 ≈ 6.75 trillion deal combinations before the draw round multiplies them further — so full traversals are impossible beyond toy shrinkage. External-sampling MCCFR is the standard middle step (the planned 'shrunken variant' phase), and it is also the traversal scheme Deep CFR uses internally, so this code gets reused in the final phase.

### Solving Large Imperfect Information Games Using CFR+ (2014)
_Oskari Tammelin · arXiv:1407.5042_
<https://arxiv.org/abs/1407.5042>

**Summary:** A short note introducing CFR+, the variant that made solving full heads-up limit hold'em feasible. Verified from the abstract: it typically outperforms the previously known algorithms (public chance sampling CFR, Pure CFR) by an order of magnitude or more in computation time while potentially using less memory.

**Key Ideas:** Three changes to vanilla CFR (details in the companion IJCAI 2015 paper): (1) regret-matching+ — clip cumulative regrets at zero each iteration so an action buried under negative regret can 'come back' immediately when it turns good; (2) alternating updates — update one player per traversal instead of both simultaneously; (3) weight later iterations linearly when averaging the strategy, so early garbage iterations fade from the answer.

**Tradeoffs:** Empirically converges far faster than vanilla CFR on full traversals — often so fast the CURRENT strategy is usable — but its clipping interacts badly with the noisy estimates of sampled (MCCFR) variants, so plain CFR+ is a full-traversal algorithm. All three tricks must be applied together as specified; cherry-picking (e.g., regret-matching+ with simultaneous updates and uniform averaging) can underperform vanilla CFR.

**Relevance:** If a shrunken Drawmaha variant is small enough for full traversals, CFR+ is the right tabular algorithm — same memory, much faster. For the sampled phases, its lesson survives in modified form via DCFR/Linear CFR (which are sampling-compatible) rather than direct use.

### Heads-up limit hold'em poker is solved (2015)
_Michael Bowling, Neil Burch, Michael Johanson, Oskari Tammelin · Science · 559 citations_
<https://www.science.org/doi/10.1126/science.1259433>

**Summary:** The landmark result: Cepheus essentially weakly solved heads-up limit Texas hold'em using CFR+ — the first competitively-played imperfect-information game to be solved. 'Essentially solved' means the computed strategy's exploitability is so low that a lifetime of human play could not statistically distinguish it from an exact equilibrium.

**Key Ideas:** Proof-of-concept that tabular CFR-family methods scale to ~10^14 information sets with enough engineering: CFR+ for fast convergence, massive parallelism over subgames, and compression of the regret/strategy tables. Also famous for a strategic finding: the solution confirms the dealer holds a positive edge and (for example) almost never folds preflop.

**Tradeoffs:** Demonstrates the ceiling of the pure tabular approach — per the companion IJCAI paper this took 4,800 CPUs for 68 days. It also shows tabular solving is only viable when the unabstracted game fits (compressed) in distributed storage; anything larger forces abstraction or function approximation, which is exactly the fork the Drawmaha project faces.

**Relevance:** Calibration for ambition: limit hold'em has ~10^14 infosets and needed a cluster. Full Drawmaha (2.6M private hands per player, a board, and a 0-5 card discard decision with 32 discard subsets) is far beyond a laptop's tabular reach — this paper is the clearest argument for why the project's endgame must be Deep CFR rather than a bigger table.

### Solving Heads-Up Limit Texas Hold'em (2015)
_Oskari Tammelin, Neil Burch, Michael Johanson, Michael Bowling · IJCAI 2015_
<https://www.ijcai.org/Abstract/15/097>

**Summary:** The technical companion to the Science paper: proves CFR+ and regret-matching+ are sound, and documents the engineering that produced Cepheus. Verified from the abstract: proves CFR+ converges with the same asymptotic rate guarantee as CFR (O(1/√T) worst case) even though it is drastically faster in practice, gives a 'tracking regret' bound for regret-matching+ that hints at why (it adapts faster when the best action changes over time), and states the 10^14-infoset scale, 4,800 CPUs, and 68 days.

**Key Ideas:** Theory catching up to practice: CFR+'s empirical convergence often looks like O(1/T) or better, but the provable guarantee is unchanged from CFR — the improvement is a constants-and-adaptivity story, not an asymptotic one. Also the definitive spec of regret-matching+ (clip-at-zero accumulation) and linear averaging.

**Tradeoffs:** Same as CFR+: spectacular on full traversals, not directly compatible with Monte Carlo sampling. The tracking-regret perspective explains a practical observation: CFR+ recovers quickly when the strategy landscape shifts mid-run, where vanilla CFR must slowly pay down old accumulated regret.

**Relevance:** The paper to read (rather than the 2-page arXiv note) for implementing CFR+ correctly, since getting the three components exactly right is a known pitfall. Its convergence discussion also sets correct expectations for the Drawmaha toy-game phase: measure exploitability against iterations and expect faster-than-√T empirical curves from CFR+.

### Solving Imperfect-Information Games via Discounted Regret Minimization (2019)
_Noam Brown, Tuomas Sandholm · AAAI 2019 (arXiv:1809.04040)_
<https://arxiv.org/abs/1809.04040>

**Summary:** Generalizes CFR+'s tricks into a family of discounting schemes and finds better ones. Verified from the abstract: variants discount early-iteration regrets (differently for positive vs negative regrets), reweight iterations when forming the output average strategy, and use non-standard regret minimizers; one variant (DCFR) outperforms CFR+ in every game tested including large realistic ones — the first algorithm to beat CFR+ — and unlike CFR+, several variants are compatible with pruning and one (Linear CFR) with sampling.

**Key Ideas:** Linear CFR (LCFR): weight iteration t's regret and strategy contributions by t — early iterations, made when both players were playing garbage, get washed out. DCFR(α, β, γ) generalizes: positive regrets scaled by t^α/(t^α+1), negative by t^β/(t^β+1), average-strategy contributions by (t/(t+1))^γ; the recommended setting is (α, β, γ) = (1.5, 0, 2). Aggressively discounting negative regret (β = 0 or -∞-like behavior) is what lets DCFR escape bad early commitments faster than CFR+'s hard clipping.

**Tradeoffs:** DCFR's theoretical bound is worse than CFR's by a constant factor, but it converges much faster in practice — the field accepts this trade. DCFR beats CFR+ on full traversals but its negative-regret discounting mixes poorly with sampling variance; LCFR is the sampling-safe member. Adds hyperparameters where CFR+ had none, though (1.5, 0, 2) is a widely reused default.

**Relevance:** The current tabular state of the art and the discounting scheme to adopt: DCFR(1.5, 0, 2) for full-traversal phases, linear weighting for the MCCFR phase. Directly load-bearing for the endgame too — Deep CFR (same authors' follow-up) uses linear (LCFR-style) weighting when sampling training data for its networks, so understanding this paper is a prerequisite for the Deep CFR lane.

### Variance Reduction in Monte Carlo Counterfactual Regret Minimization (VR-MCCFR) for Extensive Form Games using Baselines (2019)
_Martin Schmid, Neil Burch, Marc Lanctot, Matej Moravcik, Rudolf Kadlec, Michael Bowling · AAAI 2019 (arXiv:1809.03057)_
<https://arxiv.org/abs/1809.03057>

**Summary:** Imports the baseline idea from policy-gradient RL into MCCFR. Verified from the abstract: per-iteration value estimates are reformulated around state-action baselines so estimates bootstrap off other estimates within an episode while staying unbiased; with a perfect baseline the variance of value estimates can be reduced to zero. Reported results: an order-of-magnitude speedup over plain MCCFR with empirical variance down three orders of magnitude, and — by taming the noise — the first working combination of sampling with CFR+, for two orders of magnitude overall speedup.

**Key Ideas:** Control variates for counterfactual values: subtract a learned baseline (e.g., running average of an infoset-action's value) from the sampled return and add it back in expectation, exactly like advantage baselines in actor-critic — a bridge concept for a reader coming from RL.

**Tradeoffs:** Extra memory (baseline table) and implementation complexity on top of MCCFR; payoff is largest in games with long episodes and high-variance payoffs. Not necessary for correctness — plain external sampling already converges — so it is an optimization to add only if MCCFR convergence is noise-limited.

**Relevance:** Split-pot Drawmaha payoffs are relatively bounded (halved pots reduce payoff spread), but the deal/draw chance variance is enormous, so baselines are a plausible mid-project upgrade if the shrunken-variant MCCFR converges too slowly. The paper is also the cleanest conceptual link between the CFR world and the actor-critic vocabulary the project author already has.

### How Solvers Work (GTO Wizard blog) (2023)
_GTO Wizard team · GTO Wizard blog (practitioner secondary source)_
<https://blog.gtowizard.com/how-solvers-work/>

**Summary:** A practitioner-side account of what commercial poker solvers actually run: conventional solvers like PioSolver use CFR-family approximate equilibrium-finding with discounting-style improvements, and modern services layer neural networks on top of CFR to accelerate solving (as in GTO Wizard's own AI and their PLO work).

**Key Ideas:** Confirms the deployment reality: nobody ships vanilla CFR. Commercial practice = CFR+/discounted variants for tree solving, plus abstraction/NN acceleration for big games; exploitability targets (e.g., a fraction of the pot) define 'solved enough' rather than exact Nash.

**Tradeoffs:** A blog, not a paper: no proofs, limited algorithmic detail, and commercial incentive to oversell the NN layer. Useful for the 'which variant do practitioners use' question, not for theory.

**Relevance:** Validates the Drawmaha project's exact architecture (CFR core, discounting, then a neural layer, then a query dashboard) as the same shape the industry converged on, and suggests framing the dashboard's quality bar in exploitability-per-pot terms the way commercial solvers do.

