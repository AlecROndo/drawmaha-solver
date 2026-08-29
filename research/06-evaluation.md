# Research lane: evaluation

_Generated 2026-08-28 by a 9-agent literature sweep (Semantic Scholar / arXiv / web). Source material for the ML-survey paper._

## Lane narrative

The evaluation problem in one sentence: a CFR-style solver's training loss tells you nothing, because self-play produces no ground-truth labels — the only principled question is "how much would a perfect adversary win against my strategy?", and everything in this lane is a way of answering that at different scales and costs. The exact answer is exploitability (per-player) or NashConv (summed over players; 2× exploitability in heads-up zero-sum — watch for the factor-of-2 across codebases). Units are milli-big-blinds per hand (mbb/g); bb/100 = mbb/g ÷ 10; calibration points: always-fold loses 750 mbb/g, pros consider ~50 mbb/g a big edge, Cepheus's <1 mbb/g defines "essentially solved."

The lane's history is a ladder of retreats from exactness as games grew. Era 1: exact best response by tree walk — trivial for Kuhn/Leduc (OpenSpiel does it in one call), and Johanson et al. 2011 pushed it to full limit hold'em (10^14 states, 76 CPU-days) via public-tree traversal with all private hands vectorized. Their punchline reshaped the field: bots nearly optimal within their abstractions were exploitable for hundreds of mbb/g in the real game. Era 2: for games where even that fails, Lisy & Bowling's Local Best Response plays hands against your strategy while tracking the opponent's exact range, greedily maximizing EV with a call-down heuristic; its winnings are a certified lower bound on exploitability. It exposed top no-limit bots (within 24 mbb/g of each other head-to-head) as >3,180 mbb/g exploitable. Era 3: Timbers et al.'s ISMCTS-BR replaces LBR's poker heuristics with a trained RL best-responder — game-agnostic, tighter, much costlier.

Parallel track: head-to-head evaluation, which is unavoidable at full scale but only ever relative. Its enemy is variance; the tools are duplicate dealing (same decks, seats swapped — the ACPC standard, free when you control the RNG) and AIVAT (control variates from a value function + your own known strategy; unbiased regardless of value-function quality; ~10× fewer hands for significance). DeepStack is the exemplar of the full stack: 44,852 hands vs 33 pros, AIVAT-corrected 486 mbb/g, plus an LBR probe — average-case AND worst-case, because a big head-to-head win is provably compatible with huge exploitability.

The options for the Drawmaha builder, by stage: (1) Kuhn/Leduc — don't write evaluation code blind; reproduce OpenSpiel's exploitability curves as unit tests of your CFR and your best-response walker. (2) Shrunken Drawmaha — this is your Flop Hold'em: size it (smaller deck, capped betting) so exact tree-walk exploitability stays feasible, then use exploitability-vs-iterations curves as the arena where MCCFR vs Deep CFR vs architecture choices get compared honestly. Note the 5-card hand space (2.6M hands) makes the vectorized best-response pass expensive — shrink the deck, not just stakes. (3) Full game — exact numbers are gone; the rigorous package is: (a) a checkpoint ladder and fixed baselines (random, always-call, the validated shrunken-game recipe scaled up) played under duplicate dealing with bootstrap CIs in mbb/hand; (b) AIVAT on top once you have any value heuristic (a Monte Carlo pot-equity rollout suffices; quality only affects variance removed, never bias); (c) a worst-case probe — either LBR adapted to the draw round (its range update is exact but 2000× wider than hold'em's; the call-down heuristic needs a stand-pat analogue) or, cleaner, a learned exploiter: freeze the average-policy network and train a PPO/DQN best responder against it, which is a single-agent RL problem and far easier than solving the game. Report the probe's winnings as a lower bound, and treat a probe that loses as "found nothing," never "unexploitable."

Pitfalls, all documented above: abstract-game exploitability ≠ real-game exploitability; head-to-head rankings and exploitability rankings can disagree (Act1/Slumbot); LBR bounds can be vacuous (it lost to a full-cards bot whose true exploitability was ≥90 mbb/g); unpaired evaluation of 10–20 mbb/hand differences never reaches significance.

Not found / uncertain: no published evaluation methodology for any split-pot or draw poker variant — pot-half accounting and discard-information handling in LBR/AIVAT will be genuinely novel choices. Unverified background claims, labeled: that NashConv is coined in Lanctot et al. 2017 (paper confirmed; coinage not visible in abstract); that SD-CFR/DREAM report Leduc and FHP exploitability as their headline metrics (DREAM's abstract confirms benchmark-game SOTA claims but not the specific games/metric). Semantic Scholar rate-limited, so citation counts are omitted rather than guessed; every listed paper's title/year/venue was confirmed by direct fetch or search of primary pages.

## Papers

### Accelerating Best Response Calculation in Large Extensive Games (2011)
_Michael Johanson, Kevin Waugh, Michael Bowling, Martin Zinkevich · IJCAI 2011_
<https://www.ijcai.org/Proceedings/11/Papers/054.pdf>

**Summary:** The paper that made exact exploitability practical for a nontrivial poker game. A best response — the perfect counter-strategy — can in principle be computed by one walk of the game tree, but a naive walk of heads-up limit hold'em (~10^14 decision points) was hopeless. The authors restructure the walk around public states, evaluate all of a player's possible private hands simultaneously as a vector, and parallelize, cutting the cost to 76 CPU-days (one wall-clock day on a cluster). This let the field measure, for the first time, how far the top abstraction-based hold'em bots really were from equilibrium.

**Key Ideas:** Exploitability = how much a perfect counter-strategy wins against your fixed strategy (0 at Nash). Compute it by a single expectimax-style tree walk where the best responder maximizes given the evaluated strategy's action probabilities; accelerate by traversing the public tree once while carrying a vector of values for every private hand, so card-isomorphic work is shared.

**Tradeoffs:** Gives the exact, gold-standard number — no caveats about bounds. But cost scales with the real game tree: feasible up to roughly 10^13–10^14 states with serious engineering, out of reach beyond that. Also revealed a trap: agents nearly optimal inside their abstraction were still exploitable for hundreds of mbb/g in the real game, so abstract-game exploitability is not a substitute.

**Relevance:** This is the technique behind the tree-walk exploitability the Drawmaha plan needs for stages 1–2. Kuhn/Leduc are trivial for it; the shrunken Drawmaha variant should be sized so this computation stays feasible (the vectorized-over-private-hands trick matters even more with 5-card hands, since Drawmaha's private-hand vector is ~2.6M entries instead of hold'em's 1,326 — a strong argument for shrinking the deck, not just the stack, in the toy variant).

### Equilibrium Approximation Quality of Current No-Limit Poker Bots (Local Best Response) (2017)
_Viliam Lisy, Michael Bowling · AAAI-17 Workshop on Computer Poker and Imperfect Information Games_
<https://arxiv.org/abs/1612.07547>

**Summary:** Introduces Local Best Response (LBR), a cheap way to lower-bound exploitability when the real game is too big for an exact best response. LBR plays actual hands against the evaluated strategy while tracking, via Bayes' rule on the opponent's known strategy, the exact probability distribution over the opponent's private hands (their 'range'); at each decision it greedily picks the action maximizing expected value under the assumption everyone checks/calls to showdown afterward. Because LBR is itself a legal strategy, whatever it wins is a certified lower bound on exploitability. The shock result: top ACPC no-limit bots, separated by <24 mbb/g head-to-head, were all exploitable for >3,180 mbb/g — more than four times worse than folding every hand (750 mbb/g).

**Key Ideas:** Exact range tracking (no abstraction needed for beliefs) + one-step greedy lookahead with a call-down heuristic = a fast exploiter whose winnings certify a lower bound. Restricting LBR's action menu (fold/call; fold/call/pot/all-in; 56 bet sizes) and forcing it to check early rounds are knobs — the paper shows waiting until later rounds exploits far more.

**Tradeoffs:** Orders of magnitude cheaper than a true best response and abstraction-free. But it is only a lower bound and can be vacuous: against their no-card-abstraction bot LBR actually LOST (−424 to −819 mbb/g) even though the true exploitability within that betting structure was ≥90 mbb/g. LBR rankings also need not match head-to-head rankings (Act1 was least LBR-exploitable yet lost to Slumbot). Requires the ability to query the strategy for every private hand at a public state — cheap for a network, and each hand costs ~(n·|H|+1)× normal play.

**Relevance:** The realistic exploitability instrument for full-scale Drawmaha, where exact best response is impossible. The average-policy network can be queried for all opponent hands, exactly what LBR needs — but note |H| = 2,598,960 in Drawmaha vs 1,326 in hold'em, so range updates and win-probability rollouts are ~2000× costlier per decision; expect to need sampling or hand bucketing inside the LBR evaluator, and a discard-aware heuristic (a 'stand pat to showdown' assumption replacing call-down). A negative LBR result will mean 'this probe found nothing', not 'the bot is unexploitable' — report it that way.

### AIVAT: A New Variance Reduction Technique for Agent Evaluation in Imperfect Information Games (2018)
_Neil Burch, Martin Schmid, Matej Moravcik, Michael Bowling · AAAI 2018 (earlier version: AAAI-17 Computer Poker Workshop)_
<https://arxiv.org/abs/1612.06915>

**Summary:** Head-to-head poker results are drowned in luck: who got dealt what and which side of a coin-flip all-in landed. AIVAT is a provably unbiased estimator that subtracts the luck out. Using a heuristic value function over game states plus the known strategy of the agent(s) you control, it builds control-variate correction terms for both chance events (card deals) and the known players' sampled actions, so what remains is much closer to pure skill difference. Reported effect: roughly 10× fewer hands needed for the same statistical significance in no-limit hold'em.

**Key Ideas:** Control variates in a game tree: for each chance node and each known-strategy decision node, add (expected value under the value function before the event) minus (value after the event) — zero in expectation, but it cancels the variance the event injected. Works with imperfect information and needs only unilateral knowledge (your own bot's strategy), so it is valid against humans or closed-source opponents.

**Tradeoffs:** Unbiased regardless of how bad the value function is — quality only affects how much variance is removed. Needs an explicit value heuristic (in poker, expected-showdown-value style functions) and engineering to log the full decision context. Compare: duplicate dealing is simpler and needs no model but removes less variance and requires paired matches; raw bb/100 needs nothing but millions of hands.

**Relevance:** Essential for the Drawmaha dashboard-era evaluation, because there is no oracle to compare against — checkpoint-vs-checkpoint and bot-vs-human matches are the evidence, and Drawmaha's split pot plus a draw round adds even more chance variance than hold'em. A usable value function can be as crude as expected pot equity from a Monte Carlo rollout, or the Deep CFR value network itself. Without AIVAT (or at least duplicate dealing), differences of 10–50 mbb/hand will not reach significance in any reasonable number of hands.

### DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker (2017)
_Matej Moravcik, Martin Schmid, Neil Burch, Viliam Lisy, et al. (incl. Michael Bowling) · Science 356(6337)_
<https://arxiv.org/abs/1701.01724>

**Summary:** Beyond its algorithmic contribution, DeepStack is the model of a rigorous evaluation of a solver for a game with no computable exact exploitability. It ran a pre-registered-style human study — 33 professionals recruited through the International Federation of Poker, 44,852 hands over 4 weeks — and reported the AIVAT-corrected win rate: 486 mbb/g, statistically significant, against a backdrop where pros consider ~50 mbb/g a sizable edge. It also subjected itself to the adversarial probe of the day (LBR), reporting that its strategies were harder to exploit than prior abstraction-based approaches.

**Key Ideas:** Two-pronged evaluation: (1) head-to-head vs strong humans with AIVAT variance reduction making a month of play statistically decisive, (2) an exploitability lower-bound probe (LBR) as the worst-case check. Neither alone suffices: head-to-head measures average-case vs one population; LBR measures worst-case.

**Tradeoffs:** The human study is expensive and unrepeatable for a hobby project; but the structure — variance-reduced head-to-head PLUS an adversarial probe — transfers at any budget. Note the win rate is vs humans, not vs Nash: a big head-to-head win is compatible with high exploitability (the LBR paper proved this for ACPC bots), which is exactly why DeepStack ran both.

**Relevance:** The template for the Drawmaha paper/dashboard's headline evaluation: report an AIVAT- or duplicate-corrected win rate against defined opponents (earlier checkpoints, MCCFR blueprint, rule-based baselines, and yourself/friends as human opponents), alongside a separate worst-case probe. Since no Drawmaha baseline exists, the checkpoint ladder plays the role DeepStack's human pool played.

### Heads-up Limit Hold'em Poker is Solved (Cepheus) (2015)
_Michael Bowling, Neil Burch, Michael Johanson, Oskari Tammelin · Science 347(6218)_
<https://bowlingmh.github.io/papers/15science.pdf>

**Summary:** The paper that defined what 'solved' means in exploitability units. Using CFR+, Cepheus drove the exploitability of full heads-up limit hold'em (3.16 × 10^17 states) below 0.986 mbb/g after 900 CPU-years — 'essentially solved', meaning a human playing a full lifetime could not statistically distinguish it from an exact Nash equilibrium. The 1 mbb/g line has since served as the field's reference point for 'close enough to optimal'.

**Key Ideas:** Ties an abstract metric (exploitability) to an operational meaning (statistical indistinguishability from optimal over a lifetime of play). Also demonstrates the workflow of tracking exploitability as the convergence curve of a CFR run — the number IS the progress bar.

**Tradeoffs:** Exact-exploitability-as-progress-bar only works when the best-response computation is affordable relative to how often you want the number; for Cepheus it was. The mbb/g scale it popularized (1000 mbb = 1 big blind; bb/100 = mbb/g ÷ 10) is now the units everyone reports in.

**Relevance:** Gives the Drawmaha project its units and its finish lines: report everything in milli-big-blinds per hand; on the toy and shrunken games, plot exploitability vs CFR iterations (the canonical convergence figure) and pick an explicit target (e.g. <1 mbb/g on the shrunken game). For a split-pot game, define the unit carefully once (total chips won across both halves of the pot, per hand, in big blinds) and use it everywhere.

### Deep Counterfactual Regret Minimization (2019)
_Noam Brown, Adam Lerer, Sam Gross, Tuomas Sandholm · ICML 2019_
<https://arxiv.org/abs/1811.00164>

**Summary:** The paper the Drawmaha plan's final stage is built on — and the clearest example of how neural-CFR work actually evaluates itself. Deep CFR was measured two ways: exact exploitability in mbb/g on Flop Hold'em Poker (FHP: >10^12 nodes but ~10^9 infosets — big enough to be interesting, small enough that a best response is still computable), where Deep CFR reached ~37 mbb/g vs NFSP's ~47 while using 2–3 orders of magnitude fewer samples than a lossless abstraction; and head-to-head in full heads-up limit hold'em (10^14 infosets, exact exploitability infeasible), where it lost only 11±2 mbb/g to a strong abstraction baseline and beat NFSP by 43±2 mbb/g. The network was small: 98,948 parameters, 7 layers, trained with Adam at lr 0.001.

**Key Ideas:** The two-tier evaluation pattern: pick a mid-size game where exact exploitability is computable and show your neural method's convergence there against baselines; then move to the full game and fall back to head-to-head matches, since exploitability can no longer be computed. Exploitability numbers carry the scientific weight; head-to-head extends the claim to full scale.

**Tradeoffs:** The FHP results are the honest part (absolute, worst-case); the HULH head-to-heads are only relative. Neither the Deep CFR paper nor its successors computed real-game exploitability at full scale — a gap the field papers over, and one reviewers of any Drawmaha writeup will recognize as standard.

**Relevance:** Directly blesses the planned pipeline: the shrunken Drawmaha variant is the project's FHP — size it so a tree-walk best response is computable, and use it to compare MCCFR vs Deep CFR vs network/hyperparameter ablations on exact exploitability curves. Then evaluate full-game Deep CFR by head-to-head (vs the shrunken-game-validated recipe scaled up, vs checkpoints, vs simple agents) plus an LBR-style probe.

### OpenSpiel: A Framework for Reinforcement Learning in Games (2019)
_Marc Lanctot, Edward Lockhart, et al. (DeepMind) · arXiv:1908.09453_
<https://arxiv.org/abs/1908.09453>

**Summary:** The standard open-source toolkit for exactly this project's validation stage. It ships Kuhn poker, Leduc poker, and dozens of other games behind one extensive-form-game API, plus thoroughly-tested implementations of best response / exploitability (reported as NashConv and exploitability), CFR, CFR-BR, external- and outcome-sampling MCCFR, Deep CFR, and NFSP. Its exploitability functions are the de facto reference implementation researchers verify their own code against.

**Key Ideas:** python: `open_spiel.python.algorithms.exploitability.exploitability(game, policy)` walks the tree computing each player's best-response value; NashConv sums the best-response improvements over all players (in a 2-player zero-sum game, exploitability is NashConv/2). Having reference CFR + reference exploitability in one place means any discrepancy in your own solver is immediately localizable.

**Tradeoffs:** The exact tooling only scales to games it can enumerate (fine for Kuhn/Leduc/small customs, not full Drawmaha). Implementing Drawmaha inside OpenSpiel's game API is real work (draw/discard actions, split pots — OpenSpiel does support the needed machinery) but buys free, trusted evaluation and free baseline algorithms; the alternative, standalone code cross-checked against OpenSpiel on Kuhn/Leduc only, is faster to start and what many projects do.

**Relevance:** Stage-1 insurance for the Drawmaha project: before trusting your own tabular CFR, reproduce OpenSpiel's exploitability numbers on Kuhn and Leduc (e.g. verify your CFR's exploitability curve matches theirs iteration-for-iteration, and that your best-response routine returns identical values on their policies). Any custom exploitability code for the shrunken Drawmaha should be unit-tested against OpenSpiel on a game small enough to run in both.

### Approximate Exploitability: Learning a Best Response in Large Games (ISMCTS-BR) (2022)
_Finbarr Timbers, Nolan Bard, Edward Lockhart, Marc Lanctot, Martin Schmid, Neil Burch, Julian Schrittwieser, Thomas Hubert, Michael Bowling · IJCAI 2022 (arXiv 2004.09677, first posted 2020)_
<https://arxiv.org/abs/2004.09677>

**Summary:** The modern, game-agnostic answer to 'the game is too big for exact best response and LBR is poker-specific'. ISMCTS-BR trains a deep RL agent, guided by information-set MCTS, to learn a best response against a fixed evaluated agent; its winnings estimate a lower bound on exploitability, like LBR but without hand-crafted poker heuristics. Demonstrated across several two-player zero-sum games, including probing AlphaZero-based agents.

**Key Ideas:** Treat best-response computation itself as an RL problem: the evaluated agent is frozen (part of the environment), and a searcher-learner is trained to beat it. The stronger the learned exploiter, the tighter the lower bound.

**Tradeoffs:** More general and often stronger than LBR's one-step greedy probe, but far more expensive (a full RL training run per evaluation) and still only a lower bound with no tightness guarantee — a weak exploiter proves nothing. Compared to exact best response: applicable at any scale, but never certifies closeness to Nash.

**Relevance:** The fallback if adapting LBR's poker heuristics to Drawmaha's draw round proves awkward: train a best-response network against the frozen average-policy network (this is much easier than solving the game — the opponent is fixed, so it is a single-agent RL problem). Even a simple DQN/PPO exploiter trained against the final bot is a publishable-quality worst-case probe for a game with no baselines.

### The Annual Computer Poker Competition (2013)
_Nolan Bard, John Hawkin, Jonathan Rubin, Martin Zinkevich · AI Magazine 34(2)_
<https://ojs.aaai.org/index.php/aimagazine/article/view/2474>

**Summary:** Documents the head-to-head protocol that the poker research community standardized on: duplicate matches, where the same sequence of dealt cards is played twice with seats reversed, so if one bot gets lucky cards in match A, its opponent gets the same lucky cards in match B. Combined with common card seeds across pairings and bootstrap confidence intervals, this was the ACPC's answer to poker's brutal variance before AIVAT existed.

**Key Ideas:** Duplicate dealing cancels the largest single luck component (card distribution) by symmetry, with zero modeling assumptions — you just need to control the deck. The LBR paper's experiments quantify a related stack: duplicate + 'imaginary observations' shrank their confidence intervals ~20% at fixed sample size; AIVAT later delivered far more (~10× fewer hands).

**Tradeoffs:** Trivial to implement in self-play evaluation (you control the RNG) and assumption-free, but only applicable when both agents can be run on chosen decks — it works bot-vs-bot, not against a human who would remember the hand. Removes between-deal variance but not within-hand action/runout variance, which is why AIVAT still helps on top.

**Relevance:** The minimum-viable protocol for every Drawmaha checkpoint-vs-checkpoint match: deal N pre-generated decks (including the draw-round replacement cards), play each twice with seats swapped, report mean and a bootstrap CI in mbb/hand. This should be built into the match runner from day one — it is ~30 lines of code and multiplies the information per simulated hand.

### A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning (PSRO) (2017)
_Marc Lanctot, Vinicius Zambaldi, Audrunas Gruslys, et al. · NeurIPS 2017_
<https://arxiv.org/abs/1711.00832>

**Summary:** The paper generally credited with the term NashConv, the metric OpenSpiel reports: for each player, compute how much they could gain by unilaterally switching to a best response while everyone else stays fixed, then sum over players. At a Nash equilibrium the sum is zero; for two-player zero-sum games it is simply twice the average exploitability. The paper's broader contribution (policy-space response oracles, and joint-policy correlation for measuring self-play overfitting) frames why worst-case metrics are needed at all: agents trained by self-play can look strong against themselves while being terrible against anything else.

**Key Ideas:** NashConv(π) = Σᵢ [uᵢ(BRᵢ, π₋ᵢ) − uᵢ(π)]: a single scalar 'distance from equilibrium' that generalizes exploitability beyond two-player zero-sum. Self-play win rate against yourself is meaningless as a quality measure — the metric must involve an outside best response.

**Tradeoffs:** NashConv vs exploitability is purely a bookkeeping distinction in HU zero-sum (factor of 2 and sign conventions differ across papers/codebases — a classic source of off-by-2 confusion when comparing numbers); knowing both terms matters mostly for reading the literature and OpenSpiel's output correctly.

**Relevance:** Terminology hygiene for the Drawmaha writeup: state explicitly whether reported numbers are per-player exploitability or summed NashConv, in what units, and match the convention when comparing against OpenSpiel outputs on Kuhn/Leduc. Verification caveat: the paper's existence, venue, and content focus were confirmed, but the claim that NashConv is coined here comes from background knowledge — the abstract does not mention it.

