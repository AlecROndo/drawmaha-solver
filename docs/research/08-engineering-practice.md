# Research lane: engineering-practice

_Generated 2026-08-28 by a 9-agent literature sweep (Semantic Scholar / arXiv / web). Source material for the ML-survey paper._

## Lane narrative

This lane answers: what do you actually build with, and where do projects break? The story since 2013 is a steady lowering of the entry barrier. Neller & Lanctot turned CFR from a paper into a literate-programming exercise; OpenSpiel (2019) and RLCard (2019) industrialized environments and evaluation; Steinberger's PokerRL + Deep-CFR repos (2019) showed a solo engineer can reproduce Deep CFR-class results; PokerKit (2023) finally made weird variants (draws, split pots) a configuration problem instead of an engine-writing problem.

The builder's options split into four decisions. (1) Framework: OpenSpiel gives the most trustworthy tabular CFR code and an exact best-response/exploitability oracle, at the cost of writing your game in C++; RLCard is a friendly Python sandbox with weak equilibrium tooling; PokerRL is the closest architecture to the Drawmaha plan (Deep CFR + SD-CFR + a graded BR/LBR/RL-BR evaluation ladder) but is frozen on PyTorch 0.4.1 — read it, port from it, do not depend on it; PokerKit is the right rules-engine substrate for a nonstandard variant. A sane hybrid: PokerKit-checked custom engine, OpenSpiel as the toy-game referee, own MCCFR/Deep CFR code. (2) Neural variant: Deep CFR (two networks, simple single-net serving) vs SD-CFR (no average-strategy network, provably less approximation error, but serving requires checkpoint mixtures — distill at the end if the dashboard needs one net). (3) Hand evaluator: since Drawmaha hands are exactly 5 cards, Cactus Kev's 7,462-class perfect-hash approach fits natively; PokerHandEvaluator adds Omaha-mode evaluation for the board half of the split pot; keep evaluation in C/C++ or precomputed tables — it sits inside every MCCFR terminal node. (4) Compute ambition: Deep CFR's published recipe (98,948-parameter net, 40M reservoir buffers, 10K traversals/iteration, 4K SGD steps at batch 10K) is single-GPU-class and solved a 10^12-node game; ReBeL needed 90 DGX-1 machines (720 V100s) just for data generation, and its belief-state vector scales with the 2,598,960 possible Drawmaha hands. That compute table is the survey's cleanest argument for stopping at Deep CFR.

Community-documented pitfalls cluster in five places. Reach-probability indexing: regrets must be weighted by opponent-and-chance (counterfactual) reach while average-strategy accumulation is weighted by the player's own reach — swapping these is the classic silent bug, and Neller & Lanctot's explicit code is the diff target. Iteration weighting: the average strategy needs iteration-weighted accumulation, and Linear CFR replay into networks needs the 2/T rescaling Deep CFR specifies. Sampling corrections: outcome-sampling MCCFR requires dividing sampled regrets by the sampling probability; omitting the correction converges to the wrong object — external sampling avoids the division entirely, which is exactly why Deep CFR uses it. Neural-specific: retrain advantage networks from scratch each iteration (Deep CFR's ablation shows fine-tuning converges substantially worse) and use reservoir sampling, not FIFO buffers. Validation discipline: every serious codebase regression-tests against known values — Kuhn's analytic equilibrium (13 infosets), and Leduc's sequence-form-LP first-player game value −0.085606424078 — and even OpenSpiel's own Deep CFR has open user reports of failing on Kuhn (issue #1287), which is the strongest argument that exploitability curves, not loss curves, are the only trustworthy progress signal. For further reading: Max Chiswick's AI Poker Tutorial (aipokertutorial.com) is the best practitioner-level walkthrough from regret matching through abstractions, and Lanctot's EC 2016 computer poker tutorial (mlanctot.info/ecpokertutorial2016) is a good bridge to the theory lane.

Caveats and gaps. Citation counts are omitted: the Semantic Scholar API rate-limited (HTTP 429, no key) and I would not estimate them. PokerKit's abstract claims 'an extensive array of variants' but does not enumerate them; my background knowledge says it covers standard draw variants (badugi, deuce-to-seven), unverified here — and no library I found implements Drawmaha itself, so the split-pot draw engine is genuinely custom work. TwoPlusTwo-evaluator internals (large transition table, card-by-card walk) are background knowledge beyond the verified Senzee perfect-hash link. Deep CFR's paper states no wall-clock/hardware numbers — 'single-GPU-class' is an inference from its hyperparameters, not a quote. Finally, no canonical 'CFR bug catalog' exists anywhere; the pitfalls above are assembled from tutorial code, paper ablations, a student implementation report (Horáček 2021, nlp.fi.muni.cz), and GitHub issues rather than one citable source — an honest gap the survey can note.

## Papers

### An Introduction to Counterfactual Regret Minimization (2013)
_Todd W. Neller, Marc Lanctot · AAAI Model AI Assignments (EAAI)_
<https://modelai.gettysburg.edu/2013/cfr/>

**Summary:** The canonical hands-on CFR tutorial, written in literate-programming style with runnable Java. Section 2 builds regret matching from a rock-paper-scissors worked example, Section 3 introduces CFR by solving Kuhn poker, and Section 4 scaffolds a chance-sampled CFR exercise on 1-die-vs-1-die Dudo. It is the standard first implementation reference the community points beginners to.

**Key Ideas:** Regret matching, then full-tree vanilla CFR with explicit reach-probability bookkeeping, then chance-sampling — each with complete code the reader can diff their own implementation against.

**Tradeoffs:** Pedagogically unmatched for a first tabular CFR, but pre-dates CFR+, MCCFR-at-scale, and all neural methods; the Java code style also hides some indexing conventions (infoset keying by history string) that do not scale.

**Relevance:** This is the document to hand the Drawmaha author on day one: the planned 'tabular CFR validated on Kuhn' phase is literally this tutorial's exercise, and its explicit reach-probability code is the reference to debug against.

### OpenSpiel: A Framework for Reinforcement Learning in Games (2019)
_Marc Lanctot, Edward Lockhart, Jean-Baptiste Lespiau, Vinicius Zambaldi, et al. (27 authors, DeepMind) · arXiv:1908.09453_
<https://arxiv.org/abs/1908.09453>

**Summary:** DeepMind's reference library for RL and search in games: core API and games in C++ exposed to Python, with algorithms in both languages. It ships n-player, imperfect-information game support, tutorial implementations of CFR on Kuhn and Leduc poker, and 'tools to analyze learning dynamics and other common evaluation metrics.' Actively maintained (5.4k stars, 5,500+ commits as of this survey).

**Key Ideas:** One shared extensive-form game API so that any algorithm (CFR variants, best-response/exploitability computation, RL baselines) runs on any registered game; ground-truth evaluation utilities for small games.

**Tradeoffs:** The most trustworthy exploitability oracle available and the widest algorithm coverage, but registering a brand-new C++ game (a Drawmaha variant) is real engineering work, and its Deep CFR implementation has open user-reported convergence issues (e.g. GitHub issue #1287, 'deep_cfr not working on Kuhn poker') — trust its tabular code more than its neural code.

**Relevance:** Best used as the validation harness: implement toy Drawmaha as an OpenSpiel game and use its best-response/exploitability tooling as the independent referee for the author's own CFR/MCCFR code, even if the production solver is written from scratch.

### Deep Counterfactual Regret Minimization (2019)
_Noam Brown, Adam Lerer, Sam Gross, Tuomas Sandholm · ICML 2019_
<https://arxiv.org/abs/1811.00164>

**Summary:** The paper the Drawmaha project's final phase is modeled on: the first successful non-tabular CFR, using external-sampling MCCFR traversals to train an advantage (regret) network per player plus an average-strategy network, removing the need for hand abstraction. Evaluated on Flop Hold'em Poker (~10^12 nodes, ~10^9 infosets) and heads-up limit hold'em (~10^17 nodes). The appendix is a de-risked hyperparameter recipe: 7-layer card+bet-branch network of only 98,948 parameters, 40M-entry reservoir-sampled advantage and strategy memories, 10,000 traversals per CFR iteration, 4,000 SGD steps of batch 10,000 with Adam at lr 0.001 and gradient-norm clip 1.0 (FHP; 32,000 steps, batch 20,000 for HULH), Linear-CFR weighting with 2/T rescaling.

**Key Ideas:** External sampling so no importance-weight division is needed; value networks retrained from scratch each CFR iteration (their ablation shows retraining reaches substantially lower exploitability than fine-tuning one network); reservoir sampling to keep fixed-size buffers unbiased; exploitability reported in mbb/g.

**Tradeoffs:** Approximation error from two stacked networks (advantage and average-strategy), and no hardware/wall-clock numbers are stated — but the hyperparameters imply single-machine, single-GPU scale, which is exactly the budget class a solo project has. Alternatives: SD-CFR removes one network; ReBeL is stronger but needs datacenter compute.

**Relevance:** The direct blueprint for the Deep CFR phase. Its network sizes and buffer/traversal counts are the correct starting hyperparameters for full 5-card Drawmaha; its card-embedding input scheme (rank+suit+card embeddings summed) generalizes naturally to 5-card hands plus discards.

### Single Deep Counterfactual Regret Minimization (2019)
_Eric Steinberger · arXiv:1901.07621_
<https://arxiv.org/abs/1901.07621>

**Summary:** SD-CFR eliminates Deep CFR's average-strategy network: instead of training a second network on strategy samples, it stores the sequence of advantage-network checkpoints and reconstructs the average policy from them at inference time. This 'has a lower overall approximation error by avoiding the training of an average strategy network' and empirically beats Deep CFR on both exploitability and head-to-head play in poker.

**Key Ideas:** The average strategy is a linearly weighted mixture over iterations' instantaneous policies, so keeping the iteration models (or sampling one per hand) is exact where a trained average network is approximate — one fewer function-approximation error source, less memory, less training.

**Tradeoffs:** Inference must either hold many checkpoints or sample a model per playout, which complicates a live-inference dashboard compared to querying one average-policy net; Deep CFR's single distilled network is simpler to serve. Training-side, SD-CFR is strictly cheaper.

**Relevance:** The main design fork for the Drawmaha project's neural phase: the stated plan (dashboard queries the average-policy network) matches vanilla Deep CFR, but SD-CFR is cheaper and lower-error — a practical middle path is SD-CFR training plus a final one-shot distillation into an average-policy net for the dashboard.

### PokerRL (with the Deep-CFR / SD-CFR reference implementation) (2019)
_Eric Steinberger · GitHub (open-source codebase)_
<https://github.com/EricSteinberger/PokerRL>

**Summary:** The practitioner framework built by SD-CFR's author: a poker-specific multi-agent deep-RL framework with vanilla CFR, CFR+, and Linear CFR baselines, and companion repos implementing Deep CFR, SD-CFR, and NFSP on top of it. Its evaluation suite is the most complete in open source: exact best response for small games, Local Best Response (LBR), an RL-based best response (DDQN), and head-to-head matches. Ray-based distribution runs the same code on a laptop, one many-core box, or an AWS cluster; the Deep-CFR repo reproduces the SD-CFR paper's head-to-head experiments on Leduc.

**Key Ideas:** A flexible poker game engine decoupled from the learning algorithms, plus a graded ladder of exploitability estimators (exact BR → LBR → RL-BR) so evaluation degrades gracefully as the game grows past exact best-response size.

**Tradeoffs:** The single best codebase to read for the project's exact plan, but it is research code frozen around PyTorch 0.4.1 with minimal maintenance (4 commits on master, open issues) — treat it as an architecture and correctness reference to port from, not a base to build on. RLCard/OpenSpiel are healthier dependencies but have no comparable Deep CFR evaluation ladder.

**Relevance:** The closest existing artifact to the Drawmaha end-state (Deep CFR on a custom poker variant with exploitability tracking). Its engine abstraction shows how to slot a nonstandard variant in, and its LBR/RL-BR tools answer the hard question of how to measure a Drawmaha solver where exact best response is intractable.

### RLCard: A Toolkit for Reinforcement Learning in Card Games (2019)
_Daochen Zha, Kwei-Herng Lai, Yuanpu Cao, Songyi Huang, Ruzhe Wei, Junyu Guo, Xia Hu · AAAI-20 Workshop on Reinforcement Learning in Games_
<https://arxiv.org/abs/1910.04376>

**Summary:** A Python-first card-game RL toolkit with easy gym-style interfaces, spanning Blackjack, Leduc hold'em, Texas hold'em, UNO, Dou Dizhu, and Mahjong, aimed at 'bridging reinforcement learning and imperfect information games.' It prioritizes accessibility over game-theoretic rigor.

**Key Ideas:** Uniform environment abstraction over heterogeneous card games with simple state encodings and bundled baseline agents, so an applied-ML person can run an experiment in minutes.

**Tradeoffs:** Much lower barrier to entry than OpenSpiel (pure Python, familiar env API) but far weaker equilibrium tooling — nothing like OpenSpiel's exact exploitability machinery or PokerRL's BR ladder — and its CFR support is limited to small games. Fine for prototyping and NFSP/DQN-style baselines; not sufficient as a solver's evaluation ground truth.

**Relevance:** For Drawmaha it is mainly a source of clean, readable environment-design patterns (how to encode cards/actions for a network) and a quick sandbox for the Leduc validation phase; the solver itself should not depend on it.

### Combining Deep Reinforcement Learning and Search for Imperfect-Information Games (ReBeL) (2020)
_Noam Brown, Anton Bakhtin, Adam Lerer, Qucheng Gong · NeurIPS 2020_
<https://arxiv.org/abs/2007.13544>

**Summary:** ReBeL unifies self-play RL and test-time search via public belief states, provably converging to Nash in two-player zero-sum games, and reached superhuman HUNL with far less domain knowledge than prior poker AIs. Its appendix is the field's most honest compute disclosure: value/policy nets are 6-hidden-layer, 1536-unit MLPs (GeLU+LayerNorm); training ran on a single machine but data generation used 90 DGX-1 machines (8×32GB V100 each — 720 GPUs; the paper cites 'up to 128 machines with 8 GPUs each'); 1,750 epochs of 2,560,000 examples at batch 1024; subgame solving used 1024 CFR iterations. The poker code was never released — only a Liar's Dice implementation (trained on one GPU + 60 CPU threads).

**Key Ideas:** Convert the imperfect-information game into a continuous-state perfect-information game over public belief states, learn a value function on those, and run CFR-based search in depth-limited subgames at both train and test time.

**Tradeoffs:** Strongest known approach and the conceptual basis of modern commercial solvers, but data generation (sequential CFR solving of subgames at every leaf) is the bottleneck, putting faithful reproduction ~3 orders of magnitude beyond hobbyist budgets; belief-state input size also scales with the number of possible private hands — 2,598,960 in Drawmaha vs 1,326 in hold'em, making the belief vector itself enormous.

**Relevance:** For the survey, ReBeL is the honest upper bound that justifies the project's Deep CFR choice: the compute table plus the 2.6M-hand belief-state blowup is the concrete argument for why a solo Drawmaha solver should stop at Deep CFR rather than attempt belief-state search.

### PokerKit: A Comprehensive Python Library for Fine-Grained Multi-Variant Poker Game Simulations (2023)
_Juho Kim · IEEE Transactions on Games_
<https://arxiv.org/abs/2308.07327>

**Summary:** A modern, heavily tested (99% coverage, static typing) Python library for simulating 'an extensive array of poker variants' with fine-grained state control, a flexible architecture for user-defined custom games, and a unified hand-evaluation suite across different hand types. Explicitly aimed at poker AI development and tool building.

**Key Ideas:** Decompose a poker variant into composable street/action/showdown primitives so unusual games (draws, splits, mixed games) are configuration rather than new engines; one evaluation API across many hand-ranking types.

**Tradeoffs:** The best off-the-shelf fit for a nonstandard variant's rules engine and far better maintained than PokerRL, but it is a game-state library, not an RL/CFR framework — no exploitability tools, no learning algorithms — and pure-Python evaluation is slower than the C++ evaluators, which matters inside million-traversal MCCFR loops.

**Relevance:** The most Drawmaha-relevant engineering find of this lane: draw mechanics and multi-hand-type showdowns (needed for the split pot) are exactly the fine-grained variant territory PokerKit was built for. Use it to prototype and unit-test the Drawmaha rules engine and as the correctness oracle for a faster custom engine.

### Fast poker hand evaluation: Cactus Kev's 7,462-class evaluator and its perfect-hash descendants (Senzee, TwoPlusTwo, OMPEval, PokerHandEvaluator) (2006)
_Kevin 'Cactus Kev' Suffecool; Paul Senzee; zekyll (OMPEval); Henry Lee (PokerHandEvaluator) · Web/GitHub (community engineering lineage; Senzee's perfect-hash post 2006)_
<http://suffe.cool/poker/evaluator.html>

**Summary:** The hand-evaluation lineage every fast solver leans on. Cactus Kev showed all 5-card hands collapse to 7,462 distinct rank classes reachable via clever card encoding plus lookup; Paul Senzee replaced the final binary search with a precomputed perfect hash (~2.7x faster), the technique behind the TwoPlusTwo-forum table evaluator; OMPEval (github.com/zekyll/OMPEval) is a fast C++ evaluator handling 0-7 cards with only ~200kB of tables and ~10ms init; PokerHandEvaluator (github.com/HenryRLee/PokerHandEvaluator) uses perfect hashing for 7-card and Omaha evaluation, avoiding the naive 21x '7-choose-5 then evaluate' loop.

**Key Ideas:** Precompute the entire hand-rank function into a table indexed by a perfect hash of the card multiset (or an incremental card-by-card table walk), so a showdown evaluation is a handful of memory lookups.

**Tradeoffs:** Tables trade memory for speed (200kB for OMPEval vs tens of MB for uncompressed Cactus Kev-style tables vs larger transition-table designs); C/C++ evaluators are 10-100x faster than Python ones, but must be wrapped for a PyTorch pipeline. All are high-hand hold'em-centric: none natively speaks a split-pot draw variant.

**Relevance:** Drawmaha showdowns need two evaluations per terminal node (5-card hand and Omaha-style board hand) inside MCCFR loops running millions of traversals — evaluator speed is on the critical path. Since Drawmaha hands are exactly 5 cards, Cactus Kev-style 5-card perfect hashing is a perfect fit, with PokerHandEvaluator's Omaha mode covering the board side.

