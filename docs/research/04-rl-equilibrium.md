# Research lane: rl-equilibrium

_Generated 2026-08-28 by a 9-agent literature sweep (Semantic Scholar / arXiv / web). Source material for the ML-survey paper._

## Lane narrative

THE STORY OF THIS LANE. In two-player zero-sum imperfect-information games, equilibrium play is generally a MIXED strategy — you must bluff with calibrated frequencies. That single fact breaks the applied-RL default. Naive self-play policy gradient ("train PPO against a copy of itself") does gradient ascent against the CURRENT opponent, and the joint dynamics in zero-sum games are rotational, not attractive: Perolat et al. (arXiv 2002.08456, ICML) proved Follow-the-Regularized-Leader dynamics are Poincare-recurrent in imperfect-information zero-sum games — trajectories orbit the equilibrium forever, returning near their start, like rock-paper-scissors best-response chasing. Crucially, the TIME-AVERAGE of the orbit does converge; the last iterate does not. Every method in this lane is one of three ways to cash that observation.

OPTION A — AVERAGE: fictitious play best-responds to the opponent's historical average. NFSP (Heinrich & Silver 2016) neuralizes this with a DQN best-response net plus a supervised average-policy net; you deploy the average net. Simple, sample-based, proven on Leduc and Limit Hold'em — but slow, and background knowledge (not re-verified this session) says Deep CFR beat it in 2019. Note CFR itself is the same trick in regret space: its guarantee is on the AVERAGE strategy, which is why the Drawmaha plan's average-policy network is exactly the right deployment object.

OPTION B — POPULATION: PSRO (Lanctot et al., NeurIPS 2017) turns the game into a small meta-game over whole trained policies, solves it, and trains a new RL best response to the meta-Nash mixture; Pipeline PSRO (McAleer et al., NeurIPS 2020) parallelizes it with convergence intact and won Barrage Stratego (~10^50 nodes). Each iteration costs a full RL run, and mixtures over whole policies represent poker's fine-grained mixing coarsely — understand it, skip it for Drawmaha.

OPTION C — REGULARIZE THE DYNAMICS: add a KL penalty toward a reference ("magnet") policy; the regularized game has a unique equilibrium the dynamics CONTRACT to (last-iterate convergence); then move the magnet to the solution and repeat — fixed points are true Nash. R-NaD scaled this to Stratego (10^535 nodes): DeepNash (Science 2022) reached top-3 on Gravon with no search, model-free. MMD (Sokota et al., ICLR 2023) is the minimal version — mirror descent plus a magnet term, linear convergence to quantal-response equilibria, and per its abstract the first standard RL algorithm empirically competitive with CFR in tabular settings — though its deep-RL evidence is only 3x3 Dark Hex and Phantom Tic-Tac-Toe.

PPO IN POKER, HONESTLY: AlphaHoldem (AAAI 2022) — Trinal-Clip PPO plus a K-best historical opponent pool — beat Slumbot by 111.56 mbb/h and pros by 10.27 mbb/h after three days on one PC. But head-to-head is a weak proxy for equilibrium quality: per the Approximate Exploitability line of work (Timbers et al., IJCAI 2022; seen via search snippet, not the full paper), Act1 and Slumbot were statistically indistinguishable head-to-head while Act1 was ~1300 mbb/g LESS exploitable under local best response. Fan & Farina (2026 preprint) add a mechanism: GAE's advantage variance is inflated precisely because equilibrium play is stochastic, persisting even with a perfect critic; their VRPO fix lowers exploitability versus PPO baselines on HUNL.

WHY ALPHAZERO WORKS NEXT DOOR: perfect information gives every state an opponent-independent minimax value and admits a deterministic optimal policy, so MCTS-amplified greedy self-play is a genuine improvement loop. Hidden 5-card hands destroy both pillars: values depend on beliefs, beliefs on both strategies, and greedy improvement cycles.

FOR DRAWMAHA: the planned CFR-to-MCCFR-to-Deep-CFR line remains the right spine — it is the only family with routine exploitability tracking at toy scale. From this lane, the two worthwhile add-ons are NFSP as an independent neural baseline on the shrunken variant, and tabular MMD in the Kuhn/Leduc harness (one extra loss term over standard policy mirror descent). Split-pot changes nothing theoretically: two-player Drawmaha is still constant-sum, so all zero-sum convergence theory carries over.

CAVEATS / NOT FOUND: I found no published equilibrium-learning work on Drawmaha or any split-pot draw variant (not exhaustively searched). Fine mechanism details beyond abstracts (NFSP's reservoir buffer, OpenSpiel's R-NaD/NFSP implementations, Deep CFR-beats-NFSP) are background knowledge, flagged as such. DeepNash's compute budget is not stated in its abstract and I could not verify it; the Act1/Slumbot exploitability figure comes from a search snippet rather than the fetched paper.

## Papers

### Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (NFSP) (2016)
_Johannes Heinrich, David Silver · arXiv preprint 1603.01121 (widely cited; never a main-conference paper) · 475 citations_
<https://arxiv.org/abs/1603.01121>

**Summary:** Neural Fictitious Self-Play (NFSP) is the first scalable end-to-end deep-learning method that approaches a Nash equilibrium in imperfect-information games without hand-crafted abstraction. It neuralizes fictitious play: instead of best-responding to the opponent's current policy (which cycles), each agent best-responds to the opponent's time-AVERAGE policy, and that average itself converges to equilibrium. On Leduc poker NFSP approached Nash while common RL methods (DQN-style self-play) diverged; on Limit Texas Hold'em it approached the performance of state-of-the-art handcrafted-abstraction algorithms of the time.

**Key Ideas:** Two networks per agent: a best-response network trained by off-policy RL (DQN) against the opponents' average policies, and an average-policy network trained by supervised learning on the agent's own historical best-response actions (reservoir-sampled buffer). Agents act with a mix ('anticipatory dynamics') of the two. The averaging is what restores convergence — fictitious play's time-average converges in two-player zero-sum games even though the iterates themselves cycle. The average-policy network is the deliverable you play with.

**Tradeoffs:** Pros: conceptually the simplest deep equilibrium method; standard components (DQN + supervised net); no game-tree traversals needed, purely sample-based. Cons: fictitious-play convergence is slow (poor rate even in theory); later CFR-based deep methods (Deep CFR, 2019) reported beating NFSP head-to-head and on exploitability — this comparison is background knowledge, not verified this session. Off-policy RL instability adds tuning burden.

**Relevance:** The natural first NON-CFR baseline for Drawmaha: it needs only a simulator, tolerates the 2.6M-hand state space via function approximation, and its 'query the average-policy network' deployment exactly matches the planned live-inference dashboard pattern. Running NFSP on the shrunken variant next to MCCFR gives an independent sanity check on the equilibrium both should find.

### A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning (PSRO) (2017)
_Marc Lanctot, Vinicius Zambaldi, Audrunas Gruslys, Angeliki Lazaridou, Karl Tuyls, Julien Perolat, David Silver, Thore Graepel · NeurIPS 2017 · 767 citations_
<https://arxiv.org/abs/1711.00832>

**Summary:** Diagnoses why independent RL self-play fails: policies overfit to the specific opponents seen in training ('joint-policy correlation') and don't generalize at test time. Proposes Policy-Space Response Oracles (PSRO): keep a growing population of policies, build the empirical payoff matrix among them, solve that small matrix game for a meta-strategy (e.g., its Nash), then train a new deep-RL best response against that mixture and add it to the pool. Generalizes fictitious play, double oracle, and iterated best response in one framework; demonstrated on gridworld games and poker.

**Key Ideas:** Lift the game from action space to POLICY space: the intractable extensive-form game becomes a small normal-form meta-game over trained policies, which a classical solver can handle. Convergence comes from the double-oracle argument — if no new policy can beat the current meta-Nash mixture, that mixture is an equilibrium of the full game.

**Tradeoffs:** Pros: modular (any RL algorithm can be the oracle), naturally parallel over evaluations, produces interpretable populations. Cons: each iteration is a FULL RL training run, so wall-clock cost is enormous; the payoff matrix needs many rollouts; in games needing finely mixed equilibria (poker) the population needed can be large. Much more expensive per unit of exploitability reduction than CFR-family methods on poker benchmarks.

**Relevance:** For heads-up Drawmaha, PSRO is the concept to understand but probably not the tool to use: poker-style games need precise mixing (bluff frequencies), which PSRO approximates coarsely through a mixture over whole policies. Worth knowing because it frames every self-play method as 'who do I train against?' — the question the Drawmaha training loop must answer.

### Pipeline PSRO: A Scalable Approach for Finding Approximate Nash Equilibria in Large Games (2020)
_Stephen McAleer, John Lanier, Roy Fox, Pierre Baldi · NeurIPS 2020 · 98 citations_
<https://arxiv.org/abs/2006.08555>

**Summary:** Attacks PSRO's main practical flaw — it is sequential, training one best response at a time. P2SRO runs a hierarchical pipeline of RL workers, each training against the policies produced by the levels below it, so many best responses train concurrently while the convergence guarantee to approximate Nash is preserved (unlike earlier parallelizations DCH and Rectified PSRO, which can fail to converge). Achieved state-of-the-art on Barrage Stratego (game tree ~10^50), beating all existing bots.

**Key Ideas:** Parallelism with a hierarchy: lower pipeline levels have 'fixed' policies that upper levels treat as part of the meta-game, so the theoretical structure of double oracle survives parallel training. More workers = faster convergence, same guarantee.

**Tradeoffs:** Pros: the practical way to run PSRO at scale; open-source code. Cons: inherits PSRO's per-iteration cost (each worker is a full RL run); infrastructure-heavy (distributed workers); still coarse at representing finely mixed poker-style equilibria compared to CFR-family averaging.

**Relevance:** Included as the 'successor' showing the PSRO line matured into a real large-game tool. For a solo Drawmaha project on one machine, P2SRO's infrastructure cost is the clearest argument for staying with MCCFR/Deep CFR, which get their convergence from cheap averaged iterates instead of populations of trained networks.

### From Poincaré Recurrence to Convergence in Imperfect Information Games: Finding Equilibrium via Regularization (2020)
_Julien Perolat, Rémi Munos, Jean-Baptiste Lespiau, et al. (DeepMind) · ICML 2021 (per Semantic Scholar venue field; arXiv 2020) · 109 citations_
<https://arxiv.org/abs/2002.08456>

**Summary:** The theory paper behind this whole lane's central warning. It proves that Follow-the-Regularized-Leader dynamics — the idealized version of what policy-gradient self-play does — are Poincaré-RECURRENT in two-player zero-sum imperfect-information games: trajectories orbit the equilibrium and return arbitrarily close to where they started, forever, instead of converging. It then shows the fix: transform the reward by adding a regularization term (a KL penalty toward a reference policy), which makes the dynamics converge, and iterating this reward transformation converges exactly to Nash, yielding practical model-free algorithms.

**Key Ideas:** Cycling is not a hyperparameter problem — it is a geometric property of the learning vector field in zero-sum games (rotation around the equilibrium, no attraction). Regularization tilts the field inward: the entropy/KL-regularized game has a UNIQUE equilibrium the dynamics contract to. Solve the regularized game, move the reference policy to the solution, re-regularize, repeat — the fixed points of this outer loop are Nash equilibria of the true game. This is the direct ancestor of R-NaD/DeepNash.

**Tradeoffs:** Pros: gives last-iterate convergence — the policy you have at the end is the answer, no averaging over history needed (a real deployment advantage for neural nets, since averaging networks is awkward). Cons: theory is for exact dynamics; with sampling and function approximation the guarantees soften; the regularization schedule (how fast to move the magnet) is a new tuning axis.

**Relevance:** For the survey, this is THE citation for 'why naive actor-critic self-play fails in poker.' For Drawmaha it also explains a subtle point relevant to Deep CFR: methods whose guarantee lives in the AVERAGE policy (CFR, fictitious play) must train/keep an average-policy network — exactly what the planned dashboard queries — whereas regularized-dynamics methods make the final network itself the equilibrium.

### Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning (DeepNash / R-NaD) (2022)
_Julien Perolat, Bart De Vylder, Daniel Hennes, et al. (DeepMind, ~34 authors) · Science (DOI 10.1126/science.add4679) · 293 citations_
<https://arxiv.org/abs/2206.15378>

**Summary:** Scales the regularized-Nash-dynamics idea to Stratego, an imperfect-information game with ~10^535 game-tree nodes — vastly larger than Go (~10^170) or Texas Hold'em (~10^164). DeepNash is model-free self-play with no explicit game-tree search: the R-NaD algorithm 'directly modifies the underlying multi-agent learning dynamics' so that instead of cycling around Nash, the policy converges to an approximate equilibrium. It beat all prior Stratego bots and reached a top-3 all-time ranking against expert humans on the Gravon platform in 2022, learning bluffing and information-hiding behavior en route.

**Key Ideas:** Three-step loop at deep-RL scale: (1) regularize rewards with a KL penalty toward a reference ('magnet') policy, (2) run actor-critic-style learning to the regularized game's unique fixed point, (3) update the reference to the converged policy and repeat. Last-iterate convergence means no averaging and no search at deployment — the network's policy is played directly, mixed actions included.

**Tradeoffs:** Pros: the strongest existence proof that pure RL (no CFR, no search) reaches equilibrium-quality play in a huge imperfect-info game; deployment is a single cheap forward pass. Cons: DeepMind-scale engineering and compute (exact budget not stated in the abstract; not verified here); many interacting pieces (regularization schedule, neural replicator-dynamics losses); on poker specifically there is no published evidence it beats Deep CFR-style methods per unit compute.

**Relevance:** The aspirational endpoint of the RL lane: if Deep CFR ever proves unworkable on full Drawmaha, R-NaD is the principled RL alternative (an implementation exists in OpenSpiel — background knowledge, not verified this session). Also the survey's best evidence that 'model-free RL cannot handle imperfect information' is false — it just needs modified dynamics, not vanilla PPO.

### A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games (Magnetic Mirror Descent) (2022)
_Samuel Sokota, Ryan D'Orazio, J. Zico Kolter, Nicolas Loizou, Marc Lanctot, Ioannis Mitliagkas, Noam Brown, Christian Kroer · ICLR 2023 · 91 citations_
<https://arxiv.org/abs/2206.05825>

**Summary:** Introduces Magnetic Mirror Descent (MMD), a simple first-order algorithm — mirror descent plus an extra proximal 'magnet' term pulling toward a reference policy — and shows one algorithm serves three roles: it is the first linearly-convergent solver for quantal response equilibria (entropy-regularized Nash) in extensive-form games using only first-order feedback; run as tabular RL it is 'the first standard reinforcement learning algorithm to achieve empirically competitive results with CFR in tabular settings'; and as a deep self-play RL algorithm it performs well on 3x3 Dark Hex and Phantom Tic-Tac-Toe.

**Key Ideas:** Same regularization insight as R-NaD but in its minimal form: adding a KL-toward-magnet term to the mirror-descent update turns the rotational zero-sum dynamics into a contraction with LINEAR last-iterate convergence to the regularized equilibrium; annealing the regularization temperature approaches exact Nash. Practically, MMD looks like a small modification of standard policy-mirror-descent / PPO-style RL — which is exactly why it matters: it locates precisely where actor-critic 'fits' in equilibrium finding.

**Tradeoffs:** Pros: the cheapest bridge from an applied-RL toolkit to equilibrium convergence — one extra loss term; strong theory; open-source code (github.com/ssokota/mmd). Cons: converges to the QUANTAL-RESPONSE (slightly softened) equilibrium unless annealed; competitive-with-CFR evidence is tabular — deep-RL results are on small games (Dark Hex 3x3, Phantom TTT), not full-scale poker; CFR+ still wins raw tabular speed on many benchmarks.

**Relevance:** The most actionable RL-side option for Drawmaha: if the author wants an RL baseline next to Deep CFR, MMD-style regularized policy optimization is the version of 'PPO self-play' that actually has a convergence story, and its tabular form can be validated on the same Kuhn/Leduc harness planned for CFR.

### AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via End-to-End Reinforcement Learning (2022)
_Enmin Zhao, Renye Yan, Jinqiu Li, Kai Li, Junliang Xing · AAAI 2022 · 71 citations_
<https://ojs.aaai.org/index.php/AAAI/article/view/20394>

**Summary:** The main published evidence on PPO-style self-play as a poker baseline. AlphaHoldem trains a pseudo-siamese network end-to-end for heads-up no-limit hold'em with a 'Trinal-Clip' PPO loss and 'K-Best Self-Play' (the current model plays a pool of its K best historical versions). Trained in three days on one PC (~2.7 billion hands / 6.5 billion samples), it beat Slumbot by 111.56 mbb/h, the authors' DeepStack reimplementation (OpenStack) by 16.91 mbb/h, and human professionals by 10.27 mbb/h over ~2,500 hands each, deciding in 2.9 ms — over 1,000x faster than DeepStack at inference.

**Key Ideas:** Make vanilla PPO survive poker's heavy-tailed, high-variance rewards: Trinal-Clip PPO adds a third clip bound (delta-1) on the policy ratio when the advantage is negative, plus clips the value-loss target into [-delta2, delta3] (roughly pot-bounded), stabilizing training when opponents bluff; the K-best historical pool is a lightweight population that reduces overfitting to a single co-evolving opponent.

**Tradeoffs:** Pros: astonishing compute efficiency versus CFR-pipeline bots (days on one PC vs. server clusters); trivially fast deployment; no abstraction engineering. Cons: NO equilibrium guarantee — wins are head-to-head, and head-to-head results are known to be a poor proxy for exploitability (see notes); the historical-pool heuristic is exactly the kind of averaging-substitute that can silently fail; hyperparameter-sensitive.

**Relevance:** Defines the tempting shortcut the Drawmaha project should understand and (mostly) resist: a Trinal-Clip-style PPO could probably produce a fun, strong-seeming Drawmaha agent quickly, but with no existing Drawmaha bots to benchmark against, its exploitability would be unmeasurable — whereas the planned CFR line can track exploitability on the shrunken variant directly.

### Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero) (2017)
_David Silver, Thomas Hubert, Julian Schrittwieser, et al. (DeepMind) · arXiv 1712.01815 (journal version: Science, 2018) · 2131 citations_
<https://arxiv.org/abs/1712.01815>

**Summary:** The 'brief neighbors' contrast case: tabula-rasa self-play with MCTS reached superhuman play in chess, shogi, and Go within 24 hours each, defeating a world-champion program in every game, with no domain knowledge beyond the rules. Included in this lane not as a method to copy but to explain WHY its recipe works in these games and fails in poker.

**Key Ideas:** In perfect-information games every state has a well-defined minimax value independent of who is playing or what they believe, so (a) a deterministic optimal policy exists — no mixing needed, and (b) MCTS acts as a policy-improvement operator: search-amplified targets always point 'uphill,' making greedy self-play iteration a genuine improvement loop rather than a cycle.

**Tradeoffs:** Pros: simplest and best-scaled self-play recipe where it applies. Cons: both pillars collapse under hidden information — an infoset's value depends on the opponent's strategy off the observed path (so no opponent-independent value target exists), equilibria require randomization (so greedy argmax over values is wrong), and naive MCTS values computed without belief distributions are simply incorrect. Applying AlphaZero machinery directly to poker-like games yields exploitable agents.

**Relevance:** Gives the survey its cleanest one-paragraph explanation of what makes Drawmaha hard: the draw round and hidden 5-card hands mean value depends on beliefs, beliefs depend on both players' strategies, and therefore the training loop must target equilibria (CFR averaging or regularized dynamics), not value maximization against the current opponent.

### GAE Falls Short in Imperfect-Information Self-Play Reinforcement Learning (VRPO) (2026)
_Zhiyuan Fan, Gabriele Farina · arXiv preprint 2605.19235 (not yet peer-reviewed)_
<https://arxiv.org/abs/2605.19235>

**Summary:** Recent evidence pinpointing a concrete mechanical reason PPO underperforms in imperfect-information self-play: generalized advantage estimation (GAE) carries extra variance from sampling stochastic future actions — and because equilibrium play in these games is deliberately stochastic (bluffing frequencies), this variance persists even with a PERFECT critic. The authors propose Q-boosting and VRPO, replacing sampled multi-step backups with a multi-step Expected-SARSA(lambda) trace that averages out action-sampling noise while keeping PPO's clipping machinery, and report consistently lower exploitability than other PPO-based baselines on mid-sized benchmarks, with strong performance on Dou Dizhu and heads-up no-limit hold'em.

**Key Ideas:** Mixed (stochastic) equilibrium policies are not just a convergence problem for PPO — they are a VARIANCE problem: the noisier the equilibrium mixing, the noisier GAE's advantage targets. Computing expectations over own future actions (Expected SARSA traces) removes exactly that noise source.

**Tradeoffs:** Pros: a surgical, drop-in fix to standard PPO infrastructure; measured on exploitability, the right metric, not just head-to-head. Cons: 2026 preprint, unreviewed and uncited so far; requires Q-value heads over actions, which grows with action-space size.

**Relevance:** If the author ever benchmarks a PPO self-play baseline for the survey's honesty ('we tried the naive thing'), this paper predicts the failure mode to look for (high-variance advantages near mixed-strategy spots like bluff/call decisions after the draw) and provides the corrected baseline that makes the comparison fair.

