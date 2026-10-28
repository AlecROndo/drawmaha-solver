# Research lane: neural-cfr

_Generated 2026-08-28 by a 9-agent literature sweep (Semantic Scholar / arXiv / web). Source material for the ML-survey paper._

## Lane narrative

The neural-CFR story is about replacing CFR's regret TABLE with a regression problem. Tabular CFR must visit and store every infoset; Drawmaha's 2.6M hands times betting/draw histories makes that impossible without brutal abstraction. Deep CFR's move: sample the tree (external sampling — the traverser tries all its actions, opponent and chance sample one), record instantaneous advantages into a reservoir buffer, and train a per-player 'advantage network' from scratch each iteration; regret matching over its outputs is that iteration's strategy. Crucially the thing that converges to Nash is the AVERAGE of all iterates, not the last one — so Deep CFR also fills a strategy buffer and distills a final average-policy network, which is what you deploy (and what the planned dashboard should query).

The main builder options, in order of decision importance: (1) Sampling scheme. With your own simulator, use external sampling (Deep CFR/SD-CFR): no importance-sampling weights for the traverser, low variance. Outcome sampling (DREAM) exists for black-box simulators and pays for it with 1/reach importance weights whose variance requires an extra learned baseline network; ESCHER later showed removing importance sampling entirely (via a learned history-value function under a fixed sampling policy) is what makes model-free neural CFR scale. For Drawmaha the lesson is simply: you have the simulator, take the low-variance path. (2) Average-strategy recovery. Deep CFR distills a second network (extra sampling + approximation error); SD-CFR instead stores every iteration's value net (~400KB each) and reconstructs the exact average — measurably better exploitability and head-to-head play. Best hybrid for this project: run SD-CFR-style (keep checkpoints, evaluate with the exact average), then distill one average-policy net at the very end for the dashboard. (3) From-scratch retraining vs warm-starting. Deep CFR retrains each iteration because cumulative-regret targets are nonstationary and grow with T; Double Neural CFR warm-starts a cumulative-regret net (cheaper, but approximation error compounds); ReCFR reformulates targets as per-iteration recursive substitute values to dodge the problem. Retrain-from-scratch is the safe default. (4) Vanilla vs discounted updates. Tabular stages should use CFR+/DCFR; neural stages historically couldn't, and Deep (Predictive) Discounted CFR (AAAI 2026) is the first credible bridge — cite it, don't start with it.

Concrete anchors worth copying: Deep CFR used a ~99K-parameter 7-layer net (card branch + bet branch), cards encoded as summed rank/suit/card embeddings pooled over permutation-invariant groups — this transfers beautifully to 5-card Drawmaha hands and discards; Adam lr 0.001, grad clip 1, batch 10K, 4K SGD steps and 10K traversals per iteration, 40M reservoir buffers (SD-CFR used 1M buffers and 3x64 nets for Leduc — the right scale for the shrunken variant). Strategy-buffer samples must be weighted by iteration t (Linear-CFR-style averaging); forgetting this weighting is a classic silent bug.

Pitfalls: (a) regret-target variance/nonstationarity — losses that plateau high or climb are often target drift, not optimizer failure (ReCFR's problem statement is the best explanation on record); (b) distillation error — measure exploitability against the SD-CFR exact average, or you'll blame CFR for what is distillation noise; (c) outcome-sampling importance weights explode without baselines — avoid the whole issue via external sampling; (d) reservoir sampling must be genuinely uniform-over-all-iterations, and buffers too small quietly forget early iterations. Drawmaha-specific: the draw decision is a 32-way (discard-subset) action, so external-sampling traversal cost multiplies at draw nodes, and the C(47,k) replacement-card chance branching is handled by sampling — both fine, but budget traversals accordingly.

Caveats and gaps: All seven papers were confirmed by fetch (titles/venues/citation counts via Semantic Scholar's batch endpoint on 2026-08-28). Hyperparameter quotes come from WebFetch summaries of the ar5iv full texts, faithful but secondhand — re-check exact numbers against the PDFs before publishing tables. The Linear-CFR sample weighting and OpenSpiel's ready-made Deep CFR implementations (TF/PyTorch/JAX) are from background knowledge, not re-verified this session. SD-CFR and DREAM are arXiv-only (never conference-published) despite wide citation — worth stating in the survey. I found no published neural-CFR work on draw-poker variants specifically; 'Robust Deep MCCFR' (arXiv 2509.00923, 2025) surfaced in search but was not fetched/verified in depth, so it is omitted from the main list.

## Papers

### Deep Counterfactual Regret Minimization (2019)
_Noam Brown, Adam Lerer, Sam Gross, Tuomas Sandholm · ICML 2019 (arXiv 1811.00164) · 263 citations_
<https://arxiv.org/abs/1811.00164>

**Summary:** The founding paper of neural CFR and the planned centerpiece for Drawmaha. Instead of storing a regret table over every infoset, each CFR iteration runs external-sampling traversals of the game tree, records sampled 'advantages' (instantaneous counterfactual regrets) into a reservoir buffer, and trains a value network per player from scratch to predict them; the current strategy is regret matching over the network's outputs. A separate average-policy network is trained at the end from a strategy buffer, and that network is the final output. First non-tabular CFR to work in large poker: beats NFSP on Flop Hold'em (37 vs 47 mbb/g exploitability) and is competitive head-to-head with a 3.3x10^8-bucket abstraction in heads-up limit hold'em (-11±2 mbb/g).

**Key Ideas:** External-sampling MCCFR traversals feed per-player reservoir buffers (40M samples each); advantage nets retrained FROM SCRATCH every iteration with samples weighted by iteration (approximating Linear CFR); regret matching on predicted advantages gives the iterate strategy; a final average-policy network distilled from the strategy buffer is what you actually play. Card encoding: sum of rank+suit+card embeddings, sum-pooled within each permutation-invariant card group; bets as binary-occurred + float-size features; 7-layer net, ~99K parameters; Adam lr 0.001, grad-norm clip 1, batch 10K-20K, 4K-32K SGD steps/iteration, 10K traversals/iteration.

**Tradeoffs:** Pros: principled (provable convergence in the tabular limit), no hand-crafted card abstraction, modest network sizes, the whole recipe is published with hyperparameters. Cons: requires a perfect simulator (fine for Drawmaha — you own the engine); retraining from scratch each iteration is compute-heavy; the distilled average-policy net adds an extra approximation error on top of the value nets (SD-CFR's critique); cumulative-regret nonstationarity makes the targets drift across iterations.

**Relevance:** This is the algorithm the project plans to implement, and every quantity you must choose (buffer size, traversals/iter, net width, when to train the avg-policy net) has a published anchor here. The card-embedding-sum encoding transfers directly to Drawmaha's 5-card hands (permutation invariance over hole cards is even more valuable with C(52,5)=2.6M hands), and the average-policy network is exactly what the live-inference dashboard should query.

### Single Deep Counterfactual Regret Minimization (2019)
_Eric Steinberger · arXiv 1901.07621 (not conference-published) · 43 citations_
<https://arxiv.org/abs/1901.07621>

**Summary:** SD-CFR deletes Deep CFR's average-policy network entirely. Since the average strategy is just an iteration-weighted mixture of the per-iteration current strategies, and each current strategy is recoverable from that iteration's value network, SD-CFR stores every iteration's value net (~400KB each) and either samples one net per trajectory proportional to its iteration weight or averages all nets' induced strategies at a queried infoset. This removes the strategy buffer and the distillation training run — one fewer sampling error and one fewer approximation error — and it beats Deep CFR on both exploitability (Leduc) and head-to-head play (5-Flop Hold'em).

**Key Ideas:** The average policy is computed exactly from stored value networks instead of being approximated by a second network; trajectory-sampling makes play-time cost equal to one network; disk is the only growing cost. Leduc config: 3x64 FC layers, 1M buffer/player, 1,500 traversals/iter; 5-FHP config: 40M buffers, 300K traversals/iter, batch ~10K.

**Tradeoffs:** Pros: strictly less approximation error than Deep CFR, simpler (no strategy buffer/second training pipeline), and the exploitability measurements are more honest since no distillation noise. Cons: must keep hundreds of checkpoints (storage and load management); querying the exact average at one infoset means evaluating ALL stored nets — awkward for a low-latency dashboard; no conference venue (arXiv-only, but widely cited and reimplemented, e.g. as the basis of the author's open-source PokerRL/DREAM code).

**Relevance:** The cheapest big win over vanilla Deep CFR for Drawmaha: implement Deep CFR's traverser, but keep per-iteration value-net checkpoints and evaluate exploitability the SD-CFR way. For the dashboard you can still distill a single average-policy net at the very end from the checkpoint ensemble — you get SD-CFR's training accuracy and Deep CFR's one-net inference.

### DREAM: Deep Regret minimization with Advantage baselines and Model-free learning (2020)
_Eric Steinberger, Adam Lerer, Noam Brown · arXiv 2006.10410 (not conference-published) · 67 citations_
<https://arxiv.org/abs/2006.10410>

**Summary:** DREAM makes deep CFR model-free: it uses outcome-sampling MCCFR (one action sampled per decision, like standard RL rollouts) instead of external sampling, so it never needs to reset the simulator to enumerate sibling actions. The catch is that outcome sampling divides by tiny reach probabilities, exploding variance; DREAM tames this with a learned history-value baseline network Q̂ trained by expected SARSA, subtracting a predicted state-action value before the importance weight is applied. It matches SD-CFR's results on Leduc and FHP despite being simulator-free, and beats NFSP by about two orders of magnitude in sample efficiency.

**Key Ideas:** Outcome sampling + epsilon-mixed exploration policy + neural analogue of tabular VR-MCCFR baselines; three networks per player (advantage, baseline Q, and average policy). Reported config: baseline net trained 1,000 minibatches of 512/iter at lr 0.001; advantage nets 3,000 batches of 2,048 (Leduc) / 10,000 (FHP); buffers 2M (Leduc) / 40M (FHP); ~900 traversals/iter in Leduc.

**Tradeoffs:** Pros: the only choice when you cannot enumerate actions at a state (black-box simulators); best-in-class among model-free methods at the time. Cons: importance sampling variance is reduced, not eliminated — still noisier than external sampling; a third network to tune; convergence in big games remains behind simulator-based methods (ESCHER's critique).

**Relevance:** For Drawmaha you have a perfect simulator, so DREAM is NOT the recommended path — but it is the survey's clearest lesson on WHY external sampling is the default: it shows exactly where outcome-sampling variance comes from (1/reach importance weights) and how much machinery it takes to fight it. Understanding it also explains what ESCHER and later work are improving on.

### ESCHER: Eschewing Importance Sampling in Games by Computing a History Value Function to Estimate Regret (2022)
_Stephen McAleer, Gabriele Farina, Marc Lanctot, Tuomas Sandholm · ICLR 2023 (arXiv 2206.04122) · 38 citations_
<https://arxiv.org/abs/2206.04122>

**Summary:** ESCHER removes importance sampling from model-free neural CFR altogether. It trains a neural history-value function (value of a full game state, not just an infoset) under a FIXED sampling policy that does not depend on the player's current strategy; because the sampling distribution is fixed, regret estimates are unbiased without any reach-probability correction terms. Estimated regret variance is orders of magnitude lower than DREAM's, and the gap in playing strength grows with game size: in dark chess ESCHER beats DREAM and NFSP head-to-head over 90% of the time.

**Key Ideas:** Replace 'importance-weight the sampled payoff' with 'query a learned history value function' when computing counterfactual regret targets; fixed (strategy-independent) sampling policy makes the estimator unbiased; convergence to approximate Nash with high probability.

**Tradeoffs:** Pros: dramatically lower variance than any importance-sampling neural CFR, scales to games far beyond poker benchmarks, principled guarantees. Cons: needs a good history-value net (an extra learned object whose own error now enters the regret targets); more moving parts than Deep CFR; empirical poker-specific tuning less documented than Deep CFR's.

**Relevance:** Two takeaways for Drawmaha: (1) if the Deep CFR advantage-net loss looks noisy and won't converge on the full game, variance in regret targets is the first suspect and ESCHER is the modern fix; (2) its diagnosis — importance-sampling terms are what kill neural CFR at scale — justifies choosing external sampling (no such terms for the traverser) from day one.

### Double Neural Counterfactual Regret Minimization (2020)
_Hui Li, Kailiang Hu, Shaohua Zhang, Yuan Qi, Le Song · ICLR 2020 (arXiv 1812.10607) · 54 citations_
<https://arxiv.org/abs/1812.10607>

**Summary:** Contemporaneous with Deep CFR: one network for cumulative regret, one for the average strategy, developed with a recurrent/attention representation of the betting sequence rather than Deep CFR's flat feature vector. Contributes 'robust sampling' (a variant between outcome and external sampling that samples k actions) and mini-batch MCCFR. Matches tabular CFR's convergence on solvable games and reports strong play on games with billions of nodes using hundreds of times less memory than tabular CFR.

**Key Ideas:** Learn CUMULATIVE regret directly (warm-starting each iteration's network from the last, instead of Deep CFR's retrain-from-scratch on a buffer of instantaneous regrets); robust sampling interpolates between outcome sampling (k=1, cheap, noisy) and external sampling (k=all, expensive, low variance); LSTM+attention over action history.

**Tradeoffs:** Pros: incremental network updates are far cheaper than from-scratch retraining; robust sampling gives a tunable compute/variance dial. Cons: bootstrapping cumulative regret into a warm-started network accumulates approximation error across iterations (the drift problem Deep CFR's retraining avoids); pipeline is more intricate and less widely reproduced than Deep CFR.

**Relevance:** Mostly a design-space marker for the survey: it answers 'why does Deep CFR retrain from scratch — couldn't we just keep updating one net?' (you can, but error compounds). Its robust-sampling idea is worth remembering if full external sampling over Drawmaha's 32-subset discard action becomes too slow.

### Model-Free Neural Counterfactual Regret Minimization with Bootstrap Learning (ReCFR / Neural ReCFR-B) (2022)
_Weiming Liu, Bin Li, Julian Togelius · IEEE Transactions on Games (arXiv 2012.01870, first posted 2020) · 11 citations_
<https://arxiv.org/abs/2012.01870>

**Summary:** Attacks neural CFR's core statistical problem head-on: cumulative regrets keep growing across iterations, so networks that regress them face nonstationary, ever-larger, high-variance targets. ReCFR replaces cumulative regrets with Recursive Substitute Values (RSVs) — recursively defined quantities recomputed independently each iteration that still guarantee O(1/√T) convergence to Nash. Neural ReCFR-B learns RSVs by bootstrapping (TD-style), giving lower-variance training targets, and is competitive with state-of-the-art neural CFRs at much lower training cost.

**Key Ideas:** Non-cumulative, per-iteration training targets (RSVs) recovered recursively replace regret sums; bootstrap/TD learning instead of Monte Carlo regression; model-free.

**Tradeoffs:** Pros: directly fixes the target-variance/nonstationarity pitfall; cheaper training. Cons: low adoption (11 citations) so fewer reference implementations and less community debugging; bootstrapping introduces its own bias; less battle-tested on large poker than Deep CFR.

**Relevance:** For the survey's pitfalls section: this paper is the cleanest published articulation of WHY regressing cumulative regret is statistically nasty — the exact issue the Drawmaha advantage networks will face as iteration counts grow. Reading its problem statement (even skipping the method) sharpens intuition about what the advantage-net loss curve should and shouldn't look like.

### Deep (Predictive) Discounted Counterfactual Regret Minimization (2025)
_Hang Xu, Kai Li, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng · AAAI 2026 (arXiv 2511.08174) · 2 citations_
<https://arxiv.org/abs/2511.08174>

**Summary:** The newest branch of this lane (Nov 2025): existing neural CFRs approximate VANILLA CFR, but the best tabular solvers use discounted/predictive variants (CFR+, DCFR, PCFR+) that converge far faster. This paper builds a model-free neural CFR that collects variance-reduced sampled advantages from a value network, fits CUMULATIVE advantages by bootstrapping, and applies discounting and clipping operations to mimic the update rules of those advanced variants. Reports faster convergence than existing model-free neural algorithms and stronger adversarial performance in a large poker game.

**Key Ideas:** Simulate DCFR/PCFR+-style discounting inside a neural pipeline via discounting + clipping of bootstrapped cumulative advantages; variance-reduced advantage sampling from a value network.

**Tradeoffs:** Pros: imports the 1-2 order-of-magnitude speedups tabular discounting gives; model-free. Cons: brand-new (2 citations), unproven outside the authors' benchmarks, more hyperparameters (discount schedules); implementation details must be mined from the paper rather than community code.

**Relevance:** Signals where the field is going and what the Drawmaha writeup should acknowledge: the tabular-CFR stage of the project will use CFR+/DCFR discounting anyway, and this paper is the current answer to 'why can't my Deep CFR use those same tricks?' Worth citing as future work rather than implementing first.

