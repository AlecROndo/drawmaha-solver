# Research lane: search-value-nets

_Generated 2026-08-28 by a 9-agent literature sweep (Semantic Scholar / arXiv / web). Source material for the ML-survey paper._

## Lane narrative

THE STORY. In perfect-information games, search won: look ahead from the current position, evaluate leaves, back up values (chess engines, AlphaZero). In poker that recipe breaks twice. First, you don't know which state you're in (hidden cards), so you must reason over ranges — probability distributions over private hands — rather than states. Second, a subgame cannot be solved in isolation: how the opponent *arrived* at this river depends on what your strategy would do in rivers that never happened, so 'solve just this subgame optimally' can make your overall strategy MORE exploitable. This lane is the sequence of ideas that made search sound anyway. Burch et al. (2014) showed a subgame plus two small summaries — your own range, and the opponent's counterfactual values (what each of their hands was worth under the old strategy) — suffices to re-solve it safely. The intuition for SAFE RESOLVING: build a gadget where, at the subgame's entrance, each opponent hand may either play into your new strategy or take its old counterfactual value as a buyout. Solving that gadget forces your new strategy to concede no hand more than it already had — you can only improve. DeepStack (2017) made this practical by replacing 'solve to the end of the game' with a counterfactual value NETWORK evaluated at a depth limit, re-solving at every decision (continual resolving). Libratus (2017-18) got superhuman with zero neural nets: abstraction blueprint + nested safe subgame solving, re-solving finer trees as real bets arrive. Depth-limited solving (2018) made search cheap — at the depth limit the opponent picks among a few continuation strategies instead of requiring a value net — enabling Modicum on a 4-core laptop and Pluribus (2019) in 6-max. ReBeL (2020) unified everything as AlphaZero-style RL+search on public belief states with a Nash convergence proof, and Student of Games (2023) generalized to perfect AND imperfect information. GTO Wizard AI (ex-Ruse) is the commercial endpoint: street-by-street solving with value nets at street boundaries, self-play-trained across randomized stacks/rake/antes, ~3s per street, 19.4bb/100 over Slumbot (self-reported).

BUILDER'S OPTIONS, in ascending difficulty: (1) Blueprint only (the author's current Deep CFR plan) — amortized policy, milliseconds per query. (2) Blueprint + depth-limited resolve at the draw boundary, Modicum-style, using a few post-draw policies (e.g., the Deep CFR net plus perturbed variants) as continuation strategies — the cheapest search upgrade and my recommended 'phase 2'. (3) DeepStack-style continual resolving with a trained counterfactual value net. (4) ReBeL/SoG full RL+search. TRADEOFF VS AMORTIZED POLICY: search costs seconds per decision instead of a forward pass, but (a) accuracy — search corrects the net's errors at test time, while an average-policy net's errors are final; (b) flexibility — resolving natively handles bet sizes, stacks, and spots outside the training distribution, which is exactly why the commercial market went this way; (c) the value-net target (counterfactual values) is arguably easier to learn than a full policy. Cost: dramatically more engineering (belief/range tracking after every public action, gadget construction, nested solves).

DRAWMAHA-SPECIFIC WARNING: every neural system in this lane feeds range vectors into the net. Hold'em ranges are 1,326-dim; Drawmaha's are 2,598,960-dim, so DeepStack/ReBeL inputs do NOT port without aggressive hand bucketing or factorized belief representations — this is the single biggest open design problem for a Drawmaha search layer. Working in this lane's favor: the draw count (0-5) is public and slots perfectly into public-belief-state bookkeeping, pot-limit bounds bet-size branching, and one draw means only one street boundary needs a value function (which must output BOTH pot-halves for split-pot). PITFALLS: unsafe (unconstrained) resolving silently increases exploitability — always use the gadget; value-net error compounds across nested re-solves; validating by winrate against one bot instead of exploitability hides regressions.

VERIFICATION CAVEATS: all nine sources confirmed by fetching arXiv abstract pages, Science listings, or the GTO Wizard blog. Semantic Scholar rate-limited me (429s), so citation counts are omitted rather than guessed. Venue labels 'NIPS 2017 (best paper)', 'NeurIPS 2018', 'NeurIPS 2020', and 'AAAI 2014' come from background knowledge — the arXiv pages don't state them. GTO Wizard details (3s/street, 19.4bb/100, hundreds of millions of self-play hands) are the company's own claims with no peer-reviewed paper; Ruse's underlying method is unpublished, and the blog post's publication date was not verified. I did not find any public technical detail on GTO Wizard's network architecture (layers, input encoding) beyond 'neural networks supplying expected values at street boundaries'.

## Papers

### Solving Imperfect-Information Games Using Decomposition (2014)
_Neil Burch, Michael Johanson, Michael Bowling · AAAI 2014 (arXiv 1303.4441; venue from background knowledge, arXiv page does not state it)_
<https://arxiv.org/abs/1303.4441>

**Summary:** The theoretical foundation of the whole search lineage. Before this paper, nobody knew how to split an imperfect-information game into pieces, solve a piece on its own, and still guarantee the glued-together strategy is sound — because optimal play in a subgame depends on what happens elsewhere. Burch et al. give the first decomposition with optimality guarantees (CFR-D) and introduce subgame re-solving: reconstruct a subgame strategy at play time from two small summaries (your own range and the opponent's counterfactual values), with bounded whole-game error.

**Key Ideas:** A subgame plus two vectors — the player's reach probabilities and the opponent's counterfactual values at the subgame root — is enough to re-derive a safe strategy for that subgame. Re-solving uses a 'gadget' construction where the opponent can opt out of the subgame for their old counterfactual value, so the re-solved strategy can never hand them a new profitable deviation.

**Tradeoffs:** Pro: makes it possible to play games too large to store a full strategy for, with theoretical guarantees. Con: purely tabular in this paper and expensive; on its own it solved only small games — it needed DeepStack's neural value functions to scale.

**Relevance:** This is the concept the student should learn to understand everything downstream: why naive subgame solving is unsafe and what the counterfactual-value 'opt-out gadget' fixes. Even if the Drawmaha project stays on the Deep CFR path, the counterfactual-value summaries at street boundaries are the same objects a future search layer would need.

### DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker (2017)
_Matej Moravčík, Martin Schmid, Neil Burch, Viliam Lisý, Dustin Morrill, Nolan Bard, Trevor Davis, Kevin Waugh, Michael Johanson, Michael Bowling · Science (DOI aam6960); arXiv 1701.01724_
<https://arxiv.org/abs/1701.01724>

**Summary:** The first system to beat professional players at heads-up no-limit hold'em using 'continual resolving': instead of precomputing a strategy for the whole game, DeepStack re-solves a small lookahead tree at every decision, reasoning over ranges (probability distributions over hands) rather than single hands. Deep 'counterfactual value networks', trained by self-play on millions of solved situations, stand in for everything beyond the lookahead horizon. It beat professionals over 44,000 hands with statistical significance and produced strategies measurably harder to exploit than abstraction-based bots.

**Key Ideas:** Continual resolving = Burch-style safe re-solving applied at every decision point, carrying forward only (own range, opponent counterfactual values). Counterfactual value networks map (pot size, board, both players' range vectors) to a vector of counterfactual values per hand — a learned 'intuition' that terminates the search early, exactly like an evaluation function in chess but defined over ranges, not states.

**Tradeoffs:** Pro: no action abstraction of one's own tree, low exploitability, modest memory. Con: seconds of CFR per decision at play time; the value net input is a full range vector (1,326-dim in hold'em), which is what breaks when hand spaces get big; net errors compound across nested re-solves (the paper bounds this, but the bound depends on net quality).

**Relevance:** The canonical design if the author ever wants search in the Drawmaha dashboard. Direct porting is blocked by input size: a Drawmaha range is a distribution over 2,598,960 five-card hands, so a DeepStack-style range-vector input needs heavy hand bucketing or factorization first. The 'value net at a street boundary' idea, though, transfers cheaply: a net valuing the post-draw game lets any solver treat pre-draw as a one-street game.

### Safe and Nested Subgame Solving for Imperfect-Information Games (2017)
_Noam Brown, Tuomas Sandholm · NIPS 2017 (best paper award — venue/award from background knowledge; arXiv 1705.02955 confirmed)_
<https://arxiv.org/abs/1705.02955>

**Summary:** The algorithmic core of Libratus. It compares subgame-solving techniques (unsafe, re-solve, maxmargin, reach-based) and introduces two big upgrades: solving subgames in a way that exploits 'gifts' the opponent gave up earlier, and nested subgame solving — re-solving a fresh, finer-grained subgame every time the opponent takes an action, including bet sizes outside the precomputed abstraction. Nested solving replaced the older 'action translation' hack (rounding an off-tree bet to the nearest abstract one) and achieved far lower exploitability.

**Key Ideas:** Safe subgame solving constrains the new subgame strategy so the opponent's counterfactual value for entering it never rises above what the blueprint already conceded. Nested solving applies this recursively as real actions arrive, so the abstraction refines itself along the actual line of play rather than being fixed up front.

**Tradeoffs:** Pro: turns a coarse blueprint into a near-unexploitable live strategy; handles arbitrary opponent bet sizes cleanly. Con: needs a blueprint of decent quality to seed opponent counterfactual values, and each nested solve costs real compute at the table.

**Relevance:** For Drawmaha, nested solving is the principled answer to 'what if the opponent bets an amount my abstraction never considered' — pot-limit shrinks but does not eliminate that problem. It is also the cleanest paper for the survey to teach safe-resolving intuition from, since it explicitly contrasts unsafe vs safe variants and shows unsafe solving can backfire.

### Superhuman AI for Heads-Up No-Limit Poker: Libratus Beats Top Professionals (2018)
_Noam Brown, Tuomas Sandholm · Science 359(6374):418-424_
<https://www.science.org/doi/10.1126/science.aao1733>

**Summary:** The system paper for Libratus, which beat four top heads-up professionals over 120,000 hands in 2017 by a decisive margin. Libratus is a three-module architecture: (1) a blueprint strategy for the whole game computed with abstraction + Monte Carlo CFR on a supercomputer, (2) nested safe subgame solving that re-solves the current subgame in finer detail during play, and (3) a self-improver that patched the blueprint's action abstraction overnight wherever opponents found holes. Notably, Libratus used no neural networks at all.

**Key Ideas:** Blueprint-plus-resolving: play the cheap precomputed strategy early in the hand where the tree is huge, then switch to exact, finer-grained solving once the pot and remaining tree justify it. The overnight self-improvement loop shows abstraction holes — not opponent adaptation — were the main exploitable surface.

**Tradeoffs:** Pro: no function approximation means no net-error compounding; conceptually simple components. Con: enormous compute (supercomputer-scale blueprint and live solving), and the abstraction/blueprint machinery is game-specific engineering. Against DeepStack: heavier infrastructure, similar era, no learned generalization across situations.

**Relevance:** Mostly a conceptual landmark for the survey: proof that search, not deep learning, was the load-bearing ingredient in superhuman poker. For a solo Drawmaha project its compute model is out of reach, but its structure (cheap blueprint everywhere + expensive exact solving late in the hand, where Drawmaha's post-draw subgames are small) is a realistic pattern to copy at hobby scale.

### Depth-Limited Solving for Imperfect-Information Games (2018)
_Noam Brown, Tuomas Sandholm, Brandon Amos · NeurIPS 2018 (venue from background knowledge; arXiv 1805.08195 confirmed)_
<https://arxiv.org/abs/1805.08195>

**Summary:** Solves the problem that stopped Libratus-style solving from being cheap: you cannot truncate a search at a depth limit and slap a single value on the leaf, because in imperfect-information games a state's value depends on both players' strategies beyond it. The fix: at the depth limit, let the opponent choose among several precomputed 'continuation strategies', forcing the solution to be robust to different ways the rest of the game could be played. The resulting agent, Modicum, beat two prior top HUNL agents using a 4-core CPU and 16 GB of RAM — hardware that previously meant supercomputers.

**Key Ideas:** Replace a scalar leaf value with a small max over continuation policies for the opponent (a discrete, adversarial approximation of 'all the ways they could play from here'). This makes depth-limited lookahead sound-ish in imperfect information at a tiny fraction of full-subgame cost.

**Tradeoffs:** Pro: orders-of-magnitude compute reduction; the continuation-strategy set is easier to build than a range-conditioned value net. Con: guarantees degrade with a poor continuation set; still needs a blueprint to generate the continuations; less accurate than solving to the end of the game.

**Relevance:** Probably the most practically transplantable search idea for Drawmaha: solve pre-draw with a depth limit at the draw, where the 'continuation strategies' are a handful of post-draw policies (e.g., produced by the author's own Deep CFR net plus perturbations). It converts 'I need a perfect post-draw value net' into 'I need a few plausible post-draw policies', which is much easier.

### Superhuman AI for Multiplayer Poker (2019)
_Noam Brown, Tuomas Sandholm · Science (DOI 10.1126/science.aay2400)_
<https://www.science.org/doi/10.1126/science.aay2400>

**Summary:** Pluribus beat elite professionals at six-player no-limit hold'em — the first superhuman result outside two-player zero-sum, where Nash equilibrium loses its guarantees. It combined a self-play blueprint (MCCFR) with real-time depth-limited search, and famously cost under a normal workstation's worth of compute (trained on commodity hardware in days; the search runs in seconds per decision).

**Key Ideas:** Blueprint via self-play MCCFR plus depth-limited search with continuation strategies, extended to multiplayer. Pragmatic stance: abandon equilibrium guarantees (impossible in 6-max anyway) and validate empirically against top humans.

**Tradeoffs:** Pro: demonstrates the search recipe scales to multiplayer and to tiny budgets. Con: no theoretical guarantees in multiplayer; the empirical-only validation standard is expensive to replicate; less relevant machinery for strictly heads-up games.

**Relevance:** Heads-up Drawmaha keeps two-player zero-sum guarantees, so Pluribus matters less technically than its siblings — but it is the survey's best evidence that the blueprint+search pattern works on hobby-scale hardware, and its compute numbers are a useful sanity anchor for what a student project can afford.

### Combining Deep Reinforcement Learning and Search for Imperfect-Information Games (ReBeL) (2020)
_Noam Brown, Anton Bakhtin, Adam Lerer, Qucheng Gong · NeurIPS 2020 (venue from background knowledge; arXiv 2007.13544 confirmed)_
<https://arxiv.org/abs/2007.13544>

**Summary:** ReBeL is the 'AlphaZero of imperfect information': it recasts the game as a perfect-information game over public belief states (PBS) — the public observations plus both players' probability distributions over private hands — and then runs the familiar loop of self-play + search + value/policy networks on that transformed game. It provably converges to a Nash equilibrium in two-player zero-sum games, and reached superhuman HUNL performance with far less poker-specific domain knowledge than any predecessor (no hand abstractions handed to it).

**Key Ideas:** The PBS transformation: hidden information becomes part of a continuous, fully-observable state, so value functions over PBSs are well-defined. Search (CFR variants) runs on subgames rooted at PBSs; the value net is trained on the values those searches produce; sampled iteration counts during training make the value net robust at every stage of convergence.

**Tradeoffs:** Pro: cleanest theory in the lineage, minimal hand-crafted abstraction, one recipe for both training and play. Con: the PBS input embeds full range vectors, so input dimension scales with the number of private hands; training requires running search inside the training loop (expensive); much heavier engineering than Deep CFR.

**Relevance:** Conceptually the right lens for Drawmaha's draw round — the number of cards drawn is public and updates the public belief state very naturally. Practically the same warning as DeepStack: a raw PBS over 2.6M five-card hands is a ~5.2M-dim network input, so ReBeL-style training is out of reach without aggressive hand bucketing. Read it for the belief-state framing, not as a build target.

### Student of Games: A Unified Learning Algorithm for Both Perfect and Imperfect Information Games (2023)
_Martin Schmid, Matej Moravčík, Neil Burch, Rudolf Kadlec, Josh Davidson, Kevin Waugh, Nolan Bard, Finbarr Timbers, Marc Lanctot, G. Zacharias Holland, Elnaz Davoodi, Alden Christianson, Michael Bowling · Science Advances 9, eadg3256 (2023); arXiv 2112.03178_
<https://arxiv.org/abs/2112.03178>

**Summary:** The DeepStack team's culmination: one algorithm (growing-tree CFR with sound self-play, learned value-and-policy networks over public states) that plays chess, Go, heads-up no-limit hold'em, and Scotland Yard — spanning perfect and imperfect information. It beat the strongest openly available HUNL agent and state-of-the-art Scotland Yard agents, with theory showing convergence to optimal play as compute and network capacity grow.

**Key Ideas:** GT-CFR: a search that grows its lookahead tree non-uniformly (AlphaZero-style expansion) while running CFR updates for soundness in imperfect information; networks provide both counterfactual values and policy priors at the frontier; training data comes from the searches themselves ('sound self-play').

**Tradeoffs:** Pro: the most general and most modern design; subsumes DeepStack. Con: the most complex system in this lane by a wide margin (tree growth + CFR + dual-head nets + belief bookkeeping); compute-hungry; generality is wasted on a single fixed game like Drawmaha.

**Relevance:** For the survey it closes the arc — search+values went from poker-specific (DeepStack) to fully general (SoG). For the project it is a 'know it exists' citation, not a blueprint; every SoG component has a simpler ancestor that the author should implement instead.

### GTO Wizard AI Explained (commercial deployment; company blog) (2023)
_GTO Wizard (AI team founded by Philippe Beardsell and Marc-Antoine Provost, ex-Ruse AI, Mila) · GTO Wizard blog (self-published engineering post; publication date not verified)_
<https://blog.gtowizard.com/gto-wizard-ai-explained/>

**Summary:** The public description of the biggest commercial deployment of this lineage. GTO Wizard AI (built on the acquired Ruse engine) solves poker spots in real time rather than serving precomputed abstraction solutions: it solves one street at a time, with neural networks supplying expected values at the street boundary to 'eliminate the need for the computation of future streets'. Claimed results: solves up to 200bb deep with arbitrary bet sizes in ~3 seconds per street, trained by self-play over hundreds of millions of hands across randomized stacks/blinds/rake/antes, and beat Slumbot for 19.4bb/100 over 150,000 hands.

**Key Ideas:** Productionized depth-limited solving: street-by-street solves + a value net at the boundary, trained by large-scale self-play with counterfactual review of decisions. Training across randomized game configurations is what buys the 'any stack, any bet size, custom spots' capability that fixed presolved libraries cannot offer.

**Tradeoffs:** Pro: demonstrates the search+value-net architecture is the one that wins commercially when users demand custom configurations; latency (seconds) is acceptable for study tools. Con: proprietary — no paper, no reproducible details, benchmarks are self-reported marketing material; per-query cost far above a single policy-net forward pass.

**Relevance:** The clearest existence proof for the author's endgame: a study tool whose value comes from solving arbitrary user-specified spots. It also validates the exact hybrid worth stealing — the author's Deep CFR average-policy net answers instantly, and a street-limited resolve with a learned draw-boundary value function is the documented industrial path to higher accuracy.

