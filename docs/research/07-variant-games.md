# Research lane: variant-games

_Generated 2026-08-28 by a 9-agent literature sweep (Semantic Scholar / arXiv / web). Source material for the ML-survey paper._

## Lane narrative

THE STORY OF THIS LANE. Every solved variant inherits hold'em machinery by fixing one of two blow-ups. Blow-up 1 is private-state count: 1,326 hold'em combos → 270,725 PLO combos → 2,598,960 five-card Drawmaha hands. Blow-up 2 is tree shape: a draw round inserts a player-CHOSEN chance event that adds streets and emits a public signal (everyone sees how many cards you took; nobody sees which). Drawmaha has both, plus a split pot — which, per the Ho thesis, is the cheapest of the three problems: it changes only terminal utilities, so any CFR variant runs unmodified once you have a fast dual evaluator and payoff tables that encode scoop/split/quarter outcomes.

OPTIONS FOR THE HAND-SPACE BLOW-UP (PLO practice). (a) MonkerSolver: MCCFR + bucket abstraction — no preflop bucketing, postflop strength/texture/blocker buckets forced to share strategies. Tunable, battle-tested, supports split-pot O8; costs hours-to-days, big RAM, and bucketing artifacts (bluffs concentrated in few combos). (b) GTO Wizard AI: street-by-street depth-limited solving with self-play-trained EV networks replacing future streets; rivers solved exactly (Nash distance <0.1% pot), turn EV loss 0.14% pot, seconds per spot. Proprietary, but the architecture pattern is public and is the commercial cousin of the project's Deep CFR plan. (c) Vision GTO Trainer: precomputed abstracted sims served as a database/trainer — the "solve offline, serve online" delivery model the live-inference dashboard mirrors.

OPTIONS FOR THE DRAW ROUND. (a) Ostroumov's Draw Solver (2-7 triple draw; used in the 2015 Ivey match prep): handcraft preflop + first-draw strategy, solve every post-first-draw subtree independently, iterate the trunk by hand against computed EVs; ~1,000 CPU cores × 3 days per run in 2013-15; later versions solved from preflop and tracked ONE discarded card because discard blockers changed strategies materially. (b) Poker-CNN: skip equilibrium; learn draw and bet values by self-play CNN with the opponent's draw count as an input feature. Works, but admits "no internal consistency" — exploitable. (c) Zadeh 1977: pre-draw and post-draw bluffing frequencies computed analytically for five-card draw — free sanity checks for any toy-game CFR output.

PITFALLS. (1) Don't abstract away discard identity: the count is public but WHICH cards you kept/discarded drives blockers; Ostroumov's biggest win was remembering a single discard. (2) Infoset keys must include both players' full draw-count histories — it's public state, and snowing (standing pat on nothing) only emerges if the abstraction can represent it. (3) All-pairs matchup precomputation (Ho's recipe) squares badly from 4-card to 5-card hands — use suit isomorphism (Ho: 152,620 board classes) and lazy/sampled evaluation. (4) Commercial "solved" claims ship without exploitability numbers except GTO Wizard's self-reported benchmarks.

ABSENCE FINDING (load-bearing). I searched Semantic Scholar, arXiv, general web, GitHub, the commercial solver market, and Two Plus Two for Drawmaha / Sviten Special solver, bot, CFR, or equilibrium work: NOTHING exists. The game is documented (rules sites, SwC Poker runs it online, mixed-game strategy blogs), but no solver, no bot, no academic mention. Closest artifacts: Ostroumov's draw-game solvers (2-7 TD/SD, badugi — sold at $700-$13,900 tiers) and MonkerSolver's Omaha-8. The "first Drawmaha solver" claim is supported as of 2026-08.

WHAT I COULD NOT FIND / UNCERTAIN. No academic badugi work at all (only Ostroumov's commercial product). MonkerSolver and Ostroumov internals are undocumented beyond vendor pages and forums — the exact CFR variant and bucket counts are not public, and the Ostroumov 2+2 thread returned 403 so his algorithm details come only from his own retrospective. The GTO Wizard PLO blog page shows May 12, 2026, likely a republish of the ~2023 launch-era post (launch date unverified). Poker-CNN's AAAI-2016 venue is confirmed via the AAAI proceedings PDF; its human-match results were informal. From background knowledge (unverified in-session): no Annual Computer Poker Competition track ever ran a draw or split-pot game, which is why academic precedent is this thin. Downloaded source: /private/tmp/claude-501/-Users-alec/f59907ce-2eb6-48f7-aaa7-d0089c782c75/scratchpad/omaha_hilo.pdf (Ho thesis, methodology ch. 3 is the split-pot engineering recipe).

## Papers

### How I Made My First Million... Story of Draw Solver (2-7 Triple Draw / Single Draw / Badugi solver) (2025)
_Oleg Ostroumov (first-person account; mirrored by GipsyTeam) · Medium / GipsyTeam (practitioner engineering account; solver sold commercially at olegsolvers.com)_
<https://www.gipsyteam.com/news/02-02-2025/how-i-made-my-first-mil-and-helped-beat-phil-ivey-story-of-a-2-7-solver>

**Summary:** The only detailed public account of building an equilibrium solver for a real draw poker game (limit 2-7 triple draw, later single draw and badugi), developed 2013-2015 for high-stakes pros including Trueteller, and used in Kuznetsov's 2015 match prep against Phil Ivey ($400k won over 40,000 hands at $400/$800). The full game was too big to solve, so they handcrafted preflop and first-draw strategies, then solved each subtree from the second betting round independently (mimicking how hold'em solvers treat flops), manually iterating the preflop strategy against computed EVs; a later version solved from preflop. One full calculation took ~3 days on ~1,000 CPU cores (~$1,000/run), with ~20 iterations over development.

**Key Ideas:** Subgame decomposition for draw games: freeze early streets by hand, solve post-draw subtrees exactly, iterate. Critical upgrade: tracking one discarded card, because discard blockers proved strategically huge (unlocking lines like snowed T-high raises against one-card draws). Modern (2024) versions exploit better abstraction and large-RAM machines to reach 6-max.

**Tradeoffs:** Pros: makes an intractable draw game solvable with 2013-era hardware; decomposition parallelizes trivially. Cons: handcrafted trunk means no global equilibrium guarantee (early-street strategy is only locally improved); tracking discards blows up state (they capped at ONE remembered discard); algorithm is not disclosed (CFR-family implied), and no exploitability numbers are published.

**Relevance:** The closest existing artifact to a Drawmaha solver. Directly validates the project's plan of shrinking/decomposing before scaling, warns that the discard-identity information (what you threw away, not just how many) is too valuable to abstract away entirely, and gives a realistic compute anchor: a serious tabular draw-game solve was a cluster-scale job, which is exactly the gap Deep CFR is meant to close on a single machine.

### A No-Limit Omaha Hi-Lo Poker Jam/Fold Endgame Equilibrium (2015)
_Kenneth Ho (advisor: Eric Giesecke) · Master's thesis, Harvard Extension School (Harvard DASH repository)_
<https://dash.harvard.edu/entities/publication/73120378-e436-6bd4-e053-0100007fdf3b>

**Summary:** Computes an epsilon-Nash equilibrium for the jam/fold endgame of no-limit Omaha Hi-Lo — a four-card, SPLIT-POT game — using the then-new CFR+ algorithm on OpenCL/AWS, plus a linear-regression 'playability' heuristic that approximates the equilibrium from hand features. All C(52,4)=270,725 starting hands are enumerated via a combinatorial index; hi and lo hand ranks are bit-packed into a single 32-bit integer so one comparison settles both halves of the pot; and full win/quarter/tie/loss matchup distributions are pre-tabulated over suit-isomorphic board classes (152,620 types) before CFR ever runs.

**Key Ideas:** A split pot changes only the terminal utility function, never the information structure, so vanilla CFR+ applies unmodified — all the work moves into a fast dual (hi+lo) evaluator and precomputed payoff tables that record quartering/scooping outcomes. Suit-isomorphism and combinatorial indexing tame the brute-force cost (naively 3.4e17 evaluations, ~1 year at 1B/sec).

**Tradeoffs:** Pros: exact treatment of the split payoff including quarters; clean, reproducible engineering recipe; cloud+GPU made it a one-person project. Cons: only solves the jam/fold (two-action) endgame, not full betting; precomputing all-pairs matchup tables scales poorly as hand size grows (5-card hands square the table).

**Relevance:** The most direct split-pot precedent. For Drawmaha it prescribes the build order: write the dual evaluator (5-card-draw rank + Omaha rank), bit-pack both ranks for O(1) showdown comparison, put ALL scoop/split logic in the terminal payoff function, and exploit suit isomorphism aggressively — then the CFR/MCCFR/Deep CFR core needs zero modification for the split pot. Local copy of the PDF: /private/tmp/claude-501/-Users-alec/f59907ce-2eb6-48f7-aaa7-d0089c782c75/scratchpad/omaha_hilo.pdf

### Poker-CNN: A Pattern Learning Strategy for Making Draws and Bets in Poker Games (2015)
_Nikolai Yakovenko, Liangliang Cao, Colin Raffel, James Fan · arXiv:1509.06731; also published at AAAI 2016 (proceedings PDF at cdn.aaai.org confirms)_
<https://arxiv.org/abs/1509.06731>

**Summary:** The main academic work on learning to play a DRAW poker game: a convolutional network learns both the draw/discard decision and the betting decision for video poker, limit hold'em, and heads-up 2-7 triple draw, trained by iterative self-play 'bootstrapping' from a heuristic baseline with no game-specific solver. It uses a novel 2D tensor card representation extendable across variants, estimates a value for each action (pat / draw 1 / draw 2 / draw 3, bet/check/fold), and significantly beats the heuristic it was trained from. In the author's companion PokerNews writeup, the opponent's draw count is an explicit input, and he notes drawing fewer cards is an informational edge because opponents cannot distinguish which cards you kept.

**Key Ideas:** Treat poker as pattern recognition: shared card-tensor encoding + CNN value head per action; self-play data generation; the discard decision handled as just another action whose value is learned, with public draw counts fed in as features.

**Tradeoffs:** Pros: variant-agnostic, cheap, no game-theory machinery needed; handles the draw round natively. Cons: no equilibrium guarantee whatsoever — the author admits it 'lacks internal consistency' and 'does not try to solve heads-up triple draw'; results vs humans were informal; an eight-layer 2015 CNN, so architecture lessons are dated.

**Relevance:** Proof that a neural policy can learn draw/discard behavior from self-play in a game with the same public-draw-count signal Drawmaha has — but also a cautionary tale: without a regret-minimization backbone the result is exploitable. It is precisely the gap between Poker-CNN and Deep CFR that motivates the project's planned architecture (network inside CFR rather than network instead of CFR).

### How GTO Wizard Solved PLO (2026)
_GTO Wizard team (engineering blog) · GTO Wizard blog (page shows May 12, 2026; the PLO AI product it describes launched earlier — companion posts 'PLO Just Changed Forever', 'GTO Wizard AI Explained')_
<https://blog.gtowizard.com/how-gto-wizard-solved-plo/>

**Summary:** Vendor engineering account of the current state of the art for solving a 270,725-starting-hand game in real time: instead of Monker-style MCCFR over a bucketed full tree, GTO Wizard AI solves street-by-street, with neural networks (trained on self-play reinforcement-learning data) predicting hand EVs at the end of each street so future streets never need to be expanded. Rivers are solved exactly with no abstraction or NN, to Nash distance <0.1% of pot; turn benchmarks over 200 flop spots show average EV loss 0.14% of pot (90% of turns <0.33%). Any spot solves 'in a few seconds' vs hours/days for abstraction-based solvers.

**Key Ideas:** Depth-limited solving with a learned value function as the subtree stand-in (the commercial descendant of DeepStack/ReBeL ideas), plus exact tabular solving where the game is small enough (rivers). Abstraction-free solving avoids the classic artifact of bucketed solvers concentrating bluffs in a few combos.

**Tradeoffs:** Pros: near-exact accuracy, seconds not days, no bucketing artifacts, on-demand solving of arbitrary spots. Cons: entirely proprietary (no architecture, loss, or training details published); needs a large self-play RL pipeline to train the EV networks; accuracy claims are self-reported benchmarks.

**Relevance:** Shows the end-state the Drawmaha dashboard is aiming at: a value network that makes huge-hand-space spots solvable interactively. The river-exact + NN-elsewhere hybrid is a pattern the project can copy at small scale (solve the post-draw street exactly; let the network cover pre-draw streets), and its published Nash-distance/EV-loss benchmarks are the right evaluation vocabulary.

### MonkerSolver expanded syntax and custom abstraction (vendor documentation) + community documentation of its bucketing (2026)
_MonkerWare (vendor); Two Plus Two community threads · monkerware.com documentation (undated vendor page, accessed 2026-08; solver in wide PLO use since ~2017)_
<https://www.monkerware.com/syntax.html>

**Summary:** MonkerSolver is the long-standing industry-standard PLO/PLO5/Omaha-8 solver and the clearest documented example of abstraction-based solving of a 270k-hand game with MCCFR. Publicly documented behavior: preflop, no bucketing — every starting combo is distinct; postflop, strategically similar hands are merged into buckets (hand-strength tiers #n1-#nf, board-pairing/texture/flush classes, straight and blocker categories) that are forced to share one strategy, with the river called the street 'most suited to be abstracted' since it is the largest and has no draws. Also notable: it supports Omaha Hi-Lo (O8), making it the main commercial split-pot solver.

**Key Ideas:** Bucketing as the lever that turns an intractable game into a solvable one: strength + texture + blocker features define equivalence classes; users can define custom abstractions (including blocker-percentage syntax), so abstraction design is exposed as a first-class user decision.

**Tradeoffs:** Pros: solves full trees from preflop, battle-tested, supports split-pot O8; abstraction granularity is tunable against RAM/time. Cons: multi-hour-to-multi-day solves on expensive hardware; bucketed hands can't differentiate strategies within a bucket (bluff concentration, blocker blindness at coarse settings); internals never formally published — everything is vendor docs and forum archaeology.

**Relevance:** This is the 'MCCFR + abstraction on a shrunken game' rung of the project's own ladder, in production form. Its bucket vocabulary (strength percentile x draw/texture x blockers) is a ready-made feature list both for bucketing the shrunken Drawmaha variant and for the input features of the Deep CFR networks; its O8 support confirms split pots are routine for abstraction solvers.

### Computation of Optimal Poker Strategies (1977)
_Norman Zadeh · Operations Research, Vol. 25, No. 4, pp. 541-562_
<https://pubsonline.informs.org/doi/10.1287/opre.25.4.541>

**Summary:** The classical game-theoretic treatment of DRAW poker, four decades before CFR: analyzes an eight-person five-card draw game, deriving near-optimal strategies for opening, calling, and raising that incorporate pot odds and position, plus adaptive counter-strategies against known opponent habits. Crucially for this lane, it computes bluffing frequencies BEFORE the draw and explains known results about optimal bluffing and calling AFTER the draw — i.e., it treats the draw round and its information release as the central strategic object.

**Key Ideas:** Draw poker reduced to tractable stage games solved analytically: pre-draw ranges, the draw as an information event, and post-draw bluff/call ratios in the von Neumann/Kuhn tradition of indifference-based bluffing frequencies.

**Tradeoffs:** Pros: exact closed-form intuition (bluffing ratios, pat-hand credibility) that survives as sanity checks for any modern solver's output. Cons: heavy simplifying assumptions, single draw structure, no computers-scale search; 'optimal' here means within a restricted strategy family, not full-game Nash.

**Relevance:** Gives the Drawmaha project its oldest baseline intuitions to validate against: post-draw bluffing frequencies pinned to pot odds, and the pat/draw-count signaling logic. When tabular CFR on a toy draw game is running, its equilibria should reproduce Zadeh-style bluff ratios — a cheap correctness check.

### Extracting Learned Discard and Knocking Strategies from a Gin Rummy Bot (2021)
_Benjamin Goldstein, Jean-Pierre Astudillo Guerra, Emily Haigh, Bryan Cruz Ulloa, Jeremy Blum · AAAI Conference on Artificial Intelligence (EAAI-21, Gin Rummy Undergraduate Research Challenge track)_
<https://ojs.aaai.org/index.php/AAAI/article/view/17827>

**Summary:** From the AAAI/EAAI Gin Rummy challenge — the main academic venue where draw/discard games with PUBLIC discard information have been studied recently. The agent learns expected-utility values for each possible discard via machine learning and uses Monte Carlo CFR (MCCFR) to derive Nash-equilibrium knocking (endgame) strategies, then inspects the learned values to adjudicate conflicting human strategy-guide advice about which cards to throw.

**Key Ideas:** Hybrid design: learned value function for the combinatorial discard choice, exact MCCFR equilibrium for the small endgame decision — and the finding that discards leak exploitable information that both sides must reason about.

**Tradeoffs:** Pros: demonstrates MCCFR working in a draw/discard game and a practical split between learned and solved components; undergraduate-scale compute. Cons: gin rummy's discard is fully public (card identity revealed), unlike poker draws where only the COUNT is public, so information-hiding lessons transfer only partially; competition-grade rather than superhuman.

**Relevance:** The nearest academic ecosystem for 'discard actions as public signals.' Drawmaha sits between gin rummy (discard fully public) and standard poker (no discards): the number drawn is public, identities hidden. This paper's pattern — learn values for the big discard branch, solve the small betting branches exactly — is a plausible architecture for the shrunken-variant MCCFR stage.

