---
title: Changelog
description: What landed on main, day by day, newest first — read from the merge history of the repository.
---

Nearly every change reaches `main` through a pull request, and its subject states the finding. This page groups them by the day they merged and says in plain words what each day changed. Numbers in brackets are the pull requests on GitHub; the full history is `git log origin/main`.

## November 3, 2026

### Deep CFR — the ÷ pot net, run to the race's endpoint, flattens at 0.068

- **The 2,000-iteration run was made, on a Mac.** The 4 × 256 net with regret targets ÷ pot was run on from 2M to 20M hands per seat — 100.7 hours of training — and graded at 5M, 10M and 20M hands (#86).
- **The net has a floor and the table passes it.** The exact average went 0.071 → 0.068 → 0.068; the second 10M hands bought nothing (0.0679 → 0.0685). LCFR is 0.126 at 10M and 0.061 at 20M, so the table passes the net between the two.
- The homepage, Benchmarks and `/research`'s Figures 8, 9 and 13 now draw this arm against LCFR. Its best object is graded but not released.

## November 2, 2026

### Site — the card's rank and pip, settled

- The rank is 0.8 of the card's width on a 15 px floor, the oversized suit pip tucked into the corner so the rank clears it, and a ten is `T` on the static pages as in the apps (#85).

## October 30, 2026

### Site — one playing card everywhere: the bleed pip

- **The Play tab's card first** (#82): a small rank and one suit glyph too big for the card, cropped by its own edge, in four inks.
- **Then every card on the site** (#83): one drawing in the theme, each surface setting only its width.
- The rank grew from 0.36 to 0.44 of the card's width with a 12 px floor, because it read small beside the pip (#84).

## October 29, 2026

### Site — the site on paper

- **One theme for every surface** (#81): warm paper, near-black ink and one oxblood accent; Newsreader for display and research prose, Geist for the UI and every number. A product nav — Research, Solver, Benchmarks, Play, Rules — and declarative copy. The turning ASCII chip stays as the mark. Argued in `docs/superpowers/specs/2026-10-29-paper-lab-site-design.md` and enforced by `web/DESIGN.md`.

## October 28, 2026

### Site — `/rung3` becomes a research publication

- **A masthead and nine chapters** (#67), each heading stating its finding and each figure captioned with the file it reads; Deep CFR sits beside the tabular race, with the reach decomposition and a chapter on the √pot regret targets.
- **The contents rail** boxes the chapter you are reading and follows the scroll (#72), and names what each chapter is about in one line (#76).
- **Figures re-drawn to read along one axis**: the reach figures as columns with the leak per visit as a rising line (#74), chapter 05's equal-hands grid as a figure of two clocks on one ruler of hands (#79), and Figure 8's axes as a named domain a test holds every mark inside (#80).

### Play — the trainer deals against the best released Deep CFR

- **The Play tab's opponent is now rung 4's ÷ √pot exact average at 2M hands** (#69), released as `rung4-deep-pot0.5-t200`, its grade (0.086) quoted from the deal pack's manifest.
- The trainer reads in the Learn tab's register: lanes in a well, Learn's cards, the coach's mix as bands (#77).

### Solver — the hand so far is a path

- The steps of the hand are a path with the next decision forking off its tip, and the range grid's largest region fills its cell (#70).

### Repo — the README opens on the live site

- The front page opens on the live site and one results table; the top-level `research/` and `writeups/` folders fold into `docs/` (#73).
- `read_grade` refuses a malformed grade entry by name (#75); the chip mark's docstring and its tests were tightened (#71).

## October 26, 2026

### Solver — Deep CFR beside the table

- **`/solver` shows three strategies behind one switch** (#65): rung 3's tabular LCFR and rung 4's Π and exact average at 5M hands (`rung4-deep-t500`), each read from a release asset with its grade in the manifest.
- **`/rung3` gets a Learn tab** (#66): the mini-drawmaha range interface, imported from `solver-viz` rather than copied.

## October 25, 2026

### Deep CFR — the regret-rule screens and the sweep arms

- **DCFR and CFR+ lose to linear by 17×** under per-sample regrets, and get worse with training (#62).
- **The sweep arms are graded** (#63), the trainer gains `--lr` and `--save-every-s`, and two living notes start: the run log and why DCFR and CFR+ lost twice.
- Every grade quoted in the docs is now checked against its file by a test; 3 × 128's sweep grade is 0.243, not 0.244 (#64).

## October 24, 2026

### Deep CFR — the refit on Apple's GPU

- `--device mps` makes the refit 3.7× faster per seat, and the device can change on resume (#61).

## October 23, 2026

### Deep CFR — the Modal launcher and the price figure

- **One resumable trainer, a grader per grid mark** for the exact average, Π and the current iterate, plus the smoke run and the net-size sweep (#60).
- **The price figure**: Deep CFR's exploitability ÷ LCFR's, at equal hands per seat. The sweep picks 3 × 128 over 3 × 64.

## October 21, 2026

### Deep CFR — the average strategy

- **The exact running average**: own reach pushed through a predecessor table into a t-weighted sum, and **Π** fit on the policy memory, both frozen for the grader (#59).

### Repo — instructions at every subtree

- An `AGENTS.md` in every subtree, scripts grouped by what they serve, and mini-drawmaha split into `rules/`, `solver/` and `answer/` (#58).

## October 20, 2026

### Deep CFR — the loop

- **One iteration in the spec's order**, an exact resume, grid marks, and the oracle pin: with a perfect "net" the loop is LCFR bit for bit (#57).
- **The reservoir** (Algorithm R, exact resume) **and the collector**: K walks on W spawned workers, the same rows for any W (#56).

## October 19, 2026

### Deep CFR — the net, materialise and the refit

- The 3 × 64 net, materialising its outputs into the table's regret slots (Deep CFR's eq. 3 obtained by what is written), and the from-scratch refit (#55).

## October 9, 2026

### Deep CFR — the encoder

- **136 features per mini-drawmaha infoset**, and torch as an optional extra so the default suite still needs none (#52).
- The `/rung3` deal pack is checked against the grammar as it loads (#49).

## October 8, 2026

### Solver — `/solver` plays mini-drawmaha

- Reach-weighted ranges at every decision over the frozen LCFR strategy, exact to the export's 1/250 rounding (#54).

## October 7, 2026

### Mini-drawmaha — the race and the answer

- **Lockstep MCCFR on W workers**, the Modal launcher, and the four-rule race to 2M iterations (#45): LCFR wins at 0.061 chips/hand.
- **The frozen LCFR strategy** as a release asset, with `minidraw-play` and `minidraw-analysis` (#46).

### Site — `/rung3` and the UI standard

- **`/rung3`**: the research page and a browser trainer for mini-drawmaha over a pre-dealt pack (#48).
- **Every pressable answers a press**, chrome motion settles under 300 ms, and the UI standard becomes a document with a test (#47). The chip mark is redrawn to read as a chip (#51), and the headline's red squiggle grows with it (#50).

## October 1, 2026

- The rung-0 page's end label says whose numbers they are, and a re-export cannot silently revert hand edits (#44).

## September 30, 2026

### Mini-drawmaha — the exact grader and the packed table

- **The exact grader** (#36): a vector best response along the public tree, and the first checkpoint's number.
- **The infoset key keeps the board in the order it came** (#39), because the first round was played against the first card alone.
- **The packed table** (#40): four flat arrays with a ledger window per key, extra averaging rows and a discount stamp; about a microsecond per window.
- **The four regret rules and the averaging columns** (#41), each gated on Leduc against the LP value.

### Site — the pages share one theme

- The rung pages and the cover share the homepage's theme (#38); on the rung-0 page the settling dot rolls down a bowl and reaches Nash sooner (#42, #43).

## September 29, 2026

### Mini-drawmaha — the sampled learner

- **External-sampling MCCFR**, graded on Leduc first (#33).
- **The ledger table**: 3.1M spots, 1.77 GB, and a readout for one hand (#32). (The ordered-board key of #39 later doubled the rows to 6.2M.)

### Site — the cover splits into the homepage and `/rules`

- The cover becomes two static pages with a rules tab at `/rules` (#37), and a share page for the hero studies at `/hero` (#34).

## September 24, 2026

- **The deck gate** for mini-drawmaha: 1.57M infosets per player, inside the memory budget (#31).

## September 18, 2026

- **The mini-drawmaha game environment**: the deck, both scorers, the rules (#30).

## September 15–17, 2026

### Leduc — the `/rung2` dashboard

- **The action-timeline dashboard** over a committed solve (#24), and the likely-holding bar (#25).
- **Play the solver** heads-up on an oval table (#27); unreachable lines fall back to the deal (#26).
- **Edit any frequency** and read the response over its Nash baseline (#28).
- **The decision coach** (#29): a roll per decision, graded against the solver's mix.

## September 7–10, 2026

### Leduc — the second poker game, refereed by OpenSpiel

- The Leduc game environment (#18), then the solver: ledger table, CFR walk, exploitability (#22).
- **The referee's numbers are pinned** — OpenSpiel's node census, payoffs and LP game value, frozen in `tests/leduc/referee.json` — then the analysis and play CLIs (#23).

## September 4–5, 2026

### Deploy — the visualizers actually served

- **`"framework": null` pinned in `vercel.json`** (#15): without it Vercel detects Python and sends every path to a serverless function, so `/rung0` and `/rung1` never served. A test now fails the suite if the pin ever leaves (#16).
- The site is redesigned as a duotone with the games as its nav (#17).

## August 30 – September 2, 2026

### Kuhn — the first poker game

- **The game environment** (#8): frozen states, infosets that never carry the opponent's card.
- **The infoset table, and the ledger gains its two CFR weights**, keyword-only on purpose (#9).
- **Vanilla CFR** (#10), **the exact exploitability meter** over all 64 pure strategies (#11), the analysis pipeline, three figures and a play CLI (#12), and the `/rung1` visualizer (#13).
- **The exploit tab** (#14): lock a strategy and let CFR find the best response to it, served by a local endpoint.

## August 30, 2026

### Rock-paper-scissors — the first live demo

- The interactive regret-matching visualizer at `/rung0` (Vite + React + TypeScript), its layout stabilised so nothing shifts while the simulation runs (#3, #6).
- Vercel's build moves into a script to fit the 256-character `buildCommand` limit; fonts are self-hosted instead of loaded from a CDN.
- An automated Claude code review on every pull request — the repository's only CI workflow (#4).
- Figure scripts and chart UI cleaned up to the house style (#7).

## August 29, 2026

### Rock-paper-scissors — regret matching

- **The shared regret-matching ledger**, the RPS framework, the analysis and a play CLI (#1). The ledger measures regret against the strategy's expected utility — the same counterfactual form CFR uses at every infoset.
- **The research corpus**: an eight-lane literature sweep, a gaps audit, and the 26-page ML survey.

## August 28, 2026

### Repo — scaffolded: the rules and the plan to reach them

- Scaffolded: a README with the game's rules and the plan to build up to it through games with known answers.
