# RPS Regret-Matching Visualizer

Interactive teaching instrument for the rung-0 algorithm: regret matching
learning rock-paper-scissors live. Step mode shows the arithmetic of a single
update; run mode drives tens of thousands of iterations per second with
convergence charts and a ternary simplex plot (current strategy orbits forever,
average strategy spirals into ⅓⅓⅓).

The simulation matches the repo's reference implementation exactly: the
expected-utility regret form R += u − ⟨σ,u⟩ (not the textbook sampled-action
variant), verified by unit tests against the committed acceptance numbers
(self-play exploitability ≤ 0.005 at 100k iterations; vs a 50%-rock opponent
the average converges to pure paper at ≈ +0.25/round).

## Structure

```
src/sim/   pure, framework-free simulation (seeded RNG, payoffs,
           regret ledger, engine with decimated history) + unit tests
src/ui/    React panels: ledger, update trace, convergence chart,
           exploitability chart (log-log, 1/√T reference), simplex
           canvas, scoreboard, controls
```

## Commands

```
npm install
npm test        # vitest — includes the 100k-iteration acceptance runs
npm run dev     # local dev server
npm run build   # static bundle in dist/ — 100% client-side, no server code
```

## Deploy

Static Vite site. On Vercel: set the project root directory to
`web/rung0-viz`, framework preset Vite, output `dist`. No server config.
