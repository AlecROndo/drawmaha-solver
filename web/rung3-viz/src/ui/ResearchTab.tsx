import {
  DEALS,
  DECISION_NODES,
  FINAL_ITERATION,
  FROZEN,
  FROZEN_FLOATS,
  FROZEN_MB,
  HALVINGS_TO_ZERO,
  MEDIAN_REVISIT,
  SIZES,
  UNREACHED_ROWS,
  WORKERS,
  DRAW,
  ROUND_ONE,
  perHundred,
  series,
} from '../research'
import { ColumnsChart } from './ColumnsChart'
import { RaceChart } from './RaceChart'
import { ScaleChart } from './ScaleChart'
import { SeatsChart } from './SeatsChart'
import { SpotBars } from './SpotBars'

/**
 * The research view: the figures, and between them the prose that says what
 * was built and what was found. Every number in the prose that could drift
 * is read from the data; the few that are facts about the code (the tree's
 * size, the revisit gap) are named constants in `research.ts`.
 */

/** A section of prose in the theme's hairline rows: a number, a heading that states the finding, the body. */
function Prose({ rows }: { rows: { n: string; g: React.ReactNode; p: React.ReactNode }[] }) {
  return (
    <table className="rows prose">
      <tbody>
        {rows.map((row) => (
          <tr key={row.n}>
            <td className="n">{row.n}</td>
            <td className="g">{row.g}</td>
            <td className="p">{row.p}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** A formula on its own line, in the mono voice. */
const Eq = ({ children }: { children: React.ReactNode }) => <code className="eq">{children}</code>

const M = (n: number) => `${(n / 1e6).toFixed(n % 1e6 ? 2 : 0)}M`
const final = (rule: 'vanilla' | 'lcfr' | 'cfrplus' | 'dcfr') => perHundred(series(rule).at(-1)!.exploitability)

export function ResearchTab() {
  const mini = SIZES[2].infosets
  const leduc = SIZES[1].infosets
  return (
    <>
      <div className="block">
        <RaceChart />
      </div>

      <Prose
        rows={[
          {
            n: '§1',
            g: 'The game: a bomb pot, one draw, and a pot that pays twice.',
            p: (
              <>
                Fifteen cards, ranks 2–6 in three suits. Both players ante 1 from 26-chip stacks; there are no blinds and
                no preflop betting. Three private cards each, a board card, a betting round. Then the draw: P0 may throw
                one card for a face-down replacement, then P1 may, knowing only <i>whether</i> P0 drew. A second board card,
                a second betting round, and the pot splits: half to the best three cards held (<b>inner</b>), half to the
                best two held plus both board cards (<b>outer</b>). Bets are pot-sized, so the 2-chip pot climbs 2 → 8 →
                all-in and no deeper. No engine plays this game, so there is no referee: the exact best-response walk is the
                ground truth every number on this page is graded by.
              </>
            ),
          },
          {
            n: '§2',
            g: `${(mini / leduc / 1000).toFixed(0)},000 times Leduc. The whole-tree walk does not run slowly; it does not run.`,
            p: (
              <>
                Suits carry information here — both halves of the pot have flushes — so the collapse rung 2 made (drop the
                suit) is not available. Instead the three suits are relabelled jointly across hole, discards and ordered
                board to the smallest image, which takes 455 dealt hands to 95 classes and leaves{' '}
                <b>{mini.toLocaleString()} information sets</b> against Leduc’s {leduc}. The tree beneath them has{' '}
                {(DECISION_NODES / 1e9).toFixed(1)} billion decision nodes and {DEALS.toLocaleString()} private deals at
                the root. Vanilla CFR walks the whole tree every iteration; that is what rungs 1 and 2 did, and what this rung
                cannot.
              </>
            ),
          },
          {
            n: '§3',
            g: 'External sampling: the deck and the opponent are sampled; your own actions are all walked.',
            p: (
              <>
                Each traversal deals one hand, follows one outcome at every chance node and one action at every opponent
                spot, and at the traverser’s own spots walks every legal action, because regret is a comparison and a
                comparison needs all of them. The reach weight rung 2 multiplied in by hand becomes the frequency the walk
                arrives at all, so the regret update at your spot is weighted exactly 1 and the strategy sum is banked at the
                opponent’s spots, where the walk sampled from σ anyway:
                <Eq>at my spot: R(I, ·) += u − ⟨σ, u⟩</Eq>
                <Eq>at their spot: S(I, ·) += t · σ(I, ·)</Eq>
                u[k] is what my k-th action earned down the sampled path; ⟨σ, u⟩ is what my current mix earned on it; the
                difference is one unbiased sample of counterfactual regret. Multiplying the reach in on top applies the
                probability twice, and on Leduc that walk stalls at 0.14–0.20 where this one reaches 0.05.
              </>
            ),
          },
          {
            n: '§4',
            g: `Lockstep: ${WORKERS} hands per seat per iteration, every touched row banked once.`,
            p: (
              <>
                One hand per iteration is too slow for a 6.2M-row table, and W processes writing into one table would race.
                So an iteration walks {WORKERS} hands per seat against the <i>same</i> frozen σₜ and writes nothing while it
                walks; once all {WORKERS} are in, every touched row is banked once with the sum of the workers’ regrets, in a
                fixed order, so the float sums are the same however the walks were scheduled. The randomness is counter-based
                — worker w, iteration t, seat s — so a run carries no RNG state and resuming a checkpoint equals never having
                stopped. The race below is {M(FINAL_ITERATION)} iterations per rule, {M(FINAL_ITERATION * WORKERS)} sampled
                hands per seat, on Modal.
              </>
            ),
          },
        ]}
      />

      <div className="panels block">
        <ColumnsChart />
        <SeatsChart />
      </div>

      <Prose
        rows={[
          {
            n: '§5',
            g: 'Four regret rules; one walk. What happens to r is the only thing that differs.',
            p: (
              <>
                Each visit measures one sample r of counterfactual regret. The rule is what the ledger does with it:
                <Eq>vanilla: R ← R + r</Eq>
                <Eq>LCFR: R ← R + t · r</Eq>
                <Eq>CFR+: R ← max(R + r, 0)</Eq>
                <Eq>DCFR: R ← R + r, then R⁺ × t^1.5 / (t^1.5 + 1) and R⁻ × ½</Eq>
                Vanilla counts every iteration the same. LCFR counts iteration t as t iterations, so the early ones — played
                from strategies that were mostly noise — fade; after T iterations the first tenth carries 1% of the weight.
                CFR+ floors negative regret at zero, so an action that was bad early and good now is played the moment it is
                good. DCFR is the soft version: positive regret fades fast early and hardly at all later, and negative regret
                is halved every iteration. The average is not the rule’s business: every rule banks the same linear (weight t)
                average, and the uniform and quadratic columns of Fig. 2 beside it.
              </>
            ),
          },
          {
            n: '§6',
            g: `Under sampling, CFR+ and DCFR become one rule: ${final('cfrplus').toFixed(1)} against ${final('dcfr').toFixed(1)}.`,
            p: (
              <>
                DCFR is written to discount every row every iteration; touching 6.2M rows per iteration is out of the
                question, so a row is discounted lazily, all the owed iterations at once when it is next visited. That is
                exact, not an approximation: an unvisited row receives r = 0 and the per-sign factors simply multiply. It also
                explains the race. At the end of the run the median row was last visited about{' '}
                {MEDIAN_REVISIT.toLocaleString()} iterations earlier, and halving a float64 {HALVINGS_TO_ZERO.toLocaleString()}{' '}
                times takes it to exactly zero — so by the time a row comes round again its negative regret is gone, which is
                CFR+’s floor. The two are still not identical (DCFR also discounts positive regret), so the curves lie close
                rather than on top of each other. Both lose to the two rules that keep negative regret by 3–4×:{' '}
                {final('lcfr').toFixed(1)} and {final('vanilla').toFixed(1)} against {final('cfrplus').toFixed(1)} and{' '}
                {final('dcfr').toFixed(1)} chips per hundred hands. Brown & Sandholm reported that the floor interacts badly
                with sampling; this is that interaction, measured.
              </>
            ),
          },
          {
            n: '§7',
            g: `The answer: LCFR’s linear average at ${M(FINAL_ITERATION * WORKERS)} hands per seat, ${FROZEN.exploitability.toFixed(6)} chips a hand.`,
            p: (
              <>
                The winner is frozen as rung 3’s solution: {FROZEN_FLOATS.toLocaleString()} float32 probabilities,{' '}
                {FROZEN_MB} MB compressed, published as a release asset and checked against a pinned SHA-256 on load.{' '}
                {UNREACHED_ROWS.toLocaleString()} of the {mini.toLocaleString()} rows were never reached and play uniformly.
                Graded, BR₀ is {FROZEN.br0.toFixed(6)} and BR₁ is +{FROZEN.br1.toFixed(6)}: a perfect P0 still loses, because
                P1 acts last in both rounds and at the draw and the seat is worth {(-FROZEN.value_p0).toFixed(4)} a hand to P1
                in self-play (Fig. 3). For scale, uniform random play is exploitable for 484 chips per hundred hands.
              </>
            ),
          },
        ]}
      />

      <div className="block">
        <SpotBars
          n="04"
          k="Fig. 4 · round 1 by inner category"
          title="P0 leads with every category and slowplays trips; P1 folds a pair more often than a high card."
          say={
            <>
              The frozen strategy’s round-1 mix, averaged over every physical deal of the category with equal weight. What
              it does <b>holding</b> that hand, not how often the line happens. Categories commonest first.
            </>
          }
          label="Round-one betting by inner hand category"
          figures={ROUND_ONE}
        />
      </div>

      <div className="block">
        <SpotBars
          n="05"
          k="Fig. 5 · the draw by inner category"
          title="High cards and pairs throw a card; flushes and better almost never do."
          say={
            <>
              Which of the three canonical positions (low, mid, top by rank) the strategy throws, by category. P1 throws
              more often after P0 stands pat: with a straight,{' '}
              {((1 - DRAW[1].rows[2].mix[0]) * 100).toFixed(0)}% against {((1 - DRAW[2].rows[2].mix[0]) * 100).toFixed(0)}%
              after P0 drew.
            </>
          }
          label="The draw by inner hand category"
          figures={DRAW}
        />
      </div>

      <div className="panels block">
        <ScaleChart />
        <section className="panel" aria-label="What this page does not claim">
          <div className="titlebar">
            <span className="dots">
              <i />
              <i />
              <i />
            </span>
            <span className="k">caveats · what is and is not settled</span>
            <span className="no">—</span>
          </div>
          <div className="pane">
            <h3>One seed. Readings, not reasons.</h3>
            <ul className="fields caveats">
              <li>
                <span>the race</span>
                <span>
                  one run per rule, seed 0 — {final('lcfr').toFixed(1)} against {final('vanilla').toFixed(1)} between LCFR
                  and vanilla is suggestive, not settled
                </span>
              </li>
              <li>
                <span>figs. 4–5</span>
                <span>
                  a category is the three held cards alone; its deals meet different boards, so a gap between two categories
                  can come from that mix
                </span>
              </li>
              <li>
                <span>the grader</span>
                <span>
                  exact, by a vector best response along the public tree; it imports the game and not the learner, so it
                  cannot share a bug with what it grades
                </span>
              </li>
              <li>
                <span>the trainer</span>
                <span>
                  plays a pack of 600 pre-dealt hands with the frozen mix at every node; the browser reimplements the betting
                  grammar and nothing about the key
                </span>
              </li>
              <li>
                <span>next</span>
                <span>
                  rung 4, Deep CFR — its first target is this game, where this answer and the exact grader measure what the
                  networks give up
                </span>
              </li>
            </ul>
          </div>
        </section>
      </div>

      <p className="foot">
        Every number on this page is read from <code>figures/rung3/grades.json</code> (the 42 graded checkpoints of the
        race, plus uniform random) and <code>figures/rung3/answer_sheet.json</code>, both written by{' '}
        <code>uv run minidraw-analysis</code>. The grader is <code>src/drawmaha_solver/minidrawmaha/exploitability.py</code>;
        the learner is <code>mccfr.py</code> under <code>lockstep.py</code>, with the rules in <code>regret_rules.py</code>.
        Nothing here was typed in by hand.
      </p>
    </>
  )
}
