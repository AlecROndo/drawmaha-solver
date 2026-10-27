import { FORENSICS, HALVINGS_TO_ZERO, VISIT_CLOCK, globalDcfrAt, positiveFactor, visitClockAt } from '../clock'
import {
  ARMS,
  BEST,
  KEY_NUMBERS,
  PI_REFERENCE,
  PI_SWEEP,
  REFIT_ARMS,
  RULE_SCREEN,
  SIGN_FLIPS,
  SIZE_SWEEP,
  SWEEP_RULE,
  lcfrAt,
  markOf,
  piArm,
  reachWeighted,
} from '../deep'
import {
  DEALS,
  DECISION_NODES,
  DRAW,
  FINAL_ITERATION,
  FROZEN,
  FROZEN_FLOATS,
  FROZEN_MB,
  ROUND_ONE,
  SIZES,
  UNIFORM_RANDOM,
  WORKERS,
  perHundred,
  series,
  type Rule,
} from '../research'
import { AveragesChart } from './AveragesChart'
import { ColumnsChart } from './ColumnsChart'
import { DeepRaceChart } from './DeepRaceChart'
import { Figure } from './Figure'
import { fraction } from './scales'
import { LossShapes } from './LossShapes'
import { PiLossChart } from './PiLossChart'
import { RaceChart } from './RaceChart'
import { RefitChart } from './RefitChart'
import { RuleScreenChart } from './RuleScreenChart'
import { ScaleChart } from './ScaleChart'
import { SeatsChart } from './SeatsChart'
import { SizeChart } from './SizeChart'
import { SpotBars } from './SpotBars'

/**
 * The research, as a publication: the key numbers, a contents rail, ten
 * chapters and the methods. Each chapter goes top-down — the problem and
 * the big picture, then the anatomy of the formula, why it has that shape,
 * and what moves when each piece moves — and every heading states its
 * finding. Every number in the prose is read from `research.ts`, `deep.ts`
 * or `clock.ts`, which read the copies of the committed grade files under
 * `src/data/`; the few facts about the code (the tree's size, the deal
 * count) are named constants there. Nothing is typed in.
 */

/** A formula on its own line, in the mono voice. */
const Eq = ({ children }: { children: React.ReactNode }) => <code className="eq">{children}</code>

const CHAPTERS = [
  { id: 'ch-01', n: '01', short: 'The game and the grader' },
  { id: 'ch-02', n: '02', short: 'A tree no walk can cross' },
  { id: 'ch-03', n: '03', short: 'Four rules, one walk' },
  { id: 'ch-04', n: '04', short: 'Why DCFR became CFR+' },
  { id: 'ch-05', n: '05', short: 'What the answer does' },
  { id: 'ch-06', n: '06', short: 'A net for the ledger' },
  { id: 'ch-07', n: '07', short: 'Width, depth, steps, rate' },
  { id: 'ch-08', n: '08', short: 'Three averages, one loss' },
  { id: 'ch-09', n: '09', short: 'Rules under a net' },
  { id: 'ch-10', n: '10', short: 'What is not settled' },
  { id: 'methods', n: '', short: 'Methods and footnotes' },
] as const

function Chapter({ id, title, children }: { id: string; title: React.ReactNode; children: React.ReactNode }) {
  const entry = CHAPTERS.find((c) => c.id === id)!
  return (
    <section className="chapter" id={id}>
      <p className="label">Chapter {entry.n}</p>
      <h2>{title}</h2>
      {children}
    </section>
  )
}

const final = (rule: Rule) => perHundred(series(rule).at(-1)!.exploitability)
const p100 = (chipsPerHand: number, digits = 1) => perHundred(chipsPerHand).toFixed(digits)
const pct = (x: number) => `${Math.round(x * 100)}%`
const cut = (from: number, to: number) => `${Math.round((1 - to / from) * 100)}%`
const M = (n: number) => `${(n / 1e6).toFixed(n % 1e6 ? 1 : 0)}M`

export function ResearchTab() {
  const mini = SIZES[2].infosets
  const leduc = SIZES[1].infosets
  const hands = FINAL_ITERATION * WORKERS
  const best100 = markOf(BEST, 100)
  const best200 = markOf(BEST, 200)
  const best500 = markOf(BEST, 500)
  const size = (net: string) => SIZE_SWEEP.find((s) => s.net === net)!
  const refit = (net: string, steps: number) => REFIT_ARMS.find((a) => a.net === net && a.steps === steps)!
  const screen = (rule: 'linear' | 'dcfr' | 'cfrplus', t: number) => RULE_SCREEN[rule].find((p) => p.t === t)!.exact
  const flip = (rule: 'dcfr' | 'cfrplus', t: number) => SIGN_FLIPS[rule].find((f) => f.t === t && f.seat === 0)!
  const ce64 = piArm('ce-softmax-64k-cos')
  const mse64 = piArm('mse-linear-64k-cos')
  const ce16 = piArm('ce-softmax-16k')
  const control = piArm('mse-linear-16k')
  const widest = piArm('ce-softmax-1024x4-64k-cos')
  const deeper = piArm('ce-softmax-256x6-64k-cos')
  const arms500 = PI_SWEEP.filter((a) => a.t === 500)
  const neverVisited = FORENSICS.never_visited_by_2M * FORENSICS.rows
  const lateRows = 1 - FORENSICS.visited_by_10k
  const visit2M = VISIT_CLOCK.filter((v) => v.hands === 2_000_000)
  const small = ARMS['mac-3x64']

  return (
    <div className="article">
      <dl className="keynums" aria-label="Key numbers">
        <div>
          <b>
            {final('lcfr').toFixed(1)} <small>vs</small> {final('cfrplus').toFixed(1)}
          </b>
          <span>chips per 100 hands a perfect adversary wins: LCFR against CFR+, after {M(hands)} hands per seat</span>
        </div>
        <div>
          <b>
            {KEY_NUMBERS.priceAt1M.toFixed(2)}
            <small>×</small>
          </b>
          <span>the net's exploitability over the table's at one million hands per seat, about {fraction(KEY_NUMBERS.priceAt1M)}; same walk, same hands</span>
        </div>
        <div>
          <b>
            {p100(KEY_NUMBERS.netAt5M)} <small>vs</small> {p100(KEY_NUMBERS.tableAt5M)}
          </b>
          <span>net against table at five million hands; the net has flattened, the table has not</span>
        </div>
        <div>
          <b>
            {KEY_NUMBERS.ruleScreenRatio.toFixed(0)}
            <small>×</small>
          </b>
          <span>worse than linear weighting when DCFR is applied to each regret sample, and rising with training</span>
        </div>
      </dl>

      <nav className="contents" aria-label="Contents">
        <span className="k">Contents</span>
        <ol>
          {CHAPTERS.map((c) => (
            <li key={c.id}>
              <a href={`#${c.id}`}>
                {c.n && <i>{c.n}</i>}
                {c.short}
              </a>
            </li>
          ))}
        </ol>
      </nav>

      <div className="chapters">
        {/* ------------------------------------------------ Part I: the game */}
        <Chapter id="ch-01" title="A bomb pot, one draw, and a pot that pays twice, graded by an exact best response">
          <p>
            Fifteen cards, ranks 2–6 in three suits. Both players ante 1 from 26-chip stacks; there are no blinds and no
            preflop betting. Three private cards each, a board card, a betting round. Then the draw: P0 may throw one card
            for a face-down replacement, then P1 may, knowing only <i>whether</i> P0 drew. A second board card, a second
            betting round, and the pot splits: half to the best three cards held (<b>inner</b>), half to the best two held
            plus both board cards (<b>outer</b>). Bets are pot-sized, so the 2-chip pot climbs 2 → 8 → all-in and no
            deeper.
          </p>
          <p>
            Suits carry information here — both halves of the pot have flushes — so the collapse rung 2 made (drop the
            suit) is not available. Instead the three suits are relabelled jointly across hole, discards and ordered board
            to the smallest image, which leaves <b>{mini.toLocaleString()} information sets</b>,{' '}
            {Math.round(mini / leduc).toLocaleString()} times Leduc's {leduc}. The tree beneath them has{' '}
            {(DECISION_NODES / 1e9).toFixed(1)} billion decision nodes and {DEALS.toLocaleString()} private deals at the
            root.
          </p>
          <Figure
            n={1}
            caption={
              <>
                A table {Math.round(mini / leduc).toLocaleString()} times Leduc's, and a tree no walk could cross. Log
                axis; the last bar is how many hands the race sampled against it.
              </>
            }
            source="research.ts: information sets of each rung, the tree's node count and the deal count are facts about the code; the hands sampled are figures/rung3/grades.json"
            label="The size of each rung's game on a log axis"
          >
            <ScaleChart />
          </Figure>
          <h3>No engine plays this game, so the grader is the ground truth</h3>
          <p>
            Every number on this page is an exploitability: the mean of what a perfect P0 wins against the strategy's P1
            and what a perfect P1 wins against its P0, in chips per hundred hands. The walk that computes it sums over the
            hidden worlds first and takes the responder's maximum second, because a maximum inside the sum would be an
            adversary who sees face-down cards. It imports the game's rules and nothing from any learner, so a bug shared
            between a solver and its grader cannot cancel and certify what broke it. For scale, uniform random play is
            exploitable for {p100(UNIFORM_RANDOM.exploitability, 0)} chips per hundred hands.
          </p>
        </Chapter>

        <Chapter id="ch-02" title="A tree no walk can cross is sampled: the deck and the opponent are drawn, your own actions are all walked">
          <p>
            Vanilla CFR walks the whole tree every iteration; that is what rungs 1 and 2 did, and what this rung cannot.
            External sampling walks one hand at a time: it deals one hand, follows one outcome at every chance node and
            one action at every opponent spot, and at the traverser's own spots walks every legal action, because regret
            is a comparison and a comparison needs all of them.
          </p>
          <p>
            At one of your spots the walk measures one sample of counterfactual regret per action. u[k] is what your
            k-th action earned down the sampled path; ⟨σ, u⟩ is what your current mix earned on it; the difference is
            how much better action k would have done than what you actually do:
          </p>
          <Eq>at my spot: R(I, ·) += u − ⟨σ, u⟩</Eq>
          <Eq>at their spot: S(I, ·) += t · σ(I, ·)</Eq>
          <p>
            The reach weight rung 2 multiplied in by hand becomes the frequency the walk arrives at all, so the regret
            update at your spot is weighted exactly 1, and the strategy sum is banked at the opponent's spots, where the
            walk sampled from σ anyway. Multiplying the reach in on top would apply the probability twice. What moves
            what: a spot play reaches often is sampled often and its regrets settle fast; a spot reached once in a
            thousand hands collects one noisy sample per thousand hands, which is the fact the whole clock chapter turns
            on.
          </p>
          <h3>Lockstep: {WORKERS} hands per seat per iteration, every touched row banked once</h3>
          <p>
            One hand per iteration is too slow for a {(mini / 1e6).toFixed(1)}M-row table, and W processes writing into
            one table would race. So an iteration walks {WORKERS} hands per seat against the <i>same</i> frozen σₜ and
            writes nothing while it walks; once all {WORKERS} are in, every touched row is banked once with the sum of the
            workers' regrets, in a fixed order, so the float sums are the same however the walks were scheduled. The
            randomness is counter-based — worker w, iteration t, seat s — so a run carries no RNG state and resuming a
            checkpoint equals never having stopped. The race that follows is {M(FINAL_ITERATION)} iterations per rule,{' '}
            {M(hands)} sampled hands per seat.
          </p>
        </Chapter>

        {/* -------------------------------------------- Part II: the rules race */}
        <Chapter id="ch-03" title="Keeping negative regret wins, by three to four times">
          <p>
            Each visit measures one sample r of counterfactual regret. The four rules differ only in what the ledger does
            with it:
          </p>
          <Eq>vanilla: R ← R + r</Eq>
          <Eq>LCFR: R ← R + t · r</Eq>
          <Eq>CFR+: R ← max(R + r, 0)</Eq>
          <Eq>DCFR: R ← R + r, then R⁺ × t^1.5 / (t^1.5 + 1) and R⁻ × ½</Eq>
          <p>
            Vanilla counts every iteration the same. LCFR counts iteration t as t iterations, so the early ones — played
            from strategies that were mostly noise — fade; after T iterations the first tenth carries one percent of the
            weight. CFR+ floors negative regret at zero, so an action that was bad early and good now is played the
            moment it is good. DCFR is the soft version: positive regret fades fast early and hardly at all later, and
            negative regret is halved every iteration. Picture one row visited every iteration: all four do what their
            papers say. Picture a row visited once in a thousand iterations: vanilla and LCFR add the sample and wait;
            CFR+ floors the total it has; DCFR halves the negative total a thousand times before the next sample arrives.
            That asymmetry is the whole of chapter 04.
          </p>
          <Figure
            n={2}
            caption={
              <>
                Keeping negative regret wins, by three to four times: {final('lcfr').toFixed(1)} and{' '}
                {final('vanilla').toFixed(1)} against {final('cfrplus').toFixed(1)} and {final('dcfr').toFixed(1)} chips per
                hundred hands after {M(hands)} hands per seat, and CFR+ and DCFR lie on top of each other the whole way.
                Hover or arrow a checkpoint to read the column.
              </>
            }
            source="figures/rung3/grades.json · the linear average of each rule at eight checkpoints, by the exact best-response walk; both axes log; seed 0, one run per rule"
            label="Exploitability of the four regret rules over the race"
          >
            <RaceChart />
          </Figure>
          <p>
            The average is not the rule's business: every rule banks the same linear (weight t) average, and a run may
            bank further averages of the very same play at other weights for free, since nothing reads an average while
            training. Linear beats uniform under both rules that bank it, so the rule and not the average is what
            decides.
          </p>
          <Figure
            n={3}
            caption={
              <>
                Linear averaging beats uniform for vanilla and for CFR+; the rule is still what decides. Each bar is one
                average of the same run at {M(hands)} hands per seat.
              </>
            }
            source="figures/rung3/grades.json · the extra averaging columns at the race's last checkpoint"
            label="Exploitability of each averaging column at the end of the race"
          >
            <ColumnsChart />
          </Figure>
        </Chapter>

        <Chapter id="ch-04" title="Under sampling, DCFR became CFR+: its clock counts iterations nobody sampled">
          <p>
            DCFR is written to discount every row every iteration. Touching {(mini / 1e6).toFixed(1)}M rows per iteration
            is out of the question, so a row is discounted lazily, all the owed iterations at once when it is next
            visited. That is exact, not an approximation: an unvisited row receives r = 0 and the per-sign factors simply
            multiply. It also explains the race. A row visited once every g iterations pays g iterations of discount
            between two pieces of evidence, so its negative regret at the next visit is (½)^g of what was banked.
          </p>
          <p>
            How idle were the rows? The race's own checkpoints say. At {M(FINAL_ITERATION)} iterations the median row had
            last been visited {Math.round(FORENSICS.age_quantiles_iters.p50).toLocaleString()} iterations earlier;
            weighting each row by how often play actually arrives there, the median gap is still{' '}
            <b>{FORENSICS.age_quantiles_reach_weighted.p50.toLocaleString()} iterations</b>. Halving a float64{' '}
            {HALVINGS_TO_ZERO.toLocaleString()} times takes it to exactly zero, and{' '}
            {pct(FORENSICS.age_ge_1075)} of the rows ever visited were older than that. So by the time a row came round
            again its negative regret was gone, which is CFR+'s floor.
          </p>
          <p>
            The positive side failed the other way. DCFR's factor t^1.5 / (t^1.5 + 1) does its work in the first few
            dozen global ticks: by tick 10,000 it is {positiveFactor(10_000).toFixed(6)}. On the global clock those ticks
            passed before most rows were ever visited — <b>{pct(lateRows)} of rows</b> were first reached after iteration
            10,000 — so for them positive regret was never discounted at all. In the limit of rare visits, DCFR(α, β, γ)
            on the global clock is DCFR(∞, −∞, γ), which its own paper identifies with CFR+. The race used the same
            linear average for both, so the two runs were one algorithm up to the halving that did not finish, and they
            finished within {Math.abs(1 - final('dcfr') / final('cfrplus')).toFixed(3).replace(/^0\./, '0.')} of each other
            in relative terms: {final('dcfr').toFixed(1)} against {final('cfrplus').toFixed(1)}.
          </p>
          <p>
            Why LCFR is immune: its weight w(t) = t is <b>sign-blind</b>, so idle time cannot shift the balance between
            an action's good and bad history, and <b>scale-free</b>, so a rare row's updates, arriving at t ≈ j/p, keep the
            ratio t_i / t_j ≈ i / j whatever p is. A rare row gets the same linear-in-evidence weighting as a frequent
            one. A discount survives sampling exactly when it has both properties; DCFR's has neither, and CFR+'s floor
            is sign-dependent but has no clock, so under sampling CFR+ is still CFR+.
          </p>
          <p>
            The prediction that follows is that DCFR clocked on the row's own updates recovers some of its ground, and a
            first arm says so: with the stamp counting the row's updates rather than the global iteration, a run to
            {' '}{M(visit2M[0].hands)} hands per seat (W = {visit2M[0].workers}) graded{' '}
            {p100(visitClockAt(2_000_000, 'linear'))} with the linear average and {p100(visitClockAt(2_000_000, 'quadratic'))}{' '}
            with the quadratic, against the race's global-clock DCFR at {p100(globalDcfrAt(2_000_000))} and LCFR at{' '}
            {p100(lcfrAt(2_000_000))} at the same hands. Better than the wrong clock, still behind the sign-blind rule,
            and one seed at a different worker count, so a reading rather than a result.
          </p>
          <p className="callout">
            The proof of the limit, the forensics on the four checkpoints and a simulation where only the clock changes
            are in the note <code>docs/notes/regret-rules-dcfr-cfrplus.md</code>; chapter 09 is its second half.
          </p>
        </Chapter>

        <Chapter id="ch-05" title={<>The answer: LCFR's linear average at {M(hands)} hands, {FROZEN.exploitability.toFixed(4)} chips a hand</>}>
          <p>
            The winner is frozen as rung 3's solution: {FROZEN_FLOATS.toLocaleString()} float32 probabilities,{' '}
            {FROZEN_MB} MB compressed, published as a release asset and checked against a pinned SHA-256 on load. By the
            end of the race only {neverVisited.toLocaleString(undefined, { maximumFractionDigits: 0 })} rows (
            {(FORENSICS.never_visited_by_2M * 100).toFixed(2)}%) had never been visited by a traversal; a row's average
            is banked at the opponent's visits, which are rarer, and the round-2 rows that never collected a sample play
            uniform in the frozen file. Graded, BR₀ is {FROZEN.br0.toFixed(4)} and BR₁ is +{FROZEN.br1.toFixed(4)}: a
            perfect P0 still loses, because P1 acts last in both rounds and at the draw and the seat is worth{' '}
            {(-FROZEN.value_p0).toFixed(4)} a hand to P1 in self-play.
          </p>
          <Figure
            n={4}
            caption={
              <>
                A perfect P0 still loses: the seat is worth a tenth of an ante to P1. Each seat's value when the strategy
                plays itself (hollow) and what a perfect adversary in that seat wins against it (filled); exploitability
                is the mean of the two gains over the seats' own values.
              </>
            }
            source="figures/rung3/answer_sheet.json · the frozen strategy's grade, chips per hand"
            label="Best responses against the frozen strategy, by seat"
          >
            <SeatsChart />
          </Figure>
          <p>
            What the strategy does is read by inner category, averaging over every physical deal of the category with
            equal weight: what it does <i>holding</i> that hand, not how often the line happens. These are readings, not
            tested reasons: a category is the three held cards alone, and its deals meet different boards, so a gap
            between two categories can come from that mix.
          </p>
          <Figure
            n={5}
            caption="P0 leads with every category and slowplays trips; P1 folds a pair more often than a high card. Round-one betting by inner hand category, commonest first."
            source="figures/rung3/answer_sheet.json · round_one"
            label="Round-one betting by inner hand category"
          >
            <SpotBars figures={ROUND_ONE} />
          </Figure>
          <Figure
            n={6}
            caption={
              <>
                High cards and pairs throw a card; flushes and better almost never do. P1 throws more often after P0
                stands pat: with a straight, {((1 - DRAW[1].rows[2].mix[0]) * 100).toFixed(0)}% against{' '}
                {((1 - DRAW[2].rows[2].mix[0]) * 100).toFixed(0)}% after P0 drew.
              </>
            }
            source="figures/rung3/answer_sheet.json · draw"
            label="The draw by inner hand category"
          >
            <SpotBars figures={DRAW} />
          </Figure>
          <p className="callout">
            Try it: open any spot in <a href="#learn/range/tabular">Learn</a> and see what the strategy does with every
            hand it could hold there, or sit down against it in <a href="#play">Play</a> with a coach that shows its mix
            at every spot you face.
          </p>
        </Chapter>

        {/* -------------------------------------------- Part III: the net */}
        <Chapter id="ch-06" title="A network for the ledger: the same game, the same walk, the same hands, and one question">
          <p>
            Rung 4 asks what a network gives up. It trains Deep CFR on this exact game with this exact sampler, grades it
            with this exact walk, and reports one number at every point of the race's grid:
          </p>
          <Eq>price = exploitability(Deep CFR's average) ÷ exploitability(LCFR's linear average), at equal hands per seat</Eq>
          <p>
            Equal hands is the fair axis: Deep CFR at {ARMS['4x256'].marks[0].hands / ARMS['4x256'].marks[0].t} walks per
            seat per iteration has seen the same hands at iteration t as LCFR had at lockstep iteration 1,000 t, so the
            marks 10, 20, 50, 100, 200 and 500 sit on the race's checkpoints exactly. Equal hands is not equal compute: an
            iteration with two refits of a 4 × 256 net costs minutes against ten table updates, and the best arm's five
            million hands took {(best500.wallClockS / 3600).toFixed(0)} hours of training.
          </p>
          <table className="settled" aria-label="The equal-hands grid">
            <tbody>
              {BEST.marks.map((m) => (
                <tr key={m.t}>
                  <td>{m.hands.toLocaleString()} hands</td>
                  <td>
                    tabular iteration {(m.t * 1000).toLocaleString()} · Deep CFR iteration {m.t} · LCFR grades {p100(lcfrAt(m.hands))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3>One iteration: the ledger's regret half becomes a regression</h3>
          <p>
            Nothing about the walk changes. For each seat in turn, both seats' nets are written into the packed table's
            regret slots, so the ledger's own regret matching plays them; the walk samples K hands and returns the
            traverser's regret samples and the opponent's strategy samples as rows; the rows are merged into fixed-size
            reservoirs, a fair sample of everything ever offered; and the seat's net is refit from scratch on its memory.
            A net sees 136 inputs per spot — which cards are held, discarded and on the board, who drew, the betting line,
            the chips — and never a row number, so two spots that look alike get alike outputs. That is the bet:
            generalisation across the millions of spots the walk never sampled, bought with a fit.
          </p>
          <p>
            The fit is a regression. Per row, the squared error summed over the spot's legal columns, each row weighted
            by its iteration t:
          </p>
          <Eq>loss = mean over rows of (t ÷ mean t) · Σ over legal k of (y[k] − r[k])²</Eq>
          <p>
            Why squared error: the output that minimises a t-weighted squared error over a spot's samples is their
            t-weighted mean, which is exactly what LCFR's table holds there divided by Σt, and regret matching is
            invariant to that positive factor. With a perfect "net" that returns each row's weighted sum, the loop
            reproduces rung 3's LCFR bit for bit; that pin is in the test suite. What moves what: a later sample
            outvotes an earlier one t-fold; a wider net tells more rows apart; more steps settle closer to the mean. Three
            objects are graded at every mark, because they answer different questions — the <b>exact average</b>, CFR's
            actual answer, accumulated from each iteration's strategy with the seat's own reach; <b>Π</b>, a fresh policy
            net fit on the strategy samples, the only average a game <i>without</i> a table can keep; and the{' '}
            <b>current iterate</b>, which CFR never promises will converge.
          </p>
          <Figure
            n={7}
            caption={
              <>
                At one million hands the net is exploitable for {p100(best100.exact.exploitability)} against the
                table's {p100(lcfrAt(1_000_000))}, about {fraction(KEY_NUMBERS.priceAt1M)}; at five million
                it is {p100(best500.exact.exploitability)} against {p100(lcfrAt(5_000_000))}, and its curve has
                flattened where the table's keeps falling toward {final('lcfr').toFixed(1)} at {M(hands)}. The lower
                strip is the price. Hover or arrow a mark.
              </>
            }
            source={`figures/rung3/grades.json and figures/rung4/sweep_256x4-s16000-lr5e-4.json (4 × 256, 16,000-step refits at lr 5e-4) with figures/rung4/grades.json (3 × 64, the spec's default); chips per hundred hands, both axes log; seed 0`}
            label="Exploitability of LCFR and Deep CFR against hands per seat, with the ratio"
          >
            <DeepRaceChart />
          </Figure>
          <p>
            Read the shape, not only the gap. The table falls on a straight line in log–log, about 1/√N; the net starts
            an eighth as exploitable and bends flat after two million hands:{' '}
            {p100(best100.exact.exploitability)} → {p100(best200.exact.exploitability)} →{' '}
            {p100(best500.exact.exploitability)} while the table went {p100(lcfrAt(1_000_000))} →{' '}
            {p100(lcfrAt(2_000_000))} → {p100(lcfrAt(5_000_000))}. If the net stays near{' '}
            {p100(best500.exact.exploitability)}, the table passes it somewhere between five and twenty million hands;
            that is an extrapolation, and the one net graded past a million hands is one net. The spec's default size,
            3 × 64, is the faint line: about {fraction(small.marks[0].exact.exploitability / lcfrAt(small.marks[0].hands))} of the
            table at {M(small.marks[0].hands)} hands and {fraction(small.marks.at(-1)!.exact.exploitability / lcfrAt(small.marks.at(-1)!.hands))}{' '}
            by {M(small.marks.at(-1)!.hands)}, which is why the next chapter exists.
          </p>
        </Chapter>

        <Chapter id="ch-07" title={<>Width is the lever: doubling it cut {cut(size('3 × 64').exact, size('3 × 128').exact)}, an extra layer cut {cut(size('3 × 64').exact, size('4 × 64').exact)} to {cut(size('3 × 128').exact, size('4 × 128').exact)}</>}>
          <p>
            Five nets at one million hands, all with the spec's refit (4,000 Adam steps at 1e-3), all on identical hands.
            The sweep's verdict was fixed before its numbers existed — {SWEEP_RULE.rule} — and it picked{' '}
            {SWEEP_RULE.winner.split('x').reverse().join(' × ')}; the later arms went wider still. A wider layer tells
            more spots apart; a deeper one composes the same features more. On this game the first is what was missing.
          </p>
          <Figure
            n={8}
            caption={
              <>
                Width is the lever. 3 × 64 → 3 × 128 cut the one-million-hand grade{' '}
                {cut(size('3 × 64').exact, size('3 × 128').exact)}; a fourth layer cut{' '}
                {cut(size('3 × 64').exact, size('4 × 64').exact)} at 64 wide and{' '}
                {cut(size('3 × 128').exact, size('4 × 128').exact)} at 128; the widest net, 4 × 256, reached{' '}
                {p100(size('4 × 256').exact)}.
              </>
            }
            source="figures/rung4/sweep.json (3 × 64, 4 × 64, 3 × 128 on Modal), sweep_128x4.json and sweep_256x4.json (on a Mac) · the exact average at mark 100; one seed, a gap under about 10% is suggestive"
            label="Exact average of five net sizes at one million hands per seat"
          >
            <SizeChart />
          </Figure>
          <h3>A longer refit helps the small net, not the big one; halving the rate with four times the steps bought under one percent</h3>
          <p>
            Each refit is a fresh net fit to the memory's t-weighted mean, so more Adam steps settle closer to that mean,
            which helps a net that was under-fit and barely moves one near its floor. Doubling the refit from 4,000 to
            8,000 steps took 3 × 128 from {p100(refit('3 × 128', 4000).exact)} to {p100(refit('3 × 128', 8000).exact)}{' '}
            ({cut(refit('3 × 128', 4000).exact, refit('3 × 128', 8000).exact)}) and 4 × 256 from{' '}
            {p100(refit('4 × 256', 4000).exact)} to {p100(refit('4 × 256', 8000).exact)} (
            {cut(refit('4 × 256', 4000).exact, refit('4 × 256', 8000).exact)}). The learning rate sets how far each step
            goes; 16,000 steps at half the rate, with a stop rule fixed before the grade (continue past a million hands
            only below {p100(refit('4 × 256', 8000).exact * 0.9)}), graded {p100(refit('4 × 256', 16000).exact)}: the
            floor at 4 × 256 is not fit error. Early on a longer fit can be worse, so arms are compared from half a
            million hands.
          </p>
          <Figure
            n={9}
            caption={
              <>
                A doubled refit cut {cut(refit('3 × 128', 4000).exact, refit('3 × 128', 8000).exact)} at 3 × 128 and{' '}
                {cut(refit('4 × 256', 4000).exact, refit('4 × 256', 8000).exact)} at 4 × 256; four times the steps at half
                the rate cut {cut(refit('4 × 256', 8000).exact, refit('4 × 256', 16000).exact)} more. Same net, same hands;
                only the fit changes.
              </>
            }
            source="figures/rung4/sweep.json, sweep_128x3-s8000.json, sweep_256x4.json, sweep_256x4-s8000.json, sweep_256x4-s16000-lr5e-4.json · the exact average at mark 100"
            label="Exact average of five refit settings at one million hands per seat"
          >
            <RefitChart />
          </Figure>
        </Chapter>

        <Chapter id="ch-08" title="The exact average wins at every mark, and the policy net's stall was its loss">
          <p>
            CFR's answer is the average of its iterates, never the last one. The exact average is accumulated as the
            run goes, each iteration's strategy weighted by t and by the seat's own reach, pushed through a predecessor
            table over the public tree (a post-draw spot has three pre-draw predecessors, all of which count). Π is the
            original Deep CFR's average: during every phase the opponent's walks copy the strategy they sampled from into
            a policy memory, and a fresh net is fit to those rows at each mark. The current iterate is regret matching
            over the materialised table. Exact beats Π at every graded mark of every linear arm, usually by about a
            fifth; the current iterate is usually worst but not always, and it swings:{' '}
            {p100(best100.current.exploitability)} at a million hands, {p100(best200.current.exploitability)} at two,{' '}
            {p100(best500.current.exploitability)} at five.
          </p>
          <Figure
            n={10}
            caption={
              <>
                Exact &lt; Π at every mark; the current iterate is promised nothing and shows it. The 4 × 256 arm's three
                objects from a hundred thousand to five million hands; hover or arrow a mark for each seat's best
                response.
              </>
            }
            source="figures/rung4/sweep_256x4-s16000-lr5e-4.json · deep/<t>/{exact, pi, current}; chips per hundred hands, both axes log"
            label="Three graded objects of one Deep CFR arm against iteration"
          >
            <AveragesChart />
          </Figure>
          <h3>Π's stall from two to five million hands was the fit, and cross-entropy repairs it</h3>
          <p>
            Between two and five million hands the exact average fell from {p100(best200.exact.exploitability)} to{' '}
            {p100(best500.exact.exploitability)} while Π went from {p100(best200.pi.exploitability)} to{' '}
            {p100(best500.pi.exploitability)}. Π is a fresh net fit on the policy memory at each mark, so the stall is in
            the memory or in the fit. A sweep held the memory fixed — the one saved at mark 500, and at marks 100 and 200
            for three arms — and changed only the fit: {arms500.length} refits of the loss, the head, the step count, the
            schedule, the width and the depth, each frozen the way Π is frozen and graded exactly. The control, the run's
            own recipe refit, reproduces the run's Π ({p100(control.exploitability, 2)} against{' '}
            {p100(PI_REFERENCE.runPi, 2)}); the exact average's own grade is {p100(PI_REFERENCE.exact, 2)}.
          </p>
          <p>
            Cross-entropy on a softmax head beats squared error on a linear head at every step count:{' '}
            {p100(ce16.exploitability, 2)} against {p100(control.exploitability, 2)} at 16,000 steps,{' '}
            {p100(ce64.exploitability, 2)} against {p100(mse64.exploitability, 2)} at 64,000 with cosine decay, after
            which more steps are flat. Wider helps a little (a 1,024-wide cross-entropy refit reaches{' '}
            {p100(widest.exploitability, 2)}, the exact average itself), deeper does not ({p100(deeper.exploitability, 2)}{' '}
            at 256 × 6), and the KL divergence reproduces cross-entropy to the digit, as it must: the two differ by the
            targets' own entropy and share a gradient. Refit at every mark, the cross-entropy Π tracks the exact average
            (
            {[100, 200, 500].map((t) => p100(piArm(t === 500 ? 'ce-softmax-64k-cos' : `ce-softmax-64k-cos@t${t}`).exploitability, 2)).join(' → ')}
            ) where the run's Π stalled. The memory was fine; the loss was wrong for the job.
          </p>
          <Figure
            n={11}
            caption={
              <>
                Cross-entropy beats squared error at every step count and reaches {p100(ce64.exploitability, 2)} at 64k,
                near the exact average's {p100(PI_REFERENCE.exact, 2)} (left); refit at each mark it falls beside the
                exact average while the run's own Π rose (right). KL equals cross-entropy to the digit and is not drawn.
              </>
            }
            source="figures/rung4/pi_sweep/results.json with sweep_256x4-s16000-lr5e-4.json for the exact average and the run's Π · one memory per mark, one seed"
            label="Policy-net refits by loss and step count, and across marks"
          >
            <PiLossChart />
          </Figure>
          <p>
            Why the loss matters this much: a squared-error fit prices a miss by its size, and cross-entropy by its
            ratio. The best response exploits exactly the small probabilities — a bluff that should happen four percent
            of the time and happens one — and squared error is nearly indifferent to them. The reach-weighted error
            against the exact average says the same thing in numbers: {reachWeighted('run-pi (frozen t500)').rw_mad.toFixed(3)}{' '}
            for the run's Π, {reachWeighted('ce-softmax-64k-cos').rw_mad.toFixed(3)} for the cross-entropy refit, tracking
            exploitability where a plain count of zeroed small probabilities does not.
          </p>
          <Figure
            n={12}
            caption="The same 0.03 miss costs fourteen times more under cross-entropy at a rare action than under squared error, and the same at a coin flip. Both curves are the losses' definitions, computed; nothing read from a file."
            source="the two losses' formulas: squared error, and cross-entropy minus its floor (the KL divergence)"
            label="Squared error and cross-entropy for a 0.03 miss at a 4 percent action and at a 50 percent action"
          >
            <LossShapes />
          </Figure>
          <p className="callout">
            The training loop still fits Π with squared error; the sweep is reported as a sweep. All three strategies —
            the frozen table, Π and the exact average at mark 500 — are on <a href="#learn/range/deep-pi">Learn</a> behind
            one switch.
          </p>
        </Chapter>

        <Chapter id="ch-09" title={<>Per-sample DCFR and CFR+ lose to linear by {KEY_NUMBERS.ruleScreenRatio.toFixed(0)} times and get worse with training, because there is no total to test</>}>
          <p>
            Deep CFR stores no running regret. Each refit learns the t-weighted <i>mean</i> of every regret sample the
            reservoir holds, and regret matching reads that mean's sign pattern. So the only place a rule can act is on
            the individual samples: a DCFR sample from iteration s is rescaled by the discount it would have earned by
            now, using <b>that sample's own sign</b>; a CFR+ sample keeps weight 1 if positive and counts only in its own
            iteration if negative. The result, with the 4 × 256 net and 4,000-step refits: at half a million hands
            linear grades {p100(screen('linear', 50))}, per-sample DCFR {p100(screen('dcfr', 50))} and per-sample CFR+{' '}
            {p100(screen('cfrplus', 50))}, and both rise with training (DCFR {p100(screen('dcfr', 10))} →{' '}
            {p100(screen('dcfr', 20))} → {p100(screen('dcfr', 50))}).
          </p>
          <p>
            The mechanism, measured on the saved memories: by iteration 50 the screen keeps {pct(flip('dcfr', 50).negativeKept)}{' '}
            (DCFR) or {pct(flip('cfrplus', 50).negativeKept)} (CFR+) of the memory's negative regret mass against{' '}
            {pct(flip('dcfr', 50).positiveKept)} to {pct(flip('cfrplus', 50).positiveKept)} of its positive mass. A
            sample's sign is not its cell's sign: a genuinely bad action still draws some positive samples from lucky
            walks, and when the negatives are discounted away and the lucky positives are not, the fitted mean turns
            positive. {pct(flip('dcfr', 50).flipped)} of the seat-0 cells that are negative under linear weights turn
            positive by iteration 50 under DCFR, up from {pct(flip('dcfr', 10).flipped)} at iteration 10, and regret
            matching plays every positive cell. A small worked case: two actions with samples b = (+1, −3, +1, −3) and
            c = (+0.5, +0.5, +0.5, +0.5). Per-sample clipping fits both to +0.5, so they are played equally; CFR+ on the
            totals would floor b at zero and play c. A per-sample rule inflates noisy actions, and noisy actions are what
            sampling produces.
          </p>
          <Figure
            n={13}
            caption={
              <>
                Per-sample DCFR and CFR+ are {KEY_NUMBERS.ruleScreenRatio.toFixed(0)} times worse than linear at
                iteration 50 and rising while linear falls (left); the share of bad cells each rule turns positive grows
                with every mark (right, seat 0; hover a bar for the negative mass kept).
              </>
            }
            source="figures/rung4/rule_dcfr-256x4.json, rule_cfrplus-256x4.json and sweep_256x4.json (the exact average), rule_sign_flips.json (the mechanism) · chips per hundred hands"
            label="Per-sample regret rules under a net: exploitability by iteration, and the share of bad cells turned positive"
          >
            <RuleScreenChart />
          </Figure>
          <p>
            The rule that falls out of chapters 04 and 09 together: under sampling a regret discount must be blind to
            the regret's sign and carry no clock of its own. Linear weighting satisfies both. A sampled DCFR needs a visit
            clock; a neural CFR+ needs a total to test, which a net that fits a mean does not have.
          </p>
        </Chapter>

        <Chapter id="ch-10" title="What is and is not settled">
          <table className="settled">
            <tbody>
              <tr>
                <td>one seed</td>
                <td>
                  Every run on this page is seed 0, one run per arm. The 3–4× between the rules that keep negative regret
                  and the rules that discard it, the {KEY_NUMBERS.priceAt1M.toFixed(2)}× at a million hands and the {KEY_NUMBERS.ruleScreenRatio.toFixed(0)}× of
                  the per-sample screen are far outside seed noise; LCFR's {final('lcfr').toFixed(1)} against vanilla's{' '}
                  {final('vanilla').toFixed(1)}, and any gap under about ten percent, is suggestive.
                </td>
              </tr>
              <tr>
                <td>equal hands</td>
                <td>
                  Equal hands is not equal compute. A Deep CFR iteration refits two nets; the best arm's five million
                  hands took {(best500.wallClockS / 3600).toFixed(0)} hours on a laptop GPU, and the table's twenty million
                  ran on eleven Modal cores per rule.
                </td>
              </tr>
              <tr>
                <td>the endpoint</td>
                <td>
                  No net has been run to the race's own endpoint, {M(hands)} hands. Whether the table passes the net, and
                  where, is extrapolated from one arm's slope.
                </td>
              </tr>
              <tr>
                <td>the policy net</td>
                <td>
                  The cross-entropy result is a post-hoc refit on saved memories. The training loop's Π is still fit with
                  squared error, and the released Π on Learn is that one.
                </td>
              </tr>
              <tr>
                <td>the two 3 × 64 runs</td>
                <td>
                  The spec's default net was run twice, on a Mac to half a million hands and on Modal to a million; the
                  rows are identical for any worker count, so the two are one curve, but the refit arithmetic differed.
                </td>
              </tr>
              <tr>
                <td>the visit clock</td>
                <td>
                  One arm, {M(visit2M[0].hands)} hands, W = {visit2M[0].workers}, graded at four checkpoints. The full race
                  at the race's worker count has not been run.
                </td>
              </tr>
              <tr>
                <td>categories</td>
                <td>
                  Figures 5 and 6 are readings, not reasons: a category is the three held cards alone, and its deals meet
                  different boards.
                </td>
              </tr>
            </tbody>
          </table>
        </Chapter>

        <section className="methods" id="methods">
          <p className="label">Methods and footnotes</p>
          <p>
            Exploitability is the mean of the two seats' exact best responses, computed by a vector walk along the public
            tree that imports the game and nothing from any learner. The page speaks in chips per hundred hands; rung 3's
            files and the run notes for rung 4 are in chips per hand, and this page multiplies by 100 in one place.
            Nothing on this page was typed in: every number is read from a copy of a committed file.
          </p>
          <ol>
            <li>
              <code>figures/rung3/grades.json</code> — the race: every rule at eight checkpoints in every averaging column
              it banked, plus uniform random; written by <code>uv run minidraw-analysis</code>.
            </li>
            <li>
              <code>figures/rung3/answer_sheet.json</code> — the frozen strategy's grade and what it does in round 1 and at
              the draw, by category; the same command.
            </li>
            <li>
              <code>figures/rung3/visit_forensics.json</code> — how idle the rows were, from the four 2M checkpoints;{' '}
              <code>scripts/minidrawmaha/checkpoint_forensics.py</code>.
            </li>
            <li>
              <code>figures/rung3/grades_visit_clock.json</code> — the visit-clock DCFR arm, both averages;{' '}
              <code>scripts/minidrawmaha/grade_visit_clock.py</code>.
            </li>
            <li>
              <code>figures/rung4/*.json</code> — every Deep CFR arm, three objects per mark, with each run's configuration
              beside each grade; <code>scripts/deepcfr/modal_rung4.py::grade</code>. Which arm is which is{' '}
              <code>docs/notes/rung4-run-log.md</code>.
            </li>
            <li>
              <code>figures/rung4/rule_sign_flips.json</code> — the per-sample screens' mechanism;{' '}
              <code>scripts/deepcfr/rule_sign_flips.py</code>.
            </li>
            <li>
              <code>figures/rung4/pi_sweep/</code> — the policy-net loss sweep and its reach-weighted errors;{' '}
              <code>scripts/deepcfr/pi_loss_sweep.py</code> and <code>pi_reach_weighted.py</code>.
            </li>
            <li>
              The learner is <code>minidrawmaha/solver/mccfr.py</code> under <code>lockstep.py</code>, with the rules in{' '}
              <code>regret_rules.py</code>; the net, its refit and the loop are <code>deepcfr/</code>; the grader is{' '}
              <code>minidrawmaha/exploitability.py</code>. The trainer plays one seeded pack of {600} pre-dealt hands with
              the frozen mix at every node, reshuffled each sitting with the seats alternating.
            </li>
          </ol>
          <p className="authors">
            Alec Ondo, with Claude. The race ran on Modal; the Deep CFR arms ran on a Mac and on Modal; every grade is by
            the same walk. Read next: <a href="/rung2">rung 2, Leduc</a>, where the same ledger meets an outside
            referee, and <a href="/solver">/solver</a>, the range interface over all three strategies.
          </p>
        </section>
      </div>
    </div>
  )
}
