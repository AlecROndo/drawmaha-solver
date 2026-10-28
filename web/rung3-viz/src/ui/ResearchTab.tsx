import { useEffect, useState } from 'react'
import {
  ARMS,
  BEST,
  KEY_NUMBERS,
  PI_REFERENCE,
  PI_SWEEP,
  POT_ARMS,
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
import { currentSection } from './contents'
import { DeepRaceChart } from './DeepRaceChart'
import { EqualHandsChart } from './EqualHandsChart'
import { Figure } from './Figure'
import { fraction, hoursAt } from './scales'
import { REACH, bandAt, binLabel, commonInfosets, commonShare, crossingBand, onceIn, perVisit, populated, rareInfosets, rareShare } from '../reach'
import { LossShapes } from './LossShapes'
import { PiLossChart } from './PiLossChart'
import { PotChart } from './PotChart'
import { RaceChart } from './RaceChart'
import { ReachChart } from './ReachChart'
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
  { id: 'ch-01', n: '01', short: 'Rules and exact grading' },
  { id: 'ch-02', n: '02', short: 'MCCFR: sampling the tree' },
  { id: 'ch-03', n: '03', short: 'Racing four regret rules' },
  { id: 'ch-04', n: '04', short: 'The answer and its play' },
  { id: 'ch-05', n: '05', short: 'Deep CFR: net, not table' },
  { id: 'ch-06', n: '06', short: 'Net size, refits and √pot' },
  { id: 'ch-07', n: '07', short: 'Exact average vs. the net' },
  { id: 'ch-08', n: '08', short: 'DCFR, CFR+ under Deep CFR' },
  { id: 'ch-09', n: '09', short: 'Settled and open questions' },
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

/**
 * Which entry of the contents rail the reader is at (`currentSection` in
 * `contents.ts` makes the call): the last section whose top has crossed a
 * reading line 30% of the way down the viewport, or the last section once a
 * page that scrolls is at its end (the methods are short enough that their
 * top may never reach the line). Nothing is current above chapter 01.
 * Measured on scroll, on resize and whenever the body changes height (a
 * figure or a font sizing late), one layout read per section, coalesced to a
 * frame; the hash jump the rail's links make is a scroll like any other, so
 * the box follows a click too.
 */
function useCurrentSection(ids: readonly string[]): string | null {
  const [current, setCurrent] = useState<string | null>(null)
  useEffect(() => {
    let frame = 0
    const measure = () => {
      frame = 0
      const line = Math.max(innerHeight * 0.3, 120)
      const atEnd = scrollY > 0 && scrollY + innerHeight >= document.documentElement.scrollHeight - 2
      const tops = ids.map((id) => document.getElementById(id)?.getBoundingClientRect().top ?? Infinity)
      const at = currentSection(tops, line, atEnd)
      setCurrent(at < 0 ? null : ids[at])
    }
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(measure)
    }
    measure()
    addEventListener('scroll', schedule, { passive: true })
    addEventListener('resize', schedule)
    const grew = new ResizeObserver(schedule)
    grew.observe(document.body)
    return () => {
      if (frame) cancelAnimationFrame(frame)
      removeEventListener('scroll', schedule)
      removeEventListener('resize', schedule)
      grew.disconnect()
    }
  }, [ids])
  return current
}

const SECTION_IDS = CHAPTERS.map((c) => c.id)

const final = (rule: Rule) => perHundred(series(rule).at(-1)!.exploitability)
const p100 = (chipsPerHand: number, digits = 1) => perHundred(chipsPerHand).toFixed(digits)
const pct = (x: number) => (x * 100 < 1 && x > 0 ? `${(x * 100).toFixed(1)}%` : `${Math.round(x * 100)}%`)
const cut = (from: number, to: number) => `${Math.round((1 - to / from) * 100)}%`
const M = (n: number) => `${(n / 1e6).toFixed(n % 1e6 ? 1 : 0)}M`

export function ResearchTab() {
  const current = useCurrentSection(SECTION_IDS)
  const mini = SIZES[2].infosets
  const leduc = SIZES[1].infosets
  const hands = FINAL_ITERATION * WORKERS
  const best100 = markOf(BEST, 100)
  const best200 = markOf(BEST, 200)
  const best500 = markOf(BEST, 500)
  const size = (net: string) => SIZE_SWEEP.find((s) => s.net === net)!
  const refit = (net: string, steps: number) => REFIT_ARMS.find((a) => a.net === net && a.steps === steps)!
  const screen = (rule: 'linear' | 'dcfr' | 'cfrplus', t: number) => RULE_SCREEN[rule].find((p) => p.t === t)!.exact
  const pot = (k: number, lr = 0.0005) => POT_ARMS.find((a) => a.potPower === k && a.lr === lr)!
  const rawGap = (t: 100 | 200) => Math.abs(pot(0).exact[t] - pot(0, 0.001).exact[t])
  const sqrtPot10 = markOf(ARMS['pot0.5-4x256-s16000-lr5e-4'], 10).exact.exploitability
  const flip = (rule: 'dcfr' | 'cfrplus', t: number) => SIGN_FLIPS[rule].find((f) => f.t === t && f.seat === 0)!
  const ce64 = piArm('ce-softmax-64k-cos')
  const mse64 = piArm('mse-linear-64k-cos')
  const ce16 = piArm('ce-softmax-16k')
  const control = piArm('mse-linear-16k')
  const widest = piArm('ce-softmax-1024x4-64k-cos')
  const deeper = piArm('ce-softmax-256x6-64k-cos')
  const arms500 = PI_SWEEP.filter((a) => a.t === 500)
  const small = ARMS['mac-3x64']

  return (
    <div className="article">
      <dl className="keynums" aria-label="Key numbers">
        <div>
          <b>
            {final('lcfr').toFixed(1)} <small>vs</small> {final('cfrplus').toFixed(1)} <small className="unit">chips/100</small>
          </b>
          <span>what a perfect adversary wins per hundred hands against LCFR and against CFR+, after {M(hands)} hands per seat</span>
        </div>
        <div>
          <b>
            {KEY_NUMBERS.priceAt1M.toFixed(2)}
            <small>×</small>
          </b>
          <span>ratio of the net's exploitability to the table's at one million hands per seat, about {fraction(KEY_NUMBERS.priceAt1M)}; same walk, same hands</span>
        </div>
        <div>
          <b>
            {p100(KEY_NUMBERS.netAt5M)} <small>vs</small> {p100(KEY_NUMBERS.tableAt5M)} <small className="unit">chips/100</small>
          </b>
          <span>net against table at five million hands, per hundred hands; the net has flattened, the table has not</span>
        </div>
        <div>
          <b>
            {KEY_NUMBERS.ruleScreenRatio.toFixed(0)}
            <small>×</small>
          </b>
          <span>ratio of per-sample DCFR's exploitability to linear weighting's at half a million hands, and rising with training</span>
        </div>
      </dl>

      <nav className="contents" aria-label="Contents">
        <span className="k">Contents</span>
        <ol>
          {CHAPTERS.map((c) => (
            <li key={c.id} className={c.id === current ? 'current' : undefined}>
              <a href={`#${c.id}`} aria-current={c.id === current ? 'true' : undefined}>
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
            That asymmetry is why, under sampling, DCFR and CFR+ finished as one rule.
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
          <p className="callout">
            Why the two sign-tested rules collapsed onto each other: DCFR's discount runs on the global iteration clock, so
            a row sampled once in thousands of iterations is halved thousands of times between two pieces of evidence, which
            is CFR+'s floor. The proof, the forensics on the checkpoints and the visit-clock arm are in{' '}
            <code>docs/notes/regret-rules-dcfr-cfrplus.md</code>; chapter 08 meets the same failure under a net.
          </p>
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

        <Chapter id="ch-04" title={<>The answer: LCFR's linear average at {M(hands)} hands, {FROZEN.exploitability.toFixed(4)} chips a hand</>}>
          <p>
            The winner is frozen as rung 3's solution: {FROZEN_FLOATS.toLocaleString()} float32 probabilities,{' '}
            {FROZEN_MB} MB compressed, published as a release asset and checked against a pinned SHA-256 on load. The
            round-2 rows the sampler's own play rarely reaches collected few or no samples and play uniform in the frozen
            file, which is what chapter 04's reach figure measures. Graded, BR₀ is {FROZEN.br0.toFixed(4)} and BR₁ is +{FROZEN.br1.toFixed(4)}: a
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
          <h3>Where the exploitability lives: a row rots as the sampler sees it less</h3>
          <p>
            A best responder's edge can be charged to the spots it wins it at. At each of the responder's infosets,
            how often the strategy's own play brings it there, times what the best response earns from there over what
            the strategy's own mix earns, is that spot's share; summed over every infoset the shares are exactly the
            best responder's gain, and halved over the two seats they are the exploitability the race graded. The same
            walk gives each spot's frequency: how often it comes up per hand when the strategy plays itself, which is
            also how often the sampler visited it, since external sampling draws the deal and the opponent from the
            strategy. Binning the {REACH.tabular.infosets.toLocaleString()} spots by that frequency says where the{' '}
            {p100(REACH.tabular.exploitability)} chips per hundred hands sit.
          </p>
          <p>
            Two readings of the same bins. By share of the total: the common spots, reached at least{' '}
            {onceIn(1e-3)}, are {pct(commonInfosets(REACH.tabular, 1e-3))} of the infosets and carry{' '}
            {pct(commonShare(REACH.tabular, 1e-3))} of the exploitability; the middle bands, {onceIn(1e-3)} down to{' '}
            {onceIn(1e-6)}, are {pct(commonInfosets(REACH.tabular, 1e-6) - commonInfosets(REACH.tabular, 1e-3))} of
            the infosets and carry <b>{pct(commonShare(REACH.tabular, 1e-6) - commonShare(REACH.tabular, 1e-3))}</b>; the
            rare tail, {pct(rareInfosets(REACH.tabular, 1e-6))} of the infosets, carries{' '}
            {pct(rareShare(REACH.tabular, 1e-6))}, and {pct(bandAt(REACH.tabular, 0).infosets / REACH.tabular.infosets)}{' '}
            of the rows are never reached at all under the strategy's own play. By leak per visit, the reading that says
            how good a row is once play gets there: a spot in the commonest band leaks{' '}
            {(100 * perVisit(populated(REACH.tabular)[0])).toFixed(1)} chips per hundred visits; one reached{' '}
            {onceIn(1e-5)} leaks {(100 * perVisit(bandAt(REACH.tabular, 1e-5))).toFixed(1)}; {onceIn(1e-7)},{' '}
            {(100 * perVisit(bandAt(REACH.tabular, 1e-7))).toFixed(0)}; and the rarest band,{' '}
            {(100 * perVisit(populated(REACH.tabular).at(-1)!)).toFixed(0)}: <b>{(perVisit(populated(REACH.tabular).at(-1)!) / perVisit(populated(REACH.tabular)[0])).toFixed(0)} times worse</b>{' '}
            per visit from the commonest band to the rarest, rising at every step between. A row the sampler visited a
            handful of times in {M(hands)} hands is still close to the uniform it started as, and the best responder
            finds every one. That is the shape of the sampler's error, and it is what a network should change: a net's
            row at an unvisited spot is what it learned from the visited ones, not uniform. Chapter 05 puts the two side
            by side. The split by stage says the same thing another way: round 2, where the rare rows are, holds{' '}
            {pct(REACH.tabular.stages['round 2'].exploitability / REACH.tabular.exploitability)} of the table's
            exploitability.
          </p>
          <Figure
            n={7}
            caption={
              <>
                Where the frozen table's exploitability lives, and how fast a row rots as the sampler sees it less. One
                axis, how often play reaches the spot, commonest at the left. Top, each band's share of the
                exploitability (yellow) beside its share of the infosets (hollow): the rarest band holds{' '}
                {pct(populated(REACH.tabular).at(-1)!.infosets / REACH.tabular.infosets)} of the rows and{' '}
                {pct(populated(REACH.tabular).at(-1)!.exploitability / REACH.tabular.exploitability)} of the leak. Bottom, what a spot in the band leaks each time it is reached,
                rising from {(100 * perVisit(populated(REACH.tabular)[0])).toFixed(1)} chips per hundred visits in the
                commonest band to {(100 * perVisit(populated(REACH.tabular).at(-1)!)).toFixed(0)} in the rarest. Hover or
                arrow a band to read it.
              </>
            }
            source="figures/rung3/exploit_by_reach.json, written by uv run minidraw-exploit-by-reach --release tabular · one-step gains from the exact walk, summed per table key, binned by the key's frequency under the strategy's own play"
            label="Share of exploitability and share of infosets per reach band, the frozen table"
          >
            <ReachChart profiles={['tabular']} />
          </Figure>
          <p className="callout">
            Try it: open any spot in <a href="#learn/range/tabular">Learn</a> and see what the strategy does with every
            hand it could hold there, or sit down against it in <a href="#play">Play</a> with a coach that shows its mix
            at every spot you face.
          </p>
        </Chapter>

        {/* -------------------------------------------- Part III: the net */}
        <Chapter id="ch-05" title="A network for the ledger: the same game, the same walk, the same hands, and one question">
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
          <Figure
            n={8}
            caption={
              <>
                Two clocks on one ruler. Deep CFR's iteration t and the table's lockstep iteration 1,000 t have seen the
                same hands per seat, so the six marks sit on the race's checkpoints exactly, and every column lines up
                through both clocks. The table's grade at each notch, {p100(lcfrAt(BEST.marks[0].hands))} falling to{' '}
                {p100(lcfrAt(best500.hands))}, is what the price divides by; the hours under the top clock are the compute
                the equal-hands axis leaves out, {hoursAt(BEST.marks[0].wallClockS)} at the first mark and{' '}
                {hoursAt(best500.wallClockS)} at the last. Hover or arrow a notch.
              </>
            }
            source="figures/rung4/sweep_256x4-s16000-lr5e-4.json (the best arm's marks: iteration, hands per seat, wall clock) and figures/rung3/grades.json (LCFR at the same hands); hands on a log axis, the grades on a log scale of their own"
            label="Deep CFR's iteration and the table's lockstep iteration on one axis of hands per seat, with the table's grade at each mark"
          >
            <EqualHandsChart />
          </Figure>
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
            outvotes an earlier one t-fold; a wider net tells more rows apart; more steps settle closer to the mean. The
            arms on this page fit raw chips; since the pot arms at the end of chapter 06 — two arms, one seed each —
            the trainer's default divides each target by √pot at its spot first, with chapter 09 saying what that
            default has not yet been graded at. Three
            objects are graded at every mark, because they answer different questions — the <b>exact average</b>, CFR's
            actual answer, accumulated from each iteration's strategy with the seat's own reach; <b>Π</b>, a fresh policy
            net fit on the strategy samples, the only average a game <i>without</i> a table can keep; and the{' '}
            <b>current iterate</b>, which CFR never promises will converge.
          </p>
          <Figure
            n={9}
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
          <h3>The net's exploitability lives in different spots from the table's</h3>
          <p>
            Chapter 04's decomposition, run on the two Deep CFR averages at {M(best500.hands)} hands, says what the net
            changed. Per visit, the net's exact average leaks{' '}
            {(perVisit(bandAt(REACH['deep-exact'], populated(REACH.tabular)[0].floor)) / perVisit(bandAt(REACH.tabular, populated(REACH.tabular)[0].floor))).toFixed(1)} times
            what the table leaks at the commonest spots and{' '}
            {(perVisit(bandAt(REACH.tabular, populated(REACH.tabular).at(-1)!.floor)) / perVisit(bandAt(REACH['deep-exact'], populated(REACH.tabular).at(-1)!.floor))).toFixed(1)} times
            less at the rarest: its leak per visit rises{' '}
            {(perVisit(populated(REACH['deep-exact']).at(-1)!) / perVisit(populated(REACH['deep-exact'])[0])).toFixed(0)}-fold
            from the commonest band to the rarest where the table's rises{' '}
            {(perVisit(populated(REACH.tabular).at(-1)!) / perVisit(populated(REACH.tabular)[0])).toFixed(0)}-fold. So the
            net's exploitability sits where play is: round 1 and the draw hold{' '}
            {pct(1 - REACH['deep-exact'].stages['round 2'].exploitability / REACH['deep-exact'].exploitability)} of it
            against {pct(1 - REACH.tabular.stages['round 2'].exploitability / REACH.tabular.exploitability)} of the
            table's. A table is exact wherever it has samples and bleeds wherever it does not; a net fills the rare rows
            from the common ones and pays for it in the common rows, where a few thousand samples beat a regression. The
            two solvers are wrong in different places, which is also why their best responses differ. Π reads like the
            exact average here with a little more leak everywhere, as its grade says. One seed and one grade each: a
            reading, not a theorem.
          </p>
          <Figure
            n={10}
            caption={
              <>
                Where each strategy's exploitability lives, the net at {M(best500.hands)} hands and the table at{' '}
                {M(hands)}: the table's leak per visit climbs{' '}
                {(perVisit(populated(REACH.tabular).at(-1)!) / perVisit(populated(REACH.tabular)[0])).toFixed(0)}-fold
                from the commonest band to the rarest, the net's{' '}
                {(perVisit(populated(REACH['deep-exact']).at(-1)!) / perVisit(populated(REACH['deep-exact'])[0])).toFixed(0)}-fold,
                and the net leaks more than the table where play is common. Same axis as Figure 7. Top, each strategy's
                share of its own exploitability per band beside the band's share of the infosets (hollow); bottom, the
                leak per hundred visits, where the table's line climbs past the net's in the{' '}
                {binLabel(crossingBand(REACH.tabular, REACH['deep-exact'])!)} band. Hover or arrow a band to read it.
              </>
            }
            source="figures/rung3/exploit_by_reach.json, figures/rung4/exploit_by_reach_deep-exact.json, exploit_by_reach_deep-pi.json · uv run minidraw-exploit-by-reach --release tabular | deep-exact | deep-pi"
            label="Share of exploitability per reach band for the table and the two Deep CFR averages"
          >
            <ReachChart profiles={['tabular', 'deep-exact', 'deep-pi']} />
          </Figure>
        </Chapter>

        <Chapter id="ch-06" title={<>Width is the lever: doubling it cut {cut(size('3 × 64').exact, size('3 × 128').exact)}, an extra layer cut {cut(size('3 × 64').exact, size('4 × 64').exact)} to {cut(size('3 × 128').exact, size('4 × 128').exact)}</>}>
          <p>
            Five nets at one million hands, all with the spec's refit (4,000 Adam steps at 1e-3), all on identical hands.
            The sweep's verdict was fixed before its numbers existed — {SWEEP_RULE.rule} — and it picked{' '}
            {SWEEP_RULE.winner.split('x').reverse().join(' × ')}; the later arms went wider still. A wider layer tells
            more spots apart; a deeper one composes the same features more. On this game the first is what was missing.
          </p>
          <Figure
            n={11}
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
            n={12}
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
          <h3>
            Dividing each regret target by √pot cut the two-million-hand grade{' '}
            {cut(pot(0).exact[200], pot(0.5).exact[200])}: the fit spends its accuracy by chip, not by pot squared
          </h3>
          <p>
            One net fits every spot's mean regret with one accuracy budget, and the loss decides where that budget goes.
            A regret is in chips, so a spot's regrets are about as large as its pot, and a relative miss ε at a spot
            costs (ε · pot)² per row. Squared error in raw chips therefore weighs a spot by how often it is reached
            times pot²: the all-in spots at 52 chips count 676 times the 2-chip ante spots they are reached as often
            as, and the ante spots, where every hand begins, are fit last. The repair is one division before the fit,
            with P the pot at the decision:
          </p>
          <Eq>at a spot with pot P: fit y to r ÷ P^k, with k = 0 (raw), ½ or 1</Eq>
          <p>
            Why the division is free: P is shared by every row at the spot, so the spot's mean regret is scaled by one
            constant across its actions, and regret matching, which reads only a row's sign pattern and ratios, plays
            the same σ from the scaled regrets as from the raw ones. That is exact for the regrets themselves; what the
            fitted net gets <i>wrong</i>, and so the σ it actually plays, is what the weighting moves, and is the point.
            A divisor that depended on the sampled outcome would reweight rows within a spot and move its mean; this
            one cannot. What k does: dividing by P^k leaves a weight of how often × P^(2 − 2k). k = ½ leaves how often ×
            pot, the chips a σ mistake at that spot costs the player; k = 1 weighs every spot alike, however small its
            pot. The accuracy the net gives up at the biggest pots it spends at the many small ones, which is where the
            exact grader finds the exploitable play.
          </p>
          <p>
            Three 16,000-step arms at 5e-4 differ only in k and saw the same hands. At one million hands they grade{' '}
            {p100(pot(0).exact[100])} (raw), {p100(pot(0.5).exact[100])} (÷ √pot) and {p100(pot(1).exact[100])} (÷ pot);
            at two million, {p100(pot(0).exact[200])}, {p100(pot(0.5).exact[200])} and {p100(pot(1).exact[200])}. The
            only yardstick for a run-to-run difference is the two raw arms, at 5e-4 and at 1e-3, and it is a coarse one,
            since they differ in learning rate and not only in seed: {p100(rawGap(100))} apart at one million hands and{' '}
            {p100(rawGap(200))} at two. The √pot arm's gain over raw at two million, {p100(pot(0).exact[200] - pot(0.5).exact[200])},
            is about {((pot(0).exact[200] - pot(0.5).exact[200]) / rawGap(200)).toFixed(1)} times that gap at the same
            mark. The two weightings tie at two million hands, {p100(Math.abs(pot(0.5).exact[200] - pot(1).exact[200]))}{' '}
            apart, inside the gap; √pot leads at one million by {p100(pot(1).exact[100] - pot(0.5).exact[100])}, barely
            more than the raw arms' gap there; its current iterate is far better at two million (
            {p100(pot(0.5).current[200])} against {p100(pot(1).current[200])}); and it is the weighting the chip-cost
            argument picks. On that, k = ½ is the default for new runs. Early, at a hundred thousand hands, the √pot arm
            is worse than raw ({p100(sqrtPot10)} against {p100(markOf(BEST, 10).exact.exploitability)}; the ÷ pot arm was
            not graded there), as a longer fit was in the previous section; arms are compared from a million hands.
          </p>
          <Figure
            n={13}
            caption={
              <>
                ÷ √pot grades {p100(pot(0.5).exact[200])} at two million hands against raw {p100(pot(0).exact[200])},{' '}
                {cut(pot(0).exact[200], pot(0.5).exact[200])} less, and ÷ pot ties √pot at {p100(pot(1).exact[200])};
                both are already under the raw arm's last grade, {p100(best500.exact.exploitability)} at five million. Same
                net, same refit, same hands; only the targets' scale changes. The dashed ivory line is the raw arm at the
                full learning rate, the only yardstick for a run-to-run difference and a coarse one, since its learning
                rate differs too. Hover or arrow a mark to read every arm's grade there in the readout.
              </>
            }
            source="figures/rung4/sweep_256x4-s16000-lr5e-4.json, sweep_256x4-s16000.json, pot_0.5-256x4-s16000-lr5e-4.json and pot_1-256x4-s16000-lr5e-4.json · the exact average at every graded mark; chips per hundred hands, both axes log; one seed each"
            label="Exact average of the regret targets over pot^k against hands per seat"
          >
            <PotChart />
          </Figure>
          <p>
            Two confounds are open. The refit does not normalise its targets (each row is weighted by t over the
            batch's mean t and nothing else), so dividing them by up to √52 ≈ 7.2 also shrinks the residuals, and with
            them the gradient norms the clip of 1.0 sees: Adam's update is scale-free, the clip is not and acts before
            Adam sees the gradient, so it bites less often in the pot arms than in the raw arm, and part of the gain may be a changed clipping regime rather than
            the reweighting; a raw arm with a looser clip would separate the two and has not been run. And the raw arm
            at the full rate is its own single-seed gain over 5e-4 ({p100(pot(0, 0.001).exact[200])} against{' '}
            {p100(pot(0).exact[200])} at two million hands), so √pot at 1e-3, the pairing the default config now runs,
            has not been graded. Whether the pot arms flatten where the raw arm did, {p100(best500.exact.exploitability)}{' '}
            at five million hands, is the next thing to grade; both stopped at two million.
          </p>
        </Chapter>

        <Chapter id="ch-07" title="The exact average wins at every mark, and the policy net's stall was its loss">
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
            n={14}
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
            n={15}
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
            n={16}
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

        <Chapter id="ch-08" title={<>Per-sample DCFR and CFR+ lose to linear by {KEY_NUMBERS.ruleScreenRatio.toFixed(0)} times and get worse with training, because there is no total to test</>}>
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
            matching plays every positive cell. That fits the rising curve; no ablation has isolated it as the only
            cause. A small worked case: two actions with samples b = (+1, −3, +1, −3) and
            c = (+0.5, +0.5, +0.5, +0.5). Per-sample clipping fits both to +0.5, so they are played equally; CFR+ on the
            totals would floor b at zero and play c. A per-sample rule inflates noisy actions, and noisy actions are what
            sampling produces.
          </p>
          <Figure
            n={17}
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
            The rule that falls out of the race's DCFR result and this screen together: under sampling a regret discount must be blind to
            the regret's sign and carry no clock of its own. Linear weighting satisfies both. A sampled DCFR needs a visit
            clock; a neural CFR+ needs a total to test, which a net that fits a mean does not have.
          </p>
        </Chapter>

        <Chapter id="ch-09" title="What is and is not settled">
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
                <td>the pot arms</td>
                <td>
                  One seed each, stopped at two million hands. √pot with the full learning rate, the pairing new runs
                  default to, is ungraded, and part of the gain may be the gradient clip biting less on smaller targets.
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
