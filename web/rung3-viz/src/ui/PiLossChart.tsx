import { BEST, PI_REFERENCE, PI_SWEEP, markOf, piArm, type Loss } from '../deep'
import { perHundred } from '../research'
import { fmtHands, linScale, logScale } from './scales'

/**
 * The policy-net loss sweep, two panels. Left: Π refit on the mark-500
 * memory under each loss against the refit's step count. Right: the same
 * cross-entropy refit on the memory of marks 100, 200 and 500 beside the
 * exact average and the run's own Π, which is what pins the stall on the
 * fit. Chips per hundred hands.
 */

const W = 860
const H = 320
const PANEL = { l: 54, r: 24, t: 24, b: 40 }
const HALF = W / 2
const Y_LO = 10.5
const Y_HI = 14.2

const LOSS_INK: Record<Loss, string> = { ce: 'var(--loss-ce)', mse: 'var(--loss-mse)', is: 'var(--loss-is)', kl: 'var(--loss-ce)' }
const LOSS_NAME: Record<Loss, string> = { ce: 'cross-entropy, softmax head', mse: 'squared error, linear head', is: 'Itakura–Saito, softmax head', kl: 'KL divergence' }

export function PiLossChart() {
  // left: steps, the three loss families at mark 500, 256 wide, 4 deep
  const atSteps = PI_SWEEP.filter((a) => a.t === 500 && a.width === 256 && a.depth === 4 && a.loss !== 'kl' && !(a.loss === 'mse' && a.head === 'softmax'))
  const losses: Loss[] = ['ce', 'mse', 'is']
  const xs = logScale(16_000, 256_000, PANEL.l, HALF - PANEL.r)
  const ys = linScale(Y_LO, Y_HI, H - PANEL.b, PANEL.t)
  const exactRef = perHundred(PI_REFERENCE.exact)
  // right: the marks
  const marks = [100, 200, 500]
  const xm = logScale(1e6, 5e6, HALF + PANEL.l, W - PANEL.r)
  const ym = linScale(Y_LO, 16.5, H - PANEL.b, PANEL.t)
  const exactAt = marks.map((t) => ({ t, v: perHundred(markOf(BEST, t).exact.exploitability) }))
  const runPiAt = marks.map((t) => ({ t, v: perHundred(markOf(BEST, t).pi.exploitability) }))
  const ceAt = marks.map((t) => ({ t, v: perHundred(piArm(t === 500 ? 'ce-softmax-64k-cos' : `ce-softmax-64k-cos@t${t}`).exploitability) }))
  const mseAt = [200, 500].map((t) => ({ t, v: perHundred(piArm(t === 500 ? 'mse-linear-64k-cos' : `mse-linear-64k-cos@t${t}`).exploitability) }))
  const hands = (t: number) => markOf(BEST, t).hands
  const line = (pts: { t: number; v: number }[]) => pts.map((p, i) => `${i ? 'L' : 'M'} ${xm(hands(p.t)).toFixed(1)} ${ym(p.v).toFixed(1)}`).join(' ')

  return (
    <>
      <div className="legend">
        {losses.map((l) => (
          <span key={l} className="item">
            <span className="swatch" style={{ background: LOSS_INK[l] }} />
            {LOSS_NAME[l]}
          </span>
        ))}
        <span className="item">
          <span className="swatch" style={{ background: 'var(--ink-3)' }} />
          the run's Π (squared error, 16k)
        </span>
        <span className="item">
          <span className="swatch" style={{ background: 'repeating-linear-gradient(90deg, var(--ink) 0 4px, transparent 4px 7px)' }} />
          the exact average
        </span>
      </div>
      <svg className="chart piloss" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Policy-net refits by loss and step count, and across marks">
        {/* left panel */}
        <text x={PANEL.l} y={PANEL.t - 8} fontSize="10.5" fill="var(--ink-3)">
          Π refit on the 5M-hand memory, by Adam steps
        </text>
        {[11, 12, 13, 14].map((v) => (
          <g key={v}>
            <line x1={PANEL.l} x2={HALF - PANEL.r} y1={ys(v)} y2={ys(v)} stroke="var(--grid)" />
            <text x={PANEL.l - 8} y={ys(v) + 3.5} textAnchor="end" fontSize="11" fill="var(--ink-3)">
              {v}
            </text>
          </g>
        ))}
        {[16_000, 64_000, 128_000, 256_000].map((s) => (
          <text key={s} x={xs(s)} y={H - 16} textAnchor="middle" fontSize="11" fill="var(--ink-3)">
            {s / 1000}k
          </text>
        ))}
        <line x1={PANEL.l} x2={HALF - PANEL.r} y1={ys(exactRef)} y2={ys(exactRef)} stroke="var(--ink)" strokeDasharray="4 4" strokeOpacity={0.7} />
        <text x={PANEL.l + 4} y={ys(exactRef) - 5} fontSize="10.5" fill="var(--ink)">
          exact average {exactRef.toFixed(1)}
        </text>
        {losses.map((l) => {
          const pts = atSteps.filter((a) => a.loss === l).sort((a, b) => a.steps - b.steps)
          return (
            <g key={l}>
              <path d={pts.map((a, i) => `${i ? 'L' : 'M'} ${xs(a.steps).toFixed(1)} ${ys(perHundred(a.exploitability)).toFixed(1)}`).join(' ')} fill="none" stroke={LOSS_INK[l]} strokeWidth={l === 'ce' ? 2.2 : 1.5} strokeOpacity={l === 'is' ? 0.8 : 1} />
              {pts.map((a) => (
                <g key={a.id}>
                  <circle cx={xs(a.steps)} cy={ys(perHundred(a.exploitability))} r={3} fill={LOSS_INK[l]} />
                  {l !== 'is' && (
                    <text x={xs(a.steps)} y={ys(perHundred(a.exploitability)) + (l === 'ce' ? 16 : -8)} textAnchor="middle" fontSize="10.5" fill={LOSS_INK[l]}>
                      {perHundred(a.exploitability).toFixed(2)}
                    </text>
                  )}
                </g>
              ))}
            </g>
          )
        })}
        {/* right panel */}
        <text x={HALF + PANEL.l} y={PANEL.t - 8} fontSize="10.5" fill="var(--ink-3)">
          each mark's memory: exact, the run's Π, refits
        </text>
        {[11, 12, 13, 14, 15, 16].map((v) => (
          <g key={v}>
            <line x1={HALF + PANEL.l} x2={W - PANEL.r} y1={ym(v)} y2={ym(v)} stroke="var(--grid)" />
            <text x={HALF + PANEL.l - 8} y={ym(v) + 3.5} textAnchor="end" fontSize="11" fill="var(--ink-3)">
              {v}
            </text>
          </g>
        ))}
        {marks.map((t) => (
          <text key={t} x={xm(hands(t))} y={H - 16} textAnchor="middle" fontSize="11" fill="var(--ink-3)">
            {fmtHands(hands(t))}
          </text>
        ))}
        <path d={line(exactAt)} fill="none" stroke="var(--ink)" strokeWidth={1.6} strokeDasharray="4 4" />
        <path d={line(runPiAt)} fill="none" stroke="var(--ink-3)" strokeWidth={1.6} />
        <path d={line(ceAt)} fill="none" stroke="var(--loss-ce)" strokeWidth={2.2} />
        <path d={line(mseAt)} fill="none" stroke="var(--loss-mse)" strokeWidth={1.4} strokeDasharray="2 4" />
        {[
          { pts: exactAt, ink: 'var(--ink)', word: 'exact', dy: 26 },
          { pts: runPiAt, ink: 'var(--ink-3)', word: "the run's Π", dy: -6 },
          { pts: ceAt, ink: 'var(--loss-ce)', word: 'CE refit', dy: 14 },
          { pts: mseAt, ink: 'var(--loss-mse)', word: 'MSE refit', dy: -10 },
        ].map(({ pts, ink, word, dy }) => (
          <g key={word}>
            {pts.map((p) => (
              <circle key={p.t} cx={xm(hands(p.t))} cy={ym(p.v)} r={3} fill={ink} />
            ))}
            <text x={xm(hands(pts.at(-1)!.t)) - 6} y={ym(pts.at(-1)!.v) + dy} textAnchor="end" fontSize="10.5" fill={ink}>
              {word} {pts.at(-1)!.v.toFixed(1)}
            </text>
          </g>
        ))}
      </svg>
    </>
  )
}
