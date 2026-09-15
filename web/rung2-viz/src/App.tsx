import { IdentityRail, LadderLine } from './ui/site'

export default function App() {
  return (
    <>
      <LadderLine here={2} />
      <div className="shell">
        <IdentityRail now="Rung 2 · Leduc poker" next="Rung 3 · mini-Drawmaha" />
        <main>
          <p className="stop">Rung 2 · Leduc poker</p>
          <h2 className="big">The action timeline.</h2>
        </main>
      </div>
    </>
  )
}
