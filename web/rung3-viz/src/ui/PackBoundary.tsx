import { Component, type ErrorInfo, type ReactNode } from 'react'

import { PlayWindow } from './PlayPanel'

/**
 * The backstop under the trainer. `validateDeal` refuses a broken pack as it
 * loads, so `toMix` throwing mid-hand can only mean a node the grammar
 * reaches that the validator did not demand — a bug in this app, not a bad
 * deal. React unmounts the whole tree on an uncaught render error; with this
 * boundary the table alone is replaced by the message, in the trainer's own
 * window, and the rest of the page stays readable.
 */
export class PackBoundary extends Component<{ children: ReactNode }, { message: string | null }> {
  state: { message: string | null } = { message: null }

  static getDerivedStateFromError(error: Error): { message: string } {
    return { message: error.message }
  }

  /** React logs a caught error only in development; the panel asks for a bug report, so the stack must reach the console in production too. */
  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(error, info.componentStack)
  }

  render() {
    if (this.state.message === null) return this.props.children
    return (
      <PlayWindow k="01 · the table">
        <p className="playnote">
          The trainer stopped: {this.state.message}. The pack passed its checks as it loaded, so this is a spot the
          grammar reaches that the validator did not ask for — a bug worth reporting, not a bad deal. Reload to sit
          down again.
        </p>
      </PlayWindow>
    )
  }
}
