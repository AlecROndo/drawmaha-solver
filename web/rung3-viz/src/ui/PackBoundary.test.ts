import { afterEach, describe, expect, it, vi } from 'vitest'

import { PackBoundary } from './PackBoundary'

// No DOM here, so the boundary's two hooks are pinned directly rather than
// through a render: the state it derives from a throw, and what it logs.

afterEach(() => vi.restoreAllMocks())

describe('the boundary under the trainer', () => {
  it('turns a thrown error into the message it shows, and shows nothing until one is thrown', () => {
    expect(PackBoundary.getDerivedStateFromError(new Error("the deal pack has no vector at round 1 'x'"))).toEqual({
      message: "the deal pack has no vector at round 1 'x'",
    })
    expect(new PackBoundary({ children: null }).state).toEqual({ message: null })
  })

  it('logs the error and the component stack, so a production bug report has both', () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {})
    const error = new Error('the deal pack has no vector at round 2 after xx/nn at \'\'')
    new PackBoundary({ children: null }).componentDidCatch(error, { componentStack: '\n    at PlayPanel\n    at PackBoundary' })
    expect(logged).toHaveBeenCalledWith(error, '\n    at PlayPanel\n    at PackBoundary')
  })
})
