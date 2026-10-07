import { describe, expect, it } from 'vitest'

import { PackBoundary } from './PackBoundary'

describe('the boundary under the trainer', () => {
  it('turns a thrown error into the message it shows, and shows nothing until one is thrown', () => {
    // No DOM here, so the state transition is pinned rather than the render.
    expect(PackBoundary.getDerivedStateFromError(new Error("the deal pack has no vector at round 1 'x'"))).toEqual({
      message: "the deal pack has no vector at round 1 'x'",
    })
    expect(new PackBoundary({ children: null }).state).toEqual({ message: null })
  })
})
