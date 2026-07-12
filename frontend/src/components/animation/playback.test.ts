import { describe, expect, it } from 'vitest'
import { advanceFrame, clampFrame, resetFrame } from './playback'

describe('playback frame helpers', () => {
  it('clamps timeline positions and resets to the first frame', () => {
    expect(clampFrame(-2, 4)).toBe(0)
    expect(clampFrame(9, 4)).toBe(4)
    expect(resetFrame()).toBe(0)
  })

  it('drops ahead to the correct frame after a slow browser frame', () => {
    expect(advanceFrame(1, 750, 4, 2, 200)).toBe(4)
  })
})
