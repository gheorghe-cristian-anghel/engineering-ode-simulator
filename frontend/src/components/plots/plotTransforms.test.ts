import { describe, expect, it } from 'vitest'
import { buildPolyline, getEqualAspectBounds, resolveDomain } from './plotTransforms'

describe('resolveDomain', () => {
  it('honours a fixed domain and can make it symmetric around zero', () => {
    expect(resolveDomain([2, 4, Number.NaN], { min: 1, max: 5 })).toEqual({ min: 1, max: 5 })
    expect(resolveDomain([-2, 4], { symmetric: true })).toEqual({ min: -4, max: 4 })
  })

  it('ignores non-finite values and reports no domain without finite data', () => {
    expect(resolveDomain([Number.NaN, -Infinity, 3, Infinity, -1])).toEqual({ min: -1, max: 3 })
    expect(resolveDomain([Number.NaN, Infinity])).toBeNull()
  })
})

describe('buildPolyline', () => {
  it('filters invalid pairs and bounds dense paths while retaining endpoints', () => {
    const x = Array.from({ length: 10 }, (_, index) => index)
    const y = [0, 1, Number.NaN, 3, 4, 5, Infinity, 7, 8, 9]

    const points = buildPolyline(x, y, 4)

    const flattened = points.flat()

    expect(flattened).toHaveLength(4)
    expect(flattened[0]).toEqual({ x: 0, y: 0 })
    expect(flattened.at(-1)).toEqual({ x: 9, y: 9 })
    expect(flattened.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y))).toBe(true)
  })

  it('returns distinct segments around invalid samples so renderers do not bridge gaps', () => {
    expect(buildPolyline([0, 1, 2, 3, 4], [10, 11, Number.NaN, 13, 14], 5)).toEqual([
      [
        { x: 0, y: 10 },
        { x: 1, y: 11 },
      ],
      [
        { x: 3, y: 13 },
        { x: 4, y: 14 },
      ],
    ])
  })

  it('uses the bounded default cap for non-finite point limits', () => {
    const x = Array.from({ length: 1_000 }, (_, index) => index)
    const y = [...x]

    expect(buildPolyline(x, y, Number.NaN).flat()).toHaveLength(800)
    expect(buildPolyline(x, y, Infinity).flat()).toHaveLength(800)
  })
})

describe('getEqualAspectBounds', () => {
  it('adds margins and returns equal world-coordinate spans', () => {
    const bounds = getEqualAspectBounds([
      { x: 0, y: 0 },
      { x: 4, y: 1 },
    ])

    expect(bounds).not.toBeNull()
    expect(bounds?.minX).toBeCloseTo(-0.4)
    expect(bounds?.maxX).toBeCloseTo(4.4)
    expect(bounds?.minY).toBeCloseTo(-1.9)
    expect(bounds?.maxY).toBeCloseTo(2.9)
    if (!bounds) throw new Error('Expected finite points to produce bounds')
    expect(bounds.maxX - bounds.minX).toBeCloseTo(bounds.maxY - bounds.minY)
  })

  it('returns null when finite extremes or a finite margin overflow the drawable bounds', () => {
    expect(
      getEqualAspectBounds([
        { x: -Number.MAX_VALUE, y: 0 },
        { x: Number.MAX_VALUE, y: 0 },
      ]),
    ).toBeNull()
    expect(
      getEqualAspectBounds(
        [
          { x: 0, y: 0 },
          { x: 1, y: 1 },
        ],
        Number.MAX_VALUE,
      ),
    ).toBeNull()
  })
})
