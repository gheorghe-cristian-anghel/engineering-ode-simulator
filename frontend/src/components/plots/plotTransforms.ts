import type { DomainOptions, PlotBounds, PlotDomain, PlotPoint, PolylineSegment } from './plotTypes'

const DEFAULT_MAX_POLYLINE_POINTS = 800

export function resolveDomain(values: Iterable<number>, options: DomainOptions = {}): PlotDomain | null {
  if (!isValidLimit(options.min) || !isValidLimit(options.max)) return null
  if (options.min !== undefined && options.max !== undefined && options.min > options.max) return null

  let observedMin = Infinity
  let observedMax = -Infinity
  for (const value of values) {
    if (!Number.isFinite(value)) continue
    observedMin = Math.min(observedMin, value)
    observedMax = Math.max(observedMax, value)
  }

  const min = options.min ?? observedMin
  const max = options.max ?? observedMax
  if (!Number.isFinite(min) || !Number.isFinite(max) || min > max) return null

  if (!options.symmetric) return { min, max }
  const magnitude = Math.max(Math.abs(min), Math.abs(max))
  return { min: -magnitude, max: magnitude }
}

/**
 * Produces at most `maxPoints` finite pairs, sampled by source index. Invalid
 * pairs split the returned line into separate drawable segments. The original
 * arrays are never mutated and dense inputs never create a full copy.
 */
export function buildPolyline(
  xValues: readonly number[],
  yValues: readonly number[],
  maxPoints = 800,
): readonly PolylineSegment[] {
  const length = Math.min(xValues.length, yValues.length)
  const pointLimit = resolvePointLimit(maxPoints)
  let finiteCount = 0

  for (let index = 0; index < length; index += 1) {
    if (isFinitePair(xValues[index], yValues[index])) finiteCount += 1
  }
  if (finiteCount === 0) return []

  const outputCount = Math.min(finiteCount, pointLimit)
  const segments: PlotPoint[][] = []
  let activeSegment: PlotPoint[] | undefined
  let pointCount = 0
  let finiteIndex = 0

  for (let index = 0; index < length && pointCount < outputCount; index += 1) {
    const x = xValues[index]
    const y = yValues[index]
    if (!isFinitePair(x, y)) {
      activeSegment = undefined
      continue
    }

    const targetIndex = Math.round((pointCount * (finiteCount - 1)) / (outputCount - 1 || 1))
    if (finiteIndex === targetIndex) {
      if (!activeSegment) {
        activeSegment = []
        segments.push(activeSegment)
      }
      activeSegment.push({ x, y })
      pointCount += 1
    }
    finiteIndex += 1
  }

  return segments
}

/**
 * Finds padded bounds with the same x and y world-unit span. This lets SVG
 * preserve physical geometry independently of the viewport aspect ratio.
 */
export function getEqualAspectBounds(
  points: readonly PlotPoint[],
  marginFraction = 0.1,
): PlotBounds | null {
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity

  for (const point of points) {
    if (!isFinitePair(point.x, point.y)) continue
    minX = Math.min(minX, point.x)
    maxX = Math.max(maxX, point.x)
    minY = Math.min(minY, point.y)
    maxY = Math.max(maxY, point.y)
  }
  if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) {
    return null
  }

  const span = Math.max(maxX - minX, maxY - minY, 1)
  const margin = Math.max(0, Number.isFinite(marginFraction) ? marginFraction : 0) * span
  const paddedSpan = span + margin * 2
  const centerX = (minX + maxX) / 2
  const centerY = (minY + maxY) / 2
  const halfSpan = paddedSpan / 2

  const bounds = {
    minX: centerX - halfSpan,
    maxX: centerX + halfSpan,
    minY: centerY - halfSpan,
    maxY: centerY + halfSpan,
  }

  return Object.values(bounds).every(Number.isFinite) ? bounds : null
}

function isValidLimit(value: number | undefined): boolean {
  return value === undefined || Number.isFinite(value)
}

function resolvePointLimit(maxPoints: number): number {
  return Number.isFinite(maxPoints) ? Math.max(1, Math.floor(maxPoints)) : DEFAULT_MAX_POLYLINE_POINTS
}

function isFinitePair(x: number, y: number): boolean {
  return Number.isFinite(x) && Number.isFinite(y)
}
