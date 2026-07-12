/** A finite inclusive numeric extent in data coordinates. */
export interface PlotDomain {
  readonly min: number
  readonly max: number
}

/** A point in a scientific plot's world-coordinate system. */
export interface PlotPoint {
  readonly x: number
  readonly y: number
}

/** A contiguous, drawable portion of a line; separate segments represent data gaps. */
export type PolylineSegment = readonly PlotPoint[]

/** Bounds whose dimensions are expressed in the same world units. */
export interface PlotBounds {
  readonly minX: number
  readonly maxX: number
  readonly minY: number
  readonly maxY: number
}

/** Optional domain overrides used by scalar-field and line plots. */
export interface DomainOptions {
  readonly min?: number
  readonly max?: number
  readonly symmetric?: boolean
}

export interface AxisDefinition {
  readonly label: string
  readonly unit?: string
}

export interface TimeSeries {
  readonly id: string
  readonly name: string
  readonly x: readonly number[]
  readonly y: readonly number[]
  readonly color: string
  readonly role?: 'actual' | 'reference' | 'measurement' | 'estimate' | 'error'
}

export interface XYPath {
  readonly id: string
  readonly name: string
  readonly points: readonly PlotPoint[]
  readonly color: string
  readonly role?: 'actual' | 'reference'
}
