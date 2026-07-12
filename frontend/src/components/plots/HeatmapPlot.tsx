import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react'
import type { DomainOptions, PlotDomain } from './plotTypes'
import { resolveDomain } from './plotTransforms'
import { PlotPanel } from './PlotPanel'

interface HeatmapPlotProps {
  readonly field: readonly (readonly number[])[]
  readonly x: readonly number[]
  readonly y: readonly number[]
  readonly unit: string
  readonly title: string
  readonly xLabel?: string
  readonly yLabel?: string
  readonly coordinateUnit?: string
  readonly colorDomain?: DomainOptions
  readonly precision?: number
  readonly colorScaleLabel?: string
}

interface GridShape { readonly width: number; readonly height: number }
interface Probe { readonly x: number; readonly y: number; readonly value: number }

const MAX_RENDERABLE_CELLS = 65_536

export function HeatmapPlot({ field, x, y, unit, title, xLabel = 'x', yLabel = 'y', coordinateUnit = 'm', colorDomain, precision = 2, colorScaleLabel = 'Color scale' }: HeatmapPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [probe, setProbe] = useState<Probe | null>(null)
  const shape = useMemo(() => getGridShape(field, x, y), [field, x, y])
  const domain = useMemo(() => shape && resolveDomain(fieldValues(field), colorDomain), [field, colorDomain, shape])
  const xDomain = getCoordinateDomain(x)
  const yDomain = getCoordinateDomain(y)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !shape || !domain) return
    canvas.width = shape.width
    canvas.height = shape.height
    const context = canvas.getContext('2d')
    if (!context) return
    const image = context.createImageData(shape.width, shape.height)
    for (let rasterRow = 0; rasterRow < shape.height; rasterRow += 1) {
      const sourceRow = shape.height - rasterRow - 1
      for (let column = 0; column < shape.width; column += 1) {
        const pixel = (rasterRow * shape.width + column) * 4
        const [red, green, blue] = heatColor(normalize(field[sourceRow][column], domain))
        image.data[pixel] = red; image.data[pixel + 1] = green; image.data[pixel + 2] = blue; image.data[pixel + 3] = 255
      }
    }
    context.putImageData(image, 0, 0)
  }, [domain, field, shape])

  if (!shape || !domain || !xDomain || !yDomain) return <PlotPanel title={title}><p className="plot-panel__empty" role="status">The field contains missing, non-finite, mismatched grid coordinates, or exceeds the {MAX_RENDERABLE_CELLS.toLocaleString()}-cell rendering limit.</p></PlotPanel>

  const ariaLabel = `${title}, ${shape.width} by ${shape.height} field, x ${format(xDomain.min, precision)} to ${format(xDomain.max, precision)} ${coordinateUnit}, y ${format(yDomain.min, precision)} to ${format(yDomain.max, precision)} ${coordinateUnit}, ${unit}. ${colorScaleLabel} ranges from ${format(domain.min, precision)} to ${format(domain.max, precision)} ${unit}.`
  return <PlotPanel title={title} description={`${shape.width} × ${shape.height} nodes · ${unit}`}><figure className="scientific-heatmap"><div className="scientific-heatmap__main"><span className="scientific-heatmap__y-axis">{yLabel} ({coordinateUnit})</span><canvas ref={canvasRef} className="scientific-heatmap__canvas" role="img" aria-label={ariaLabel} onPointerMove={(event) => setProbe(readProbe(event, shape, field, x, y))} onPointerLeave={() => setProbe(null)} /></div><aside className="scientific-heatmap__colorbar" aria-label={`${colorScaleLabel} ${format(domain.min, precision)} to ${format(domain.max, precision)} ${unit}`}><span>{format(domain.max, precision)}</span><i aria-hidden="true" /><span>{format(domain.min, precision)}</span><small>{unit}</small></aside><figcaption className="scientific-heatmap__caption"><span>{xLabel} ({coordinateUnit}): {format(xDomain.min, precision)} to {format(xDomain.max, precision)} · {yLabel} ({coordinateUnit}): {format(yDomain.min, precision)} to {format(yDomain.max, precision)}</span><span>{probe ? `${xLabel} ${format(probe.x, precision)}, ${yLabel} ${format(probe.y, precision)}: ${format(probe.value, precision)} ${unit}` : 'Move over the field to inspect a value.'}</span></figcaption></figure></PlotPanel>
}

function getGridShape(field: readonly (readonly number[])[], x: readonly number[], y: readonly number[]): GridShape | null {
  const height = field.length; const width = field[0]?.length ?? 0
  if (height === 0 || width === 0 || width * height > MAX_RENDERABLE_CELLS || x.length !== width || y.length !== height || !x.every(Number.isFinite) || !y.every(Number.isFinite)) return null
  return field.every((row) => row.length === width && row.every(Number.isFinite)) ? { width, height } : null
}
function* fieldValues(field: readonly (readonly number[])[]): Iterable<number> { for (const row of field) for (const value of row) yield value }
function getCoordinateDomain(values: readonly number[]): PlotDomain | null { return resolveDomain(values) }
function normalize(value: number, domain: PlotDomain): number { return domain.max === domain.min ? 0.5 : Math.min(1, Math.max(0, (value - domain.min) / (domain.max - domain.min))) }
function heatColor(value: number): [number, number, number] { const stops: readonly [number, number, number][] = [[11, 41, 64], [0, 153, 177], [255, 213, 79], [238, 91, 59]]; const scaled = value * (stops.length - 1); const lower = Math.floor(scaled); const upper = Math.ceil(scaled); const fraction = scaled - lower; return stops[lower].map((channel, index) => Math.round(channel + (stops[upper][index] - channel) * fraction)) as [number, number, number] }
function readProbe(event: PointerEvent<HTMLCanvasElement>, shape: GridShape, field: readonly (readonly number[])[], x: readonly number[], y: readonly number[]): Probe | null { const bounds = event.currentTarget.getBoundingClientRect(); if (bounds.width === 0 || bounds.height === 0) return null; const column = Math.min(shape.width - 1, Math.max(0, Math.floor(((event.clientX - bounds.left) / bounds.width) * shape.width))); const visualRow = Math.min(shape.height - 1, Math.max(0, Math.floor(((event.clientY - bounds.top) / bounds.height) * shape.height))); const row = shape.height - visualRow - 1; return { x: x[column], y: y[row], value: field[row][column] } }
function format(value: number, precision = 2): string { return value.toFixed(precision) }
