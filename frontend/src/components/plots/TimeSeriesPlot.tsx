import { useMemo, useState, type PointerEvent } from 'react'
import type { AxisDefinition, PlotDomain, TimeSeries } from './plotTypes'
import { buildPolyline, resolveDomain } from './plotTransforms'
import { PlotPanel } from './PlotPanel'

interface TimeSeriesPlotProps {
  readonly title: string
  readonly series: readonly TimeSeries[]
  readonly xAxis: AxisDefinition
  readonly yAxis: AxisDefinition
  readonly maxVertices?: number
}

interface Readout { readonly name: string; readonly x: number; readonly y: number; readonly unit?: string }

const VIEW_WIDTH = 720
const VIEW_HEIGHT = 310
const PADDING = { left: 62, right: 22, top: 20, bottom: 48 }

export function TimeSeriesPlot({ title, series, xAxis, yAxis, maxVertices = 800 }: TimeSeriesPlotProps) {
  const [readout, setReadout] = useState<Readout | null>(null)
  const prepared = useMemo(() => series.map((source) => ({ source, segments: buildPolyline(source.x, source.y, maxVertices) })).filter(({ segments }) => segments.length > 0), [series, maxVertices])
  const xDomain = useMemo(() => resolveDomain(series.flatMap((item) => item.x)), [series])
  const yDomain = useMemo(() => resolveDomain(series.flatMap((item) => item.y)), [series])
  if (!xDomain || !yDomain || prepared.length === 0) return <PlotPanel title={title}><p className="plot-panel__empty" role="status">No finite samples are available for this plot.</p></PlotPanel>

  const plotWidth = VIEW_WIDTH - PADDING.left - PADDING.right
  const plotHeight = VIEW_HEIGHT - PADDING.top - PADDING.bottom
  const safeXDomain = xDomain
  const safeYDomain = yDomain
  const xToSvg = (value: number) => PADDING.left + normalize(value, safeXDomain) * plotWidth
  const yToSvg = (value: number) => PADDING.top + (1 - normalize(value, safeYDomain)) * plotHeight
  const ariaLabel = `${title}. ${axisText(xAxis)} horizontal axis. ${axisText(yAxis)} vertical axis. ${prepared.map(({ source }) => source.name).join(', ')}.`

  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect()
    const width = rect.width || VIEW_WIDTH
    const left = rect.width ? rect.left : 0
    const x = safeXDomain.min + ((event.clientX - left) / width) * (safeXDomain.max - safeXDomain.min)
    const candidates = prepared.flatMap(({ source }) => source.x.map((sampleX, index) => ({ source, x: sampleX, y: source.y[index] })).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y)))
    const nearest = candidates.reduce<typeof candidates[number] | null>((best, point) => !best || Math.abs(point.x - x) < Math.abs(best.x - x) ? point : best, null)
    setReadout(nearest ? { name: nearest.source.name, x: nearest.x, y: nearest.y, unit: yAxis.unit } : null)
  }

  return <PlotPanel title={title} description={`${axisText(yAxis)} vs ${axisText(xAxis)}`}>
    <figure className="scientific-timeseries">
      <svg className="scientific-timeseries__svg" role="img" aria-label={ariaLabel} viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`} onPointerMove={onPointerMove} onPointerLeave={() => setReadout(null)}>
        <line x1={PADDING.left} x2={VIEW_WIDTH - PADDING.right} y1={VIEW_HEIGHT - PADDING.bottom} y2={VIEW_HEIGHT - PADDING.bottom} className="scientific-timeseries__axis" />
        <line x1={PADDING.left} x2={PADDING.left} y1={PADDING.top} y2={VIEW_HEIGHT - PADDING.bottom} className="scientific-timeseries__axis" />
        <text x={VIEW_WIDTH / 2} y={VIEW_HEIGHT - 10} textAnchor="middle">{axisText(xAxis)}</text>
        <text x={18} y={VIEW_HEIGHT / 2} textAnchor="middle" transform={`rotate(-90 18 ${VIEW_HEIGHT / 2})`}>{axisText(yAxis)}</text>
        <text x={PADDING.left} y={VIEW_HEIGHT - PADDING.bottom + 17}>{format(xDomain.min)}</text><text x={VIEW_WIDTH - PADDING.right} y={VIEW_HEIGHT - PADDING.bottom + 17} textAnchor="end">{format(xDomain.max)}</text>
        <text x={PADDING.left - 8} y={PADDING.top + 4} textAnchor="end">{format(yDomain.max)}</text><text x={PADDING.left - 8} y={VIEW_HEIGHT - PADDING.bottom} textAnchor="end">{format(yDomain.min)}</text>
        {prepared.map(({ source, segments }) => segments.map((segment, index) => <path key={`${source.id}-${index}`} data-testid={`series-${source.id}`} d={pathData(segment, xToSvg, yToSvg)} fill="none" stroke={source.color} strokeWidth="2.5" strokeDasharray={source.role === 'reference' ? '7 5' : undefined} vectorEffect="non-scaling-stroke" />))}
      </svg>
      <figcaption>{readout ? `${readout.name}: ${format(readout.y)} ${readout.unit ?? ''} at ${format(readout.x)} ${xAxis.unit ?? ''}` : 'Move across the plot to inspect the nearest finite sample.'}</figcaption>
      <ul className="scientific-timeseries__legend" aria-label={`${title} legend`}>{prepared.map(({ source }) => <li key={source.id}><i style={{ backgroundColor: source.color, borderTop: source.role === 'reference' ? `2px dashed ${source.color}` : undefined }} />{source.name}</li>)}</ul>
    </figure>
  </PlotPanel>
}

function axisText(axis: AxisDefinition): string { return axis.unit ? `${axis.label} (${axis.unit})` : axis.label }
function normalize(value: number, domain: PlotDomain): number { return domain.max === domain.min ? 0.5 : (value - domain.min) / (domain.max - domain.min) }
function pathData(points: readonly { x: number; y: number }[], xToSvg: (value: number) => number, yToSvg: (value: number) => number): string { return points.map((point, index) => `${index === 0 ? 'M' : 'L'}${xToSvg(point.x).toFixed(2)} ${yToSvg(point.y).toFixed(2)}`).join(' ') }
function format(value: number): string { return value.toFixed(Math.abs(value) >= 100 ? 0 : 2) }
