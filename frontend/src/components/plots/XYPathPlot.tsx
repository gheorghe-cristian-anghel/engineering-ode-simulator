import { useMemo } from 'react'
import type { AxisDefinition, PlotPoint, XYPath } from './plotTypes'
import { getEqualAspectBounds } from './plotTransforms'
import { PlotPanel } from './PlotPanel'

interface ObstacleGeometry {
  readonly center: PlotPoint
  readonly radius: number
  readonly influenceRadius: number
}

interface XYPathPlotProps {
  readonly title: string
  readonly paths: readonly XYPath[]
  readonly waypoints: readonly PlotPoint[]
  readonly obstacle?: ObstacleGeometry
  readonly xAxis: AxisDefinition
  readonly yAxis: AxisDefinition
  readonly maxVertices?: number
}

const VIEW_WIDTH = 620
const VIEW_HEIGHT = 360
const PLOT_SIZE = 274
const PLOT_LEFT = 80
const PLOT_TOP = 30

export function XYPathPlot({ title, paths, waypoints, obstacle, xAxis, yAxis, maxVertices = 800 }: XYPathPlotProps) {
  const finitePaths = useMemo(() => paths.map((path) => ({ ...path, points: decimate(path.points, maxVertices) })).filter((path) => path.points.length > 0), [paths, maxVertices])
  const bounds = useMemo(() => getEqualAspectBounds([
    ...finitePaths.flatMap((path) => path.points),
    ...waypoints,
    ...(obstacle ? [{ x: obstacle.center.x - obstacle.influenceRadius, y: obstacle.center.y - obstacle.influenceRadius }, { x: obstacle.center.x + obstacle.influenceRadius, y: obstacle.center.y + obstacle.influenceRadius }] : []),
  ]), [finitePaths, waypoints, obstacle])

  if (!bounds || finitePaths.length === 0) return <PlotPanel title={title}><p className="plot-panel__empty" role="status">No finite path points are available for this plot.</p></PlotPanel>

  const span = bounds.maxX - bounds.minX
  const project = (point: PlotPoint) => ({ x: PLOT_LEFT + ((point.x - bounds.minX) / span) * PLOT_SIZE, y: PLOT_TOP + ((bounds.maxY - point.y) / span) * PLOT_SIZE })
  const radius = (value: number) => (value / span) * PLOT_SIZE
  const actual = finitePaths.find((path) => path.role === 'actual') ?? finitePaths[finitePaths.length - 1]
  const start = actual.points[0]
  const end = actual.points[actual.points.length - 1]

  return <PlotPanel title={title} description={`${axisText(yAxis)} vs ${axisText(xAxis)}; equal x/y scale.`}>
    <figure className="scientific-xypath">
      <svg className="scientific-xypath__svg" role="img" aria-label={`${title}. ${axisText(xAxis)} horizontal axis. ${axisText(yAxis)} vertical axis.`} viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`} preserveAspectRatio="xMidYMid meet">
        <rect x={PLOT_LEFT} y={PLOT_TOP} width={PLOT_SIZE} height={PLOT_SIZE} className="scientific-xypath__frame" />
        <text x={PLOT_LEFT + PLOT_SIZE / 2} y={VIEW_HEIGHT - 14} textAnchor="middle">{axisText(xAxis)}</text>
        <text x={20} y={PLOT_TOP + PLOT_SIZE / 2} textAnchor="middle" transform={`rotate(-90 20 ${PLOT_TOP + PLOT_SIZE / 2})`}>{axisText(yAxis)}</text>
        <text x={PLOT_LEFT} y={PLOT_TOP + PLOT_SIZE + 18}>{format(bounds.minX)}</text>
        <text x={PLOT_LEFT + PLOT_SIZE} y={PLOT_TOP + PLOT_SIZE + 18} textAnchor="end">{format(bounds.maxX)}</text>
        <text x={PLOT_LEFT - 8} y={PLOT_TOP + 4} textAnchor="end">{format(bounds.maxY)}</text>
        <text x={PLOT_LEFT - 8} y={PLOT_TOP + PLOT_SIZE} textAnchor="end">{format(bounds.minY)}</text>
        {obstacle && Number.isFinite(obstacle.radius) && Number.isFinite(obstacle.influenceRadius) && <Obstacle obstacle={obstacle} project={project} radius={radius} />}
        {finitePaths.map((path) => <path key={path.id} data-testid={`path-${path.id}`} d={pathData(path.points, project)} fill="none" stroke={path.color} strokeWidth="2.5" strokeDasharray={path.role === 'reference' ? '7 5' : undefined} vectorEffect="non-scaling-stroke" />)}
        {waypoints.map((waypoint, index) => { const point = project(waypoint); return <g key={`${waypoint.x}-${waypoint.y}-${index}`} aria-label={`Waypoint ${index + 1}`}><rect x={point.x - 3} y={point.y - 3} width="6" height="6" fill="#111827" /><text x={point.x + 6} y={point.y - 6}>W{index + 1}</text></g> })}
        <Marker label="Start marker" point={project(start)} color="#16a34a" />
        <Marker label="End marker" point={project(end)} color="#dc2626" />
      </svg>
      <figcaption>Reference uses a dashed line; start, end, waypoints, and obstacle boundaries are labeled directly on the geometry.</figcaption>
      <ul className="scientific-xypath__legend" aria-label={`${title} legend`}>
        {finitePaths.map((path) => <li key={path.id}><i style={{ backgroundColor: path.color, borderTop: path.role === 'reference' ? `2px dashed ${path.color}` : undefined }} />{path.name}</li>)}
      </ul>
    </figure>
  </PlotPanel>
}

function Obstacle({ obstacle, project, radius }: { readonly obstacle: ObstacleGeometry; readonly project: (point: PlotPoint) => PlotPoint; readonly radius: (value: number) => number }) {
  const center = project(obstacle.center)
  return <g>
    <circle aria-label="Obstacle influence radius" cx={center.x} cy={center.y} r={radius(obstacle.influenceRadius)} className="scientific-xypath__influence" />
    <circle aria-label="Obstacle" cx={center.x} cy={center.y} r={radius(obstacle.radius)} className="scientific-xypath__obstacle" />
  </g>
}

function Marker({ label, point, color }: { readonly label: string; readonly point: PlotPoint; readonly color: string }) { return <circle aria-label={label} cx={point.x} cy={point.y} r="4.5" fill={color} stroke="white" strokeWidth="1.5" /> }
function decimate(points: readonly PlotPoint[], limit: number): readonly PlotPoint[] { const finite = points.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y)); if (finite.length <= limit) return finite; const stride = (finite.length - 1) / Math.max(1, Math.floor(limit) - 1); return Array.from({ length: Math.floor(limit) }, (_, index) => finite[Math.min(finite.length - 1, Math.round(index * stride))]) }
function pathData(points: readonly PlotPoint[], project: (point: PlotPoint) => PlotPoint): string { return points.map((point, index) => { const p = project(point); return `${index === 0 ? 'M' : 'L'}${p.x.toFixed(2)} ${p.y.toFixed(2)}` }).join(' ') }
function axisText(axis: AxisDefinition): string { return axis.unit ? `${axis.label} (${axis.unit})` : axis.label }
function format(value: number): string { return value.toFixed(Math.abs(value) >= 100 ? 0 : 2) }
