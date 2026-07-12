import { HeatmapPlot } from './plots/HeatmapPlot'

interface HeatmapCanvasProps {
  field: number[][]
  x: number[]
  y: number[]
  unit: string
}

/** @deprecated Use HeatmapPlot for new scientific field visualizations. */
export function HeatmapCanvas({ field, x, y, unit }: HeatmapCanvasProps) {
  return <HeatmapPlot field={field} x={x} y={y} unit={unit} title="Final heat field" precision={3} colorScaleLabel="Temperature range" />
}
