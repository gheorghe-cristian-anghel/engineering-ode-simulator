import { useEffect, useMemo, useRef } from 'react'

interface HeatmapCanvasProps {
  field: number[][]
  x: number[]
  y: number[]
  unit: string
}

export function HeatmapCanvas({ field, x, y, unit }: HeatmapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const values = useMemo(() => field.flat(), [field])
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || field.length === 0 || field[0].length === 0) return
    const height = field.length
    const width = field[0].length
    canvas.width = width
    canvas.height = height
    const context = canvas.getContext('2d')
    if (!context) return
    const image = context.createImageData(width, height)
    field.forEach((_row, rowIndex) => field[height - rowIndex - 1].forEach((value, columnIndex) => {
      const pixel = (rowIndex * width + columnIndex) * 4
      const [red, green, blue] = heatColor(normalize(value, minimum, maximum))
      image.data[pixel] = red
      image.data[pixel + 1] = green
      image.data[pixel + 2] = blue
      image.data[pixel + 3] = 255
    }))
    context.putImageData(image, 0, 0)
  }, [field, maximum, minimum])

  return (
    <figure className="heatmap-figure">
      <canvas
        ref={canvasRef}
        className="heatmap-canvas"
        role="img"
        aria-label={`Final heat field from x ${x[0]?.toFixed(2)} to ${x.at(-1)?.toFixed(2)} and y ${y[0]?.toFixed(2)} to ${y.at(-1)?.toFixed(2)} ${unit}. Temperature ranges from ${minimum.toFixed(3)} to ${maximum.toFixed(3)}.`}
      />
      <figcaption className="heatmap-caption">
        <span>Final temperature field</span>
        <span className="heatmap-legend" aria-label={`Temperature range ${minimum.toFixed(3)} to ${maximum.toFixed(3)} ${unit}`}>
          <i aria-hidden="true" />
          {minimum.toFixed(3)} — {maximum.toFixed(3)} {unit}
        </span>
      </figcaption>
    </figure>
  )
}

function normalize(value: number, minimum: number, maximum: number) {
  return maximum === minimum ? 0.5 : (value - minimum) / (maximum - minimum)
}

function heatColor(value: number): [number, number, number] {
  const stops: Array<[number, number, number]> = [[11, 41, 64], [0, 153, 177], [255, 213, 79], [238, 91, 59]]
  const scaled = Math.min(stops.length - 1, Math.max(0, value * (stops.length - 1)))
  const lower = Math.floor(scaled)
  const upper = Math.ceil(scaled)
  const fraction = scaled - lower
  return stops[lower].map((channel, index) => Math.round(channel + (stops[upper][index] - channel) * fraction)) as [number, number, number]
}
