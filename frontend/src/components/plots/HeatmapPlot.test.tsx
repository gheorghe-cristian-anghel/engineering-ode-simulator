import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { HeatmapPlot } from './HeatmapPlot'

describe('HeatmapPlot', () => {
  afterEach(() => cleanup())

  it('announces field dimensions, coordinate extents, and units', () => {
    render(
      <HeatmapPlot
        field={[[0, 0.5], [1, 0.25]]}
        x={[0, 2]}
        y={[-1, 1]}
        unit="K"
        title="Final temperature"
      />,
    )

    expect(screen.getByRole('img', { name: /final temperature.*2 by 2.*x 0.00 to 2.00 m.*y -1.00 to 1.00 m.*K/i })).toBeInTheDocument()
    expect(screen.getByText(/x \(m\): 0.00 to 2.00.*y \(m\): -1.00 to 1.00/i)).toBeInTheDocument()
    expect(screen.getAllByText(/y \(m\)/i)).not.toHaveLength(0)
    expect(screen.getByRole('complementary', { name: /color scale.*K/i })).toBeInTheDocument()
  })

  it('uses the configured scalar color-scale label in its accessible description', () => {
    render(
      <HeatmapPlot
        field={[[1]]}
        x={[2]}
        y={[3]}
        unit="Pa"
        title="Pressure field"
        colorScaleLabel="Pressure"
      />,
    )

    expect(screen.getByRole('img', { name: /pressure ranges from 1.00 to 1.00 Pa/i })).toBeInTheDocument()
    expect(screen.queryByRole('img', { name: /temperature ranges/i })).not.toBeInTheDocument()
  })

  it('shows numeric minimum and maximum endpoints for both coordinate axes', () => {
    render(<HeatmapPlot field={[[0, 1], [2, 3]]} x={[10, 20]} y={[-5, 15]} unit="K" title="Temperature" />)

    expect(screen.getByText(/x \(m\): 10.00 to 20.00/i)).toBeInTheDocument()
    expect(screen.getByText(/y \(m\): -5.00 to 15.00/i)).toBeInTheDocument()
  })

  it('rejects grids exceeding the 65,536-cell canvas safety limit before allocating a canvas image', () => {
    const getContext = vi.spyOn(HTMLCanvasElement.prototype, 'getContext')
    const width = 257
    const height = 256
    const field = Array.from({ length: height }, (_, row) => Array.from({ length: width }, (_, column) => row + column))

    render(<HeatmapPlot field={field} x={Array.from({ length: width }, (_, index) => index)} y={Array.from({ length: height }, (_, index) => index)} unit="K" title="Oversized field" />)

    expect(screen.getByRole('status')).toHaveTextContent(/exceeds the 65.?536-cell rendering limit/i)
    expect(getContext).not.toHaveBeenCalled()
  })

  it('renders the first field row at the lower y bound', () => {
    const pixels = new Uint8ClampedArray(8)
    const context = { createImageData: () => ({ data: pixels }), putImageData: () => undefined }
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as unknown as CanvasRenderingContext2D)

    render(<HeatmapPlot field={[[0], [1]]} x={[0]} y={[0, 1]} unit="K" title="Temperature" />)

    expect(pixels[0]).toBe(238)
    expect(screen.getByText(/y \(m\): 0.00 to 1.00/i)).toBeInTheDocument()
  })
})
