import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { HeatmapCanvas } from './HeatmapCanvas'

describe('HeatmapCanvas', () => {
  afterEach(() => cleanup())
  it('provides an accessible summary and final-temperature range', () => {
    render(<HeatmapCanvas field={[[0, 0.5], [1, 0.25]]} x={[0, 1]} y={[0, 1]} unit="arbitrary" />)

    expect(screen.getByRole('img', { name: /final heat field/i })).toBeInTheDocument()
    expect(screen.getByLabelText(/Temperature range 0.000 to 1.000 arbitrary/)).toBeInTheDocument()
  })

  it('renders a constant field without a divide-by-zero color scale', () => {
    render(<HeatmapCanvas field={[[2, 2], [2, 2]]} x={[0, 1]} y={[0, 1]} unit="K" />)

    expect(screen.getByLabelText(/Temperature range 2.000 to 2.000 K/)).toBeInTheDocument()
  })

  it('keeps the accessible field summary available when canvas is unsupported', () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)
    render(<HeatmapCanvas field={[[2]]} x={[0]} y={[0]} unit="K" />)

    expect(screen.getByRole('img', { name: /Temperature range ranges from 2.000 to 2.000/i })).toBeInTheDocument()
  })

  it('draws low y values at the bottom, matching the solver coordinate system', () => {
    const pixels = new Uint8ClampedArray(8)
    const context = { createImageData: () => ({ data: pixels }), putImageData: () => undefined }
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as unknown as CanvasRenderingContext2D)

    render(<HeatmapCanvas field={[[0], [1]]} x={[0]} y={[0, 1]} unit="K" />)

    expect(pixels[0]).toBe(238)
  })
})
