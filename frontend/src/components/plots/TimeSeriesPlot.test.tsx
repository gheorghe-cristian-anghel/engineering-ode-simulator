import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { TimeSeriesPlot } from './TimeSeriesPlot'

describe('TimeSeriesPlot', () => {
  afterEach(() => cleanup())

  it('renders an external legend, unit-aware axes, and dashed references', () => {
    render(<TimeSeriesPlot title="Motor speed" xAxis={{ label: 'Time', unit: 's' }} yAxis={{ label: 'Speed', unit: 'rad/s' }} series={[
      { id: 'reference', name: 'Reference', x: [0, 1], y: [0, 10], color: '#2563eb', role: 'reference' },
      { id: 'actual', name: 'Actual', x: [0, 1], y: [0, 9], color: '#ef4444' },
    ]} />)

    expect(screen.getByRole('img', { name: /motor speed.*time.*s.*speed.*rad\/s/i })).toBeInTheDocument()
    expect(screen.getByRole('list', { name: /motor speed legend/i })).toHaveTextContent('Reference')
    expect(screen.getByTestId('series-reference')).toHaveAttribute('stroke-dasharray', '7 5')
  })

  it('filters non-finite samples and exposes a hover readout', () => {
    render(<TimeSeriesPlot title="Current" xAxis={{ label: 'Time', unit: 's' }} yAxis={{ label: 'Current', unit: 'A' }} series={[
      { id: 'current', name: 'Current', x: [0, Number.NaN, 2], y: [1, 2, 3], color: '#16a34a' },
    ]} />)
    fireEvent.pointerMove(screen.getByRole('img'), { clientX: 120, clientY: 100 })
    expect(screen.getByText(/Current:/)).toBeInTheDocument()
    expect(screen.queryByText(/NaN/)).not.toBeInTheDocument()
  })

  it('shows an understandable empty state when no finite samples are available', () => {
    render(<TimeSeriesPlot title="Error" xAxis={{ label: 'Time', unit: 's' }} yAxis={{ label: 'Error', unit: 'A' }} series={[
      { id: 'error', name: 'Error', x: [Number.NaN], y: [Number.NaN], color: '#ef4444' },
    ]} />)
    expect(screen.getByRole('status')).toHaveTextContent(/no finite samples/i)
  })
})
