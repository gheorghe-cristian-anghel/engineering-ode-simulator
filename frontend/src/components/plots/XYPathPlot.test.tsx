import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { XYPathPlot } from './XYPathPlot'

describe('XYPathPlot', () => {
  it('shows path geometry markers, obstacle annotations, and an external legend', () => {
    render(<XYPathPlot
      title="Obstacle avoidance path"
      xAxis={{ label: 'East', unit: 'm' }}
      yAxis={{ label: 'North', unit: 'm' }}
      paths={[
        { id: 'reference', name: 'Reference path', color: '#2563eb', role: 'reference', points: [{ x: 0, y: 0 }, { x: 4, y: 0 }] },
        { id: 'actual', name: 'Actual path', color: '#f97316', role: 'actual', points: [{ x: 0, y: 0 }, { x: 3.8, y: 0.2 }] },
      ]}
      waypoints={[{ x: 0, y: 0 }, { x: 4, y: 0 }]}
      obstacle={{ center: { x: 2, y: 0 }, radius: 0.3, influenceRadius: 0.8 }}
    />)

    expect(screen.getByLabelText('Obstacle avoidance path legend')).toHaveTextContent('Reference path')
    expect(screen.getByLabelText('Obstacle avoidance path legend')).toHaveTextContent('Actual path')
    expect(screen.getByLabelText('Start marker')).toBeInTheDocument()
    expect(screen.getByLabelText('End marker')).toBeInTheDocument()
    expect(screen.getByLabelText('Waypoint 1')).toBeInTheDocument()
    expect(screen.getByLabelText('Obstacle')).toBeInTheDocument()
    expect(screen.getByLabelText('Obstacle influence radius')).toBeInTheDocument()
    expect(screen.getByRole('img')).toHaveAttribute('preserveAspectRatio', 'xMidYMid meet')
  })
})
