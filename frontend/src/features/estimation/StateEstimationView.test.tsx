import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { StateEstimationView } from './StateEstimationView'

describe('StateEstimationView', () => {
  afterEach(() => cleanup())

  it('maps DC motor current and speed indices to unit-aware scientific plots', () => {
    render(<StateEstimationView result={{ time: [0, 1], true_state: [[1, 2], [2, 3]], measurements: [2.2, 3.1], estimates: [[0.9, 2.1], [1.9, 2.9]], errors: [[0.1, -0.1], [0.1, 0.1]], metrics: { rms_current_error: 0.1, rms_speed_error: 0.1, final_current_error: 0.1, final_speed_error: 0.1 }, method: { name: 'Kalman', model: 'DC motor' }, units: { time: 's', current: 'A', speed: 'rad/s', voltage: 'V' } }} />)
    expect(screen.getByRole('img', { name: /current state.*current.*A/i })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /speed estimation.*speed.*rad\/s/i })).toBeInTheDocument()
    expect(screen.getByText('No uncertainty covariance was returned by this simulation.')).toBeInTheDocument()
  })
})
