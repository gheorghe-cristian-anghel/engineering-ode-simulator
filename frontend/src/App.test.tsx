import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { App } from './App'

describe('App', () => {
  afterEach(() => cleanup())

  it('exposes the Kalman estimation simulation through navigation and its direct route', () => {
    render(<MemoryRouter initialEntries={['/simulations/estimation/kalman']}><App /></MemoryRouter>)

    expect(screen.getByRole('link', { name: /kalman state estimation/i })).toHaveAttribute('href', '/simulations/estimation/kalman')
    expect(screen.getByRole('heading', { name: /kalman state estimation/i })).toBeInTheDocument()
  })

  it('exposes the UAV path simulation through navigation and its direct route', () => {
    render(<MemoryRouter initialEntries={['/simulations/uav-path']}><App /></MemoryRouter>)

    expect(screen.getByRole('link', { name: /uav path planning/i })).toHaveAttribute('href', '/simulations/uav-path')
    expect(screen.getByRole('heading', { name: /uav path planning/i })).toBeInTheDocument()
  })
})
