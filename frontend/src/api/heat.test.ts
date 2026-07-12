import { describe, expect, it, vi } from 'vitest'
import { ApiError, runHeatSimulation } from './heat'

const request = {
  alpha: 0.01,
  width: 1,
  height: 1,
  nx: 21,
  ny: 21,
  dt: 0.001,
  t_final: 0.1,
  boundary_type: 'dirichlet' as const,
  boundary_value: 0,
  initial_condition: { kind: 'gaussian' as const, amplitude: 1, width: 0.08 },
}

describe('runHeatSimulation', () => {
  it('posts a typed heat request and returns the simulation result', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          coordinates: { x: [0, 1], y: [0, 1] },
          final_field: [[0, 0], [0, 0]],
          snapshots: [],
          stability: { rx: 0.1, ry: 0.1, sum: 0.2, limit: 0.5, is_stable: true },
          thermal_metrics: { initial_min: 0, initial_max: 1, final_min: 0, final_max: 0.5, final_mean: 0.2 },
          method: { name: 'explicit finite difference', equation: 'u_t', boundary_type: 'dirichlet', actual_dt: 0.001 },
          units: { coordinates: 'm', time: 's', temperature: 'arbitrary' },
        }),
        { status: 200 },
      ),
    )
    vi.stubGlobal('fetch', fetchMock)

    const result = await runHeatSimulation(request)

    expect(result.stability.is_stable).toBe(true)
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/pde/heat-2d',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('translates FastAPI validation details into an actionable API error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: [{ msg: 'Input should be greater than 0' }] }), { status: 422 }),
      ),
    )

    await expect(runHeatSimulation(request)).rejects.toEqual(
      new ApiError('Input should be greater than 0', 422),
    )
  })

  it('uses a backend detail string for bounded simulation errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'requested heat simulation exceeds the step limit' }), { status: 400 })))

    await expect(runHeatSimulation(request)).rejects.toEqual(new ApiError('requested heat simulation exceeds the step limit', 400))
  })

  it('explains when the API cannot be reached', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Network failure')))

    await expect(runHeatSimulation(request)).rejects.toEqual(new ApiError('Could not reach the simulation API. Start the FastAPI server and try again.'))
  })

  it('falls back to a status message for an unstructured error body', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 503 })))

    await expect(runHeatSimulation(request)).rejects.toEqual(new ApiError('Simulation request failed (503).', 503))
  })

  it('falls back to a status message when an error response is not JSON', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('service unavailable', { status: 503 })))

    await expect(runHeatSimulation(request)).rejects.toEqual(new ApiError('Simulation request failed (503).', 503))
  })
})
