export type BoundaryType = 'dirichlet' | 'neumann'
export type InitialConditionKind = 'gaussian' | 'rectangle' | 'sine'
export type BoundaryValue = number | [number, number, number, number]

export interface HeatRequest {
  alpha: number
  width: number
  height: number
  nx: number
  ny: number
  dt?: number
  t_final: number
  boundary_type: BoundaryType
  boundary_value: BoundaryValue
  initial_condition: {
    kind: InitialConditionKind
    amplitude: number
    center?: [number, number]
    width?: number
    x_range?: [number, number]
    y_range?: [number, number]
    mode_x?: number
    mode_y?: number
  }
  include_snapshots?: boolean
  max_snapshots?: number
}

export interface HeatResponse {
  coordinates: { x: number[]; y: number[] }
  final_field: number[][]
  snapshots: Array<{ time: number; x: number[]; y: number[]; field: number[][]; downsample_stride: number }>
  stability: { rx: number; ry: number; sum: number; limit: number; is_stable: boolean }
  thermal_metrics: { initial_min: number; initial_max: number; final_min: number; final_max: number; final_mean: number }
  method: { name: string; equation: string; boundary_type: string; actual_dt: number }
  units: { coordinates: string; time: string; temperature: string }
}

export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message)
    this.name = 'ApiError'
  }
}

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? ''

export async function runHeatSimulation(request: HeatRequest, signal?: AbortSignal): Promise<HeatResponse> {
  let response: Response
  try {
    response = await fetch(`${apiBaseUrl}/api/pde/heat-2d`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError('Could not reach the simulation API. Start the FastAPI server and try again.')
  }

  if (!response.ok) throw new ApiError(await readError(response), response.status)
  return (await response.json()) as HeatResponse
}

async function readError(response: Response): Promise<string> {
  const fallback = `Simulation request failed (${response.status}).`
  try {
    const body: unknown = await response.json()
    if (!body || typeof body !== 'object' || !('detail' in body)) return fallback
    const { detail } = body as { detail: unknown }
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => (item && typeof item === 'object' && 'msg' in item ? String(item.msg) : ''))
        .filter(Boolean)
      return messages.join('; ') || fallback
    }
  } catch {
    return fallback
  }
  return fallback
}
