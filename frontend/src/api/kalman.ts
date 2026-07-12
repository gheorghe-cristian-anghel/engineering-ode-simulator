import { ApiError } from './heat'

export interface KalmanRequest { voltage: number; t_final: number; dt: number; measurement_noise_std: number; random_seed: number }
export interface KalmanResponse {
  time: number[]; true_state: number[][]; measurements: number[]; estimates: number[][]; errors: number[][]
  metrics: { rms_current_error: number; rms_speed_error: number; final_current_error: number; final_speed_error: number }
  method: { name: string; model: string }; units: { time: string; current: string; speed: string; voltage: string }
  covariance?: number[][][]
}
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? ''
export async function runKalmanSimulation(request: KalmanRequest, signal?: AbortSignal): Promise<KalmanResponse> {
  let response: Response
  try { response = await fetch(`${apiBaseUrl}/api/estimation/kalman-filter`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request), signal }) }
  catch (error) { if (error instanceof DOMException && error.name === 'AbortError') throw error; throw new ApiError('Could not reach the simulation API. Start the FastAPI server and try again.') }
  if (!response.ok) { let message = `Simulation request failed (${response.status}).`; try { const body = await response.json() as { detail?: unknown }; if (typeof body.detail === 'string') message = body.detail } catch { /* use fallback */ } throw new ApiError(message, response.status) }
  return await response.json() as KalmanResponse
}
