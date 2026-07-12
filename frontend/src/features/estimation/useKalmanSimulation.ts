import { useRef, useState } from 'react'
import { type KalmanRequest, type KalmanResponse, runKalmanSimulation } from '../../api/kalman'
import { ApiError } from '../../api/heat'
export function useKalmanSimulation() {
  const controller = useRef<AbortController | null>(null); const [result, setResult] = useState<KalmanResponse | null>(null); const [error, setError] = useState<string | null>(null); const [isLoading, setIsLoading] = useState(false)
  async function run(request: KalmanRequest) { controller.current?.abort(); const current = new AbortController(); controller.current = current; setIsLoading(true); setError(null); try { setResult(await runKalmanSimulation(request, current.signal)) } catch (caught) { if (!(caught instanceof DOMException && caught.name === 'AbortError')) setError(caught instanceof ApiError ? caught.message : 'An unexpected error occurred.') } finally { if (controller.current === current) setIsLoading(false) } }
  return { result, error, isLoading, run }
}
