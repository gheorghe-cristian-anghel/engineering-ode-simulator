import { useRef, useState } from 'react'
import { ApiError, type HeatRequest, type HeatResponse, runHeatSimulation } from '../../api/heat'

export function useHeatSimulation() {
  const controller = useRef<AbortController | null>(null)
  const [result, setResult] = useState<HeatResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  async function run(request: HeatRequest) {
    controller.current?.abort()
    controller.current = new AbortController()
    setIsLoading(true); setError(null)
    try { setResult(await runHeatSimulation(request, controller.current.signal)) }
    catch (caught) {
      if (!(caught instanceof DOMException && caught.name === 'AbortError')) setError(caught instanceof ApiError ? caught.message : 'An unexpected error occurred.')
    } finally { setIsLoading(false) }
  }
  return { result, error, isLoading, run }
}
