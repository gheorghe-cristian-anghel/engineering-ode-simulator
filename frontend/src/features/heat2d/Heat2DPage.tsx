import { useState } from 'react'
import type { HeatRequest } from '../../api/heat'
import { AssumptionsPanel } from '../../components/AssumptionsPanel'
import { HeatmapCanvas } from '../../components/HeatmapCanvas'
import { MetricCard } from '../../components/MetricCard'
import { StatusMessage } from '../../components/StatusMessage'
import { HeatControls } from './HeatControls'
import { useHeatSimulation } from './useHeatSimulation'

const defaultRequest: HeatRequest = { alpha: 0.01, width: 1, height: 1, nx: 41, ny: 41, dt: 0.008, t_final: 0.8, boundary_type: 'dirichlet', boundary_value: 0, initial_condition: { kind: 'gaussian', amplitude: 1, width: 0.08 } }
const number = (value: number) => value.toFixed(4)

export function Heat2DPage() {
  const [request, setRequest] = useState(defaultRequest)
  const { result, error, isLoading, run } = useHeatSimulation()
  return <div className="page"><header className="page-header"><span className="eyebrow">PDEs / diffusion</span><h1>2D Heat Equation</h1><p>Explore heat diffusion on a plate with the existing bounded explicit finite-difference solver.</p></header>
    <div className="simulation-layout"><HeatControls value={request} onChange={setRequest} disabled={isLoading} onRun={() => run(request)} />
      <section className="results-panel" aria-live="polite"><div className="panel-heading"><div><span className="eyebrow">Result field</span><h2>Final temperature</h2></div>{result && <span className={`stability-badge ${result.stability.is_stable ? 'stability-badge--good' : ''}`}>{result.stability.is_stable ? 'Stable' : 'Unstable'} · Σr {number(result.stability.sum)}</span>}</div>
        {isLoading && <StatusMessage kind="loading" title="Solving heat field">The backend is computing a bounded numerical solution.</StatusMessage>}
        {error && <StatusMessage kind="error" title="Simulation could not run">{error}</StatusMessage>}
        {!result && !isLoading && !error && <StatusMessage kind="empty" title="Ready to simulate">Configure the plate and run the model to inspect its final heat field.</StatusMessage>}
        {result && <><HeatmapCanvas field={result.final_field} x={result.coordinates.x} y={result.coordinates.y} unit={result.units.temperature} /><div className="metrics-grid"><MetricCard label="Stability sum" value={number(result.stability.sum)} detail={`limit ≤ ${number(result.stability.limit)}`} tone={result.stability.is_stable ? 'good' : 'warning'} /><MetricCard label="Final maximum" value={number(result.thermal_metrics.final_max)} detail={result.units.temperature} /><MetricCard label="Final mean" value={number(result.thermal_metrics.final_mean)} detail={result.units.temperature} /><MetricCard label="Actual Δt" value={number(result.method.actual_dt)} detail={result.units.time} /></div><p className="interpretation">The initial hot region spreads across the plate; the shared final-field scale highlights the resulting thermal distribution.</p></>}
      </section></div>
    <AssumptionsPanel>This educational prototype uses a uniform grid, constant thermal diffusivity, and an explicit finite-difference update. Stability and backend work limits are safeguards, not a substitute for validation against a physical system.</AssumptionsPanel>
  </div>
}
