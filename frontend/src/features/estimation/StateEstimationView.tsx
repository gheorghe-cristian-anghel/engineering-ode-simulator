import type { KalmanResponse } from '../../api/kalman'
import { TimeSeriesPlot } from '../../components/plots/TimeSeriesPlot'
interface StateEstimationViewProps { readonly result: KalmanResponse }
const colors = { truth: '#2563eb', measurement: '#f59e0b', estimate: '#16a34a', error: '#dc2626' }
export function StateEstimationView({ result }: StateEstimationViewProps) {
  const timeAxis = { label: 'Time', unit: result.units.time }
  return <section className="state-estimation-view">
    <TimeSeriesPlot title="Current state" xAxis={timeAxis} yAxis={{ label: 'Current', unit: result.units.current }} series={[{ id: 'true-current', name: 'True current', x: result.time, y: column(result.true_state, 0), color: colors.truth }, { id: 'estimate-current', name: 'Estimated current', x: result.time, y: column(result.estimates, 0), color: colors.estimate }]} />
    <TimeSeriesPlot title="Speed estimation" xAxis={timeAxis} yAxis={{ label: 'Speed', unit: result.units.speed }} series={[{ id: 'true-speed', name: 'True speed', x: result.time, y: column(result.true_state, 1), color: colors.truth }, { id: 'measurement-speed', name: 'Noisy measurement', x: result.time, y: result.measurements, color: colors.measurement, role: 'measurement' }, { id: 'estimate-speed', name: 'Estimated speed', x: result.time, y: column(result.estimates, 1), color: colors.estimate }]} />
    <TimeSeriesPlot title="Estimation error" xAxis={timeAxis} yAxis={{ label: 'Error', unit: result.units.speed }} series={[{ id: 'current-error', name: 'Current error', x: result.time, y: column(result.errors, 0), color: colors.error }, { id: 'speed-error', name: 'Speed error', x: result.time, y: column(result.errors, 1), color: '#7c3aed' }]} />
    {!result.covariance && <p className="state-estimation-view__note">No uncertainty covariance was returned by this simulation.</p>}
  </section>
}
function column(states: readonly (readonly number[])[], index: number): number[] { return states.map((state) => state[index] ?? Number.NaN) }
