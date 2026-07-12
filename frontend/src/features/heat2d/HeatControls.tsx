import type { HeatRequest } from '../../api/heat'

interface HeatControlsProps { value: HeatRequest; onChange: (next: HeatRequest) => void; disabled: boolean; onRun: () => void }
const numericKeys = ['width', 'height', 'alpha', 'nx', 'ny', 't_final', 'dt'] as const
const labels: Record<(typeof numericKeys)[number], string> = { width: 'Plate width (m)', height: 'Plate height (m)', alpha: 'Thermal diffusivity α', nx: 'Grid points X', ny: 'Grid points Y', t_final: 'Final time (s)', dt: 'Time step (s)' }

export function HeatControls({ value, onChange, disabled, onRun }: HeatControlsProps) {
  const updateNumber = (key: (typeof numericKeys)[number], next: string) => onChange({ ...value, [key]: Number(next) })
  return <form className="control-panel" onSubmit={(event) => { event.preventDefault(); onRun() }}>
    <div className="panel-heading"><div><span className="eyebrow">Configuration</span><h2>Heat model inputs</h2></div><span className="model-tag">Explicit FD</span></div>
    <div className="field-grid">{numericKeys.map((key) => <label key={key} className="field"><span>{labels[key]}</span><input type="number" min={key === 'nx' || key === 'ny' ? 3 : 0.0001} step={key === 'nx' || key === 'ny' ? 1 : 'any'} value={value[key]} onChange={(event) => updateNumber(key, event.target.value)} disabled={disabled} /></label>)}</div>
    <label className="field"><span>Boundary condition</span><select value={value.boundary_type} onChange={(event) => onChange({ ...value, boundary_type: event.target.value as HeatRequest['boundary_type'] })} disabled={disabled}><option value="dirichlet">Fixed temperature</option><option value="neumann">Zero flux</option></select></label>
    <label className="field"><span>Initial condition</span><select value={value.initial_condition.kind} onChange={(event) => onChange({ ...value, initial_condition: { ...value.initial_condition, kind: event.target.value as HeatRequest['initial_condition']['kind'] } })} disabled={disabled}><option value="gaussian">Gaussian hot spot</option><option value="rectangle">Rectangular hot region</option><option value="sine">Sine field</option></select></label>
    <button className="primary-button" type="submit" disabled={disabled}>{disabled ? 'Running simulation…' : 'Run simulation'}</button>
  </form>
}
