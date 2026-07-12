interface AssumptionsPanelProps { children: string }

export function AssumptionsPanel({ children }: AssumptionsPanelProps) {
  return <aside className="assumptions-panel"><span>Scope and assumptions</span><p>{children}</p></aside>
}
