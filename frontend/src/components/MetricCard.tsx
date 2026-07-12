interface MetricCardProps { label: string; value: string; detail?: string; tone?: 'normal' | 'good' | 'warning' }

export function MetricCard({ label, value, detail, tone = 'normal' }: MetricCardProps) {
  return <article className={`metric-card metric-card--${tone}`}><span>{label}</span><strong>{value}</strong>{detail && <small>{detail}</small>}</article>
}
