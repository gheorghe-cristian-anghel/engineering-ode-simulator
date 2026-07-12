import type { PropsWithChildren, ReactNode } from 'react'

interface PlotPanelProps extends PropsWithChildren {
  readonly title: string
  readonly description?: ReactNode
  readonly className?: string
}

export function PlotPanel({ title, description, className = '', children }: PlotPanelProps) {
  return (
    <section className={`plot-panel ${className}`.trim()} aria-label={title}>
      <header className="plot-panel__header">
        <h3>{title}</h3>
        {description && <p>{description}</p>}
      </header>
      {children}
    </section>
  )
}
