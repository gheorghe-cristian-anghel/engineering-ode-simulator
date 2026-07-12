interface StatusMessageProps { kind: 'error' | 'loading' | 'empty'; title: string; children: string }

export function StatusMessage({ kind, title, children }: StatusMessageProps) {
  return <section className={`status-message status-message--${kind}`} role={kind === 'error' ? 'alert' : 'status'}><strong>{title}</strong><p>{children}</p></section>
}
