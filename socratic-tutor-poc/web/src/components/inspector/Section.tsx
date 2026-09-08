import type { ReactNode } from 'react';
import type { EngineKind } from '../../types';

/**
 * One collapsible trace section. The coloured band encodes which engine produced
 * the values inside: amber = LLM, blue = deterministic rules, green = data, red = policy.
 */
export function Section({
  title,
  engine,
  hint,
  defaultOpen = true,
  ready = true,
  pendingLabel = 'waiting for this step…',
  children,
}: {
  title: string;
  engine: EngineKind;
  hint?: ReactNode;
  defaultOpen?: boolean;
  ready?: boolean;
  pendingLabel?: string;
  children: ReactNode;
}) {
  if (!ready) {
    return (
      <div className="trace-section">
        <div style={{ display: 'flex', gap: 8, padding: '9px 14px', alignItems: 'center' }}>
          <span className={`band ${engine}`} style={{ width: 3, height: 16, borderRadius: 2, opacity: 0.4 }} />
          <span className="title muted">{title}</span>
        </div>
        <div className="pending">{pendingLabel}</div>
      </div>
    );
  }

  return (
    <details className="trace-section" open={defaultOpen}>
      <summary>
        <span className={`band ${engine}`} />
        <span className="chevron">▶</span>
        <span className="title">{title}</span>
        {hint && <span className="hint">{hint}</span>}
      </summary>
      <div className="trace-body">{children}</div>
    </details>
  );
}
