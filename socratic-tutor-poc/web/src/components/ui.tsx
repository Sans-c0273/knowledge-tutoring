import type { ReactNode } from 'react';

export type Tone = 'blue' | 'amber' | 'green' | 'red' | 'grey';

export function Pill({ tone = 'grey', children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span className={`pill ${tone}`} title={title}>
      {children}
    </span>
  );
}

export function Bar({ value, tone = 'blue' }: { value: number; tone?: Tone }) {
  const colour = {
    blue: 'var(--blue)',
    amber: 'var(--amber)',
    green: 'var(--green)',
    red: 'var(--red)',
    grey: 'var(--muted)',
  }[tone];
  return (
    <div className="bar" role="presentation">
      <div style={{ width: `${Math.max(0, Math.min(100, value))}%`, background: colour }} />
    </div>
  );
}

export function Mono({ children }: { children: ReactNode }) {
  return <span className="mono-value">{children}</span>;
}

/** Thai strings must carry lang="th" so the Thai font stack and line height apply. */
const THAI_RANGE = /[\u0E00-\u0E7F]/;

export function isThai(text: string): boolean {
  return THAI_RANGE.test(text);
}

export function langOf(text: string): 'th' | 'en' {
  return isThai(text) ? 'th' : 'en';
}
