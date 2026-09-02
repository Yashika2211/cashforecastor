export function fmtPct(v, digits = 1) {
  return v == null ? '—' : `${(v * 100).toFixed(digits)}%`
}

export function fmtInt(v) {
  return v == null ? '—' : Math.round(v).toLocaleString('en-IN')
}

export function fmtRate(v) {
  return v == null ? '—' : `${fmtInt(v)}/s`
}

// Green above `good`, amber above `warn`, red below, grey when unknown.
// Every 0-1 "how healthy is this" score in the dashboard (coverage,
// confidence, match quality) reads off the same scale with its own cutoffs.
export function toneForScore(v, { good = 0.8, warn = 0.7 } = {}) {
  if (v == null) return 'var(--muted)'
  if (v >= good) return 'var(--good)'
  if (v >= warn) return 'var(--warn)'
  return 'var(--bad)'
}
