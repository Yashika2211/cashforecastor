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

// Which matching stage resolved a record, and how confident that stage is by
// nature: a clean 1:1 amount match needs no caution, a timing near-miss is a
// mild flag, a batch resolution is the most involved path. Shared between the
// matches table (per-row color) and the rule-breakdown chart (per-bar color)
// so the two views can never drift apart.
export function toneForRule(rule) {
  if (!rule) return 'var(--muted)'
  if (rule === 'exact_1to1' || rule === 'fee_adjusted_1to1') return 'var(--good)'
  if (rule.startsWith('timing_near_miss')) return 'var(--warn)'
  if (rule.startsWith('batch_subset_sum')) return 'var(--accent)'
  return 'var(--muted)'
}

export function ruleGroup(rule) {
  if (rule === 'exact_1to1' || rule === 'fee_adjusted_1to1') return '1:1 match'
  if (rule?.startsWith('timing_near_miss')) return 'timing near-miss'
  if (rule?.startsWith('batch_subset_sum')) return 'batch settlement'
  return 'other'
}

const WARN_REASON_CODES = new Set(['DUPLICATE_SUSPECTED', 'TIMING_OUT_OF_WINDOW', 'AMOUNT_MISMATCH'])
const BAD_REASON_CODES = new Set(['ORPHAN_LEDGER_TXN', 'ORPHAN_BANK_CREDIT', 'UNRESOLVED_SUBSET_SUM'])

// Same sharing rationale as toneForRule, for the exceptions table + chart.
export function toneForReason(code) {
  if (WARN_REASON_CODES.has(code)) return 'var(--warn)'
  if (BAD_REASON_CODES.has(code)) return 'var(--bad)'
  return 'var(--muted)'
}

export function reasonLabel(code) {
  return code.toLowerCase().replaceAll('_', ' ')
}
