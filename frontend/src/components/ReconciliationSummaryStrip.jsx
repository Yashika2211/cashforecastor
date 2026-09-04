import BreakdownBarChart from './BreakdownBarChart'
import { fmtPct, fmtRate, ruleGroup, toneForRule, toneForReason, reasonLabel } from '../format'

function chip(label, value, color = 'var(--text)') {
  return (
    <div key={label} style={{ display: 'flex', flexDirection: 'column', gap: '2px', padding: '8px 12px', background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: '4px' }}>
      <span style={{ color: 'var(--muted)', fontSize: '9px', letterSpacing: '0.05em', textTransform: 'uppercase', fontFamily: 'var(--font-ui)' }}>{label}</span>
      <span style={{ color, fontSize: '15px', fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }}>{value}</span>
    </div>
  )
}

// Fixed display order -- the engine's own pass order (1:1 first, then
// timing, then batch) -- pre-seeded so the chart's shape doesn't shuffle
// between runs even if a group's count happens to be zero.
const RULE_GROUP_ORDER = ['1:1 match', 'timing near-miss', 'batch settlement']

function groupRuleCounts(ruleCounts) {
  if (!ruleCounts) return []
  const groups = new Map(RULE_GROUP_ORDER.map(label => [label, { key: label, label, value: 0, color: 'var(--muted)' }]))
  for (const [rule, count] of Object.entries(ruleCounts)) {
    const g = groups.get(ruleGroup(rule))
    if (!g) continue
    g.value += count
    g.color = toneForRule(rule)
  }
  return [...groups.values()]
}

// Worse-first: orphans and unresolved batches before duplicates and
// near-misses, so the chart reads in the same "how bad is this" order the
// exceptions list below it implies.
const REASON_ORDER = [
  'ORPHAN_LEDGER_TXN', 'ORPHAN_BANK_CREDIT', 'UNRESOLVED_SUBSET_SUM',
  'DUPLICATE_SUSPECTED', 'AMOUNT_MISMATCH', 'TIMING_OUT_OF_WINDOW',
]

function reasonItems(reasonCounts) {
  if (!reasonCounts) return []
  return REASON_ORDER.filter(code => code in reasonCounts).map(code => ({
    key: code, label: reasonLabel(code), value: reasonCounts[code], color: toneForReason(code),
  }))
}

export default function ReconciliationSummaryStrip({ summary, loading, error }) {
  return (
    <div>
      <div style={{ color: 'var(--muted)', fontSize: '10px', letterSpacing: '0.05em', textTransform: 'uppercase', fontFamily: 'var(--font-ui)', marginBottom: '10px' }}>
        reconciliation run — ledger vs bank settlement feed
      </div>

      {loading && <div style={{ color: 'var(--muted)', fontSize: '12px' }}>loading…</div>}
      {error   && <div style={{ color: 'var(--bad)',   fontSize: '12px' }}>error: {error}</div>}

      {!loading && !error && summary && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '16px' }}>
            {chip('match rate', fmtPct(summary.match_rate), 'var(--accent)')}
            {chip('throughput', fmtRate(summary.records_per_second))}
            {summary.precision != null && chip('precision', fmtPct(summary.precision), 'var(--good)')}
            {summary.recall != null && chip('recall', fmtPct(summary.recall), 'var(--good)')}
            {summary.false_match_rate_pooled != null &&
              chip('false-match rate', fmtPct(summary.false_match_rate_pooled), summary.false_match_rate_pooled > 0 ? 'var(--bad)' : 'var(--good)')}
            {summary.wrong_attribution_rate != null &&
              chip('wrong-attribution', fmtPct(summary.wrong_attribution_rate), summary.wrong_attribution_rate > 0 ? 'var(--bad)' : 'var(--good)')}
            {summary.trap_recall != null && chip('trap recall', fmtPct(summary.trap_recall), 'var(--accent)')}
          </div>

          {summary.rule_counts && (
            <div style={{ marginBottom: '16px' }}>
              <BreakdownBarChart label="resolved by" items={groupRuleCounts(summary.rule_counts)} />
            </div>
          )}
          {summary.reason_code_counts && (
            <BreakdownBarChart label="exceptions by reason" items={reasonItems(summary.reason_code_counts)} />
          )}
        </>
      )}

      {!loading && !error && !summary && (
        <div style={{ color: 'var(--muted)', fontSize: '12px' }}>run run_reconciliation.py first</div>
      )}

      {!loading && !error && summary && summary.precision == null && (
        <div style={{ marginTop: '8px', color: 'var(--muted)', fontSize: '10px', fontFamily: 'var(--font-ui)' }}>
          run score_reconciliation.py to see precision / recall / false-match rate against ground truth
        </div>
      )}
    </div>
  )
}
