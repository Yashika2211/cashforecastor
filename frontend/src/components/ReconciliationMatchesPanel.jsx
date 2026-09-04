import RecordTable from './RecordTable'
import { toneForScore, toneForRule } from '../format'

const COLUMNS = [
  { key: 'ledger', header: 'ledger txns', width: '1.4fr',
    render: m => (
      <span style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
        {m.ledger_txn_ids.join(', ')}
        {m.member_count > 1 && <span style={{ color: 'var(--muted)' }}> ({m.member_count})</span>}
      </span>
    ) },
  { key: 'utr', header: 'utr', width: '1.1fr',
    render: m => <span style={{ color: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>{m.utr}</span> },
  { key: 'rule', header: 'rule', width: '1.1fr',
    render: m => <span style={{ color: toneForRule(m.rule), fontFamily: 'var(--font-mono)', fontSize: '10px' }}>{m.rule}</span> },
  { key: 'delta', header: 'Δ', width: '70px', align: 'right',
    cellStyle: m => ({ fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums', color: m.delta != null && Math.abs(m.delta) > 0.01 ? 'var(--warn)' : 'var(--muted)' }),
    render: m => m.delta != null ? m.delta.toFixed(2) : '—' },
  { key: 'conf', header: 'conf', width: '50px', align: 'right',
    cellStyle: m => ({ color: toneForScore(m.confidence, { good: 0.85, warn: 0.6 }), fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }),
    render: m => m.confidence != null ? m.confidence.toFixed(2) : '—' },
]

export default function ReconciliationMatchesPanel({ matches, loading, error }) {
  return (
    <RecordTable
      label="reconciliation matches"
      badge={{ tone: 'var(--good)', text: n => `${n} resolved` }}
      columns={COLUMNS}
      rows={matches}
      rowKey={m => m.match_group_id}
      loading={loading}
      error={error}
      emptyLabel="no matches"
      notRunLabel="run run_reconciliation.py first"
      maxHeight="calc(100vh - 220px)"
      footnote="rule = matching stage that resolved the record · conf = secondary triage score, never decides the match"
    />
  )
}
