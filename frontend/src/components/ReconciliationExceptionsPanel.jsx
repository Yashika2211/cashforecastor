import RecordTable from './RecordTable'
import { toneForScore } from '../format'

const WARN_CODES = new Set(['DUPLICATE_SUSPECTED', 'TIMING_OUT_OF_WINDOW', 'AMOUNT_MISMATCH'])
const BAD_CODES = new Set(['ORPHAN_LEDGER_TXN', 'ORPHAN_BANK_CREDIT', 'UNRESOLVED_SUBSET_SUM'])

function reasonColor(code) {
  if (WARN_CODES.has(code)) return 'var(--warn)'
  if (BAD_CODES.has(code)) return 'var(--bad)'
  return 'var(--muted)'
}

const COLUMNS = [
  { key: 'record', header: 'record', width: '1.1fr',
    render: r => (
      <span style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
        {r.record_id}
        <span style={{ color: 'var(--muted)' }}> ({r.record_type[0]})</span>
      </span>
    ) },
  { key: 'merchant', header: 'merchant', width: '0.9fr',
    render: r => <span style={{ color: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>{r.merchant_id || '—'}</span> },
  { key: 'reason', header: 'reason', width: '1.3fr',
    render: r => <span style={{ color: reasonColor(r.reason_code), fontFamily: 'var(--font-mono)', fontSize: '10px' }}>{r.reason_code}</span> },
  { key: 'detail', header: 'detail', width: '1.6fr',
    cellStyle: () => ({ color: '#9AA0A8', fontSize: '11px', fontFamily: 'var(--font-ui)' }),
    render: r => r.detail || '—' },
  { key: 'delta', header: 'Δ', width: '60px', align: 'right',
    cellStyle: () => ({ fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums', color: 'var(--muted)' }),
    render: r => r.delta != null ? r.delta.toFixed(2) : '—' },
  { key: 'conf', header: 'conf', width: '50px', align: 'right',
    cellStyle: r => ({ color: toneForScore(r.confidence, { good: 0.7, warn: 0.5 }), fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }),
    render: r => r.confidence != null ? r.confidence.toFixed(2) : '—' },
]

export default function ReconciliationExceptionsPanel({ records, loading, error }) {
  return (
    <RecordTable
      label="reconciliation exceptions"
      badge={{ tone: 'var(--bad)', text: n => `${n} unresolved` }}
      columns={COLUMNS}
      rows={records}
      rowKey={r => `${r.record_type}-${r.record_id}`}
      loading={loading}
      error={error}
      emptyLabel="no exceptions"
      notRunLabel="run run_reconciliation.py first"
      footnote="AMOUNT_MISMATCH · ORPHAN_BANK_CREDIT · ORPHAN_LEDGER_TXN · DUPLICATE_SUSPECTED · TIMING_OUT_OF_WINDOW · UNRESOLVED_SUBSET_SUM"
    />
  )
}
