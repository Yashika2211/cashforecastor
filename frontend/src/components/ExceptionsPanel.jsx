import RecordTable from './RecordTable'
import { toneForScore } from '../format'

// Extract a short human-readable trigger from the reason string
function shortReason(reason) {
  if (!reason) return '—'
  const m = reason.match(/P90-P10 = ([\d,]+).*?\((.+?) of/)
  if (m) return `band ${m[2]} of median`
  return reason.slice(0, 60)
}

const COLUMNS = [
  { key: 'merchant', header: 'merchant', width: '130px',
    render: f => <span style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)' }}>{f.merchant_id}</span> },
  { key: 'date', header: 'date', width: '90px',
    render: f => <span style={{ color: 'var(--muted)', fontFamily: 'var(--font-mono)' }}>{f.flag_date}</span> },
  { key: 'conf', header: 'conf', width: '44px', align: 'right',
    cellStyle: f => ({ color: toneForScore(f.confidence_score, { good: 0.7, warn: 0.5 }), fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }),
    render: f => f.confidence_score != null ? f.confidence_score.toFixed(2) : '—' },
  { key: 'trigger', header: 'trigger',
    cellStyle: () => ({ color: 'var(--muted)', fontSize: '11px', fontFamily: 'var(--font-ui)' }),
    render: f => shortReason(f.reason) },
]

export default function ExceptionsPanel({ flags, loading, error }) {
  return (
    <RecordTable
      label="exceptions"
      badge={{ tone: 'var(--warn)', text: n => `${n} flagged` }}
      columns={COLUMNS}
      rows={flags}
      rowKey={(_, i) => i}
      loading={loading}
      error={error}
      emptyLabel="no flags"
      notRunLabel="run train_and_backtest.py first"
      maxHeight="240px"
      footnote="flagged when band > 50% of |p50| · conf = 1/(1 + band/median)"
    />
  )
}
