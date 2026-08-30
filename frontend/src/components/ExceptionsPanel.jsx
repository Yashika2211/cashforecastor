function confColor(s) {
  if (s == null) return 'var(--muted)'
  if (s >= 0.7)  return 'var(--good)'
  if (s >= 0.5)  return 'var(--warn)'
  return 'var(--bad)'
}

// Extract a short human-readable trigger from the reason string
function shortReason(reason) {
  if (!reason) return '—'
  const m = reason.match(/P90-P10 = ([\d,]+).*?\((.+?) of/)
  if (m) return `band ${m[2]} of median`
  return reason.slice(0, 60)
}

export default function ExceptionsPanel({ flags, loading, error }) {
  return (
    <div>
      <div style={{ color: 'var(--muted)', fontSize: '10px', letterSpacing: '0.05em', textTransform: 'uppercase', fontFamily: 'var(--font-ui)', marginBottom: '10px', display: 'flex', justifyContent: 'space-between' }}>
        <span>exceptions</span>
        {flags?.length > 0 && (
          <span style={{ color: 'var(--warn)', fontFamily: 'var(--font-mono)' }}>{flags.length} flagged</span>
        )}
      </div>

      {loading && <div style={{ color: 'var(--muted)', fontSize: '12px' }}>loading…</div>}
      {error   && <div style={{ color: 'var(--bad)',   fontSize: '12px' }}>error: {error}</div>}

      {!loading && !error && flags?.length > 0 && (
        <div style={{ maxHeight: '240px', overflowY: 'auto' }}>
          {/* header row */}
          <div style={headerRow}>
            <span>merchant</span>
            <span>date</span>
            <span style={{ textAlign: 'right' }}>conf</span>
            <span>trigger</span>
          </div>
          {flags.map((f, i) => (
            <div key={i} style={{
              ...dataRow,
              borderBottom: '1px solid var(--border)',
              background: i % 2 === 0 ? 'transparent' : 'var(--surface2)',
            }}>
              <span style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)' }}>{f.merchant_id}</span>
              <span style={{ color: 'var(--muted)', fontFamily: 'var(--font-mono)' }}>
                {f.flag_date}
              </span>
              <span style={{ textAlign: 'right', color: confColor(f.confidence_score), fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }}>
                {f.confidence_score != null ? f.confidence_score.toFixed(2) : '—'}
              </span>
              <span style={{ color: '#9AA0A8', fontSize: '11px', fontFamily: 'var(--font-ui)' }}>
                {shortReason(f.reason)}
              </span>
            </div>
          ))}
        </div>
      )}

      {!loading && !error && flags?.length === 0 && (
        <div style={{ color: 'var(--muted)', fontSize: '12px' }}>no flags</div>
      )}

      {!loading && !error && !flags && (
        <div style={{ color: 'var(--muted)', fontSize: '12px' }}>run train_and_backtest.py first</div>
      )}

      <div style={{ marginTop: '8px', color: 'var(--muted)', fontSize: '10px', fontFamily: 'var(--font-ui)' }}>
        flagged when band &gt; 50% of |p50| · conf = 1/(1 + band/median)
      </div>
    </div>
  )
}

const gridCols = { display: 'grid', gridTemplateColumns: '130px 90px 44px 1fr', gap: '0 10px', alignItems: 'start' }

const headerRow = {
  ...gridCols,
  padding: '4px 6px',
  color: 'var(--muted)',
  fontSize: '10px',
  letterSpacing: '0.03em',
  textTransform: 'uppercase',
  fontFamily: 'var(--font-ui)',
  borderBottom: '1px solid var(--border)',
  marginBottom: '1px',
}

const dataRow = {
  ...gridCols,
  padding: '5px 6px',
  fontSize: '12px',
  lineHeight: '1.4',
}
