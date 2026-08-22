function fmtDate(d) {
  if (!d) return '—'
  return new Date(d).toLocaleDateString('en-IN', { year: 'numeric', month: 'short', day: 'numeric' })
}

function confColor(score) {
  if (score === null || score === undefined) return 'var(--muted)'
  if (score >= 0.7) return '#00c2a8'
  if (score >= 0.5) return '#e2b24a'
  return '#e05c5c'
}

export default function ExceptionsPanel({ flags, loading, error }) {
  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: '4px',
      padding: '16px',
    }}>
      <div style={{ color: 'var(--muted)', fontSize: '11px', marginBottom: '12px' }}>
        low-confidence flags — merchant-days where the P10–P90 band is unusually wide
      </div>

      {loading && <div style={{ color: 'var(--muted)' }}>loading…</div>}
      {error && <div style={{ color: '#e05c5c' }}>error: {error}</div>}

      {!loading && !error && flags?.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1px', maxHeight: '280px', overflowY: 'auto' }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '160px 100px 60px 1fr',
            gap: '0 12px',
            padding: '4px 8px',
            color: 'var(--muted)',
            fontSize: '11px',
            borderBottom: '1px solid var(--border)',
            marginBottom: '2px',
          }}>
            <span>merchant</span>
            <span>date</span>
            <span style={{ textAlign: 'right' }}>conf.</span>
            <span>reason</span>
          </div>

          {flags.map((f, i) => (
            <div key={i} style={{
              display: 'grid',
              gridTemplateColumns: '160px 100px 60px 1fr',
              gap: '0 12px',
              padding: '6px 8px',
              background: i % 2 === 0 ? 'transparent' : '#131313',
              fontSize: '12px',
              lineHeight: '1.5',
              alignItems: 'start',
            }}>
              <span style={{ color: 'var(--accent)', fontVariantNumeric: 'tabular-nums' }}>
                {f.merchant_id}
              </span>
              <span style={{ color: 'var(--muted)' }}>{fmtDate(f.flag_date)}</span>
              <span style={{ textAlign: 'right', color: confColor(f.confidence_score), fontVariantNumeric: 'tabular-nums' }}>
                {f.confidence_score !== null && f.confidence_score !== undefined
                  ? f.confidence_score.toFixed(2)
                  : '—'}
              </span>
              <span style={{ color: '#a0a0a0', fontSize: '11px' }}>{f.reason}</span>
            </div>
          ))}
        </div>
      )}

      {!loading && !error && flags?.length === 0 && (
        <div style={{ color: 'var(--muted)' }}>no flags — all merchant forecasts within normal confidence bounds</div>
      )}

      {!loading && !error && !flags && (
        <div style={{ color: 'var(--muted)' }}>
          no exception data — run <code style={{ color: 'var(--accent)' }}>python train_and_backtest.py</code> first
        </div>
      )}
    </div>
  )
}
