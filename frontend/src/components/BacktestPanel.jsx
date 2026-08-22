const CATEGORY_LABELS = {
  overall: 'overall',
  saas_subscription: 'saas_subscription',
  d2c_ecommerce: 'd2c_ecommerce',
  marketplace: 'marketplace',
  food_delivery: 'food_delivery',
}

function coverageColor(v) {
  if (v === null || v === undefined) return 'var(--muted)'
  if (v >= 0.80) return '#00c2a8'
  if (v >= 0.70) return '#e2b24a'
  return '#e05c5c'
}

function pct(v) {
  if (v === null || v === undefined) return '—'
  return (v * 100).toFixed(1) + '%'
}

function num(v) {
  if (v === null || v === undefined) return '—'
  return v.toLocaleString('en-IN', { maximumFractionDigits: 0 })
}

export default function BacktestPanel({ metrics, loading, error }) {
  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: '4px',
      padding: '16px',
    }}>
      <div style={{ color: 'var(--muted)', fontSize: '11px', marginBottom: '12px' }}>
        backtest accuracy — walk-forward, train / 14-day calib / 14-day test
      </div>

      {loading && <div style={{ color: 'var(--muted)' }}>loading…</div>}
      {error && <div style={{ color: '#e05c5c' }}>error: {error}</div>}

      {!loading && !error && metrics?.length > 0 && (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'inherit', fontSize: '12px' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--muted)' }}>
              <th style={th('left')}>category</th>
              <th style={th('right')}>cov (raw)</th>
              <th style={th('right')}>cov (cal)</th>
              <th style={th('right')}>pb P50</th>
              <th style={th('right')}>pb P10</th>
              <th style={th('right')}>pb P90</th>
              <th style={th('right')}>n days</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((row) => (
              <tr
                key={row.merchant_category}
                style={{
                  borderBottom: '1px solid #1e1e1e',
                  background: row.merchant_category === 'overall' ? '#1a1a1a' : 'transparent',
                }}
              >
                <td style={td('left', row.merchant_category === 'overall' ? 'var(--accent)' : 'var(--text)')}>
                  {CATEGORY_LABELS[row.merchant_category] || row.merchant_category}
                </td>
                <td style={td('right', 'var(--muted)')}>
                  {pct(row.raw_coverage)}
                </td>
                <td style={td('right', coverageColor(row.coverage_p10_p90))}>
                  {pct(row.coverage_p10_p90)}
                </td>
                <td style={td('right')}>{num(row.pinball_p50)}</td>
                <td style={td('right')}>{num(row.pinball_p10)}</td>
                <td style={td('right')}>{num(row.pinball_p90)}</td>
                <td style={td('right', 'var(--muted)')}>{row.n_days_total ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {!loading && !error && !metrics?.length && (
        <div style={{ color: 'var(--muted)' }}>
          no backtest data — run <code style={{ color: 'var(--accent)' }}>python train_and_backtest.py</code> first
        </div>
      )}

      <div style={{ marginTop: '10px', color: 'var(--muted)', fontSize: '11px', lineHeight: '1.7' }}>
        target coverage: 80% (P10–P90 band should contain the actual 8 times in 10).
        cov (raw) = leakage-fixed trajectories only. cov (cal) = + CQR calibration.
        pinball loss in ₹ — lower is better.
      </div>
    </div>
  )
}

const th = (align) => ({
  textAlign: align,
  padding: '6px 10px',
  fontWeight: 'normal',
  fontSize: '11px',
})

const td = (align, color = 'var(--text)') => ({
  textAlign: align,
  padding: '7px 10px',
  color,
  fontVariantNumeric: 'tabular-nums',
})
