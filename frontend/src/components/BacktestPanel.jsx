import FoldReliabilityStrip from './FoldReliabilityStrip'

const LABELS = {
  overall:           'overall',
  saas_subscription: 'saas / subscription',
  d2c_ecommerce:     'd2c e-commerce',
  marketplace:       'marketplace',
  food_delivery:     'food delivery',
}

function covColor(v) {
  if (v == null) return 'var(--muted)'
  if (v >= 0.80) return 'var(--good)'
  if (v >= 0.70) return 'var(--warn)'
  return 'var(--bad)'
}

function pct(v)  { return v == null ? '—' : (v * 100).toFixed(1) + '%' }
function num(v)  { return v == null ? '—' : Math.round(v).toLocaleString('en-IN') }

export default function BacktestPanel({ metrics, folds, foldsLoading, foldsError, loading, error }) {
  return (
    <div>
      <div style={{ color: 'var(--muted)', fontSize: '10px', letterSpacing: '0.05em', textTransform: 'uppercase', fontFamily: 'var(--font-ui)', marginBottom: '10px' }}>
        backtest — 11 folds · train / 14-day calib / 14-day test
      </div>

      <FoldReliabilityStrip folds={folds} loading={foldsLoading} error={foldsError} />

      {loading && <div style={{ color: 'var(--muted)', fontSize: '12px' }}>loading…</div>}
      {error   && <div style={{ color: 'var(--bad)',   fontSize: '12px' }}>error: {error}</div>}

      {!loading && !error && metrics?.length > 0 && (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['category', 'raw', 'global', 'per-cat', 'pb P50', 'pb P10', 'pb P90', 'n'].map((h, i) => (
                <th key={h} style={thStyle(i === 0 ? 'left' : 'right')}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.map((row) => {
              const isOverall = row.merchant_category === 'overall'
              return (
                <tr key={row.merchant_category} style={{
                  borderBottom: '1px solid var(--border)',
                  background: isOverall ? 'var(--surface2)' : 'transparent',
                }}>
                  <td style={tdStyle('left', isOverall ? 'var(--text)' : 'var(--muted)')}>
                    {LABELS[row.merchant_category] || row.merchant_category}
                  </td>
                  <td style={tdStyle('right', 'var(--muted)')}>{pct(row.raw_coverage)}</td>
                  <td style={tdStyle('right', 'var(--muted)')}>{pct(row.global_coverage)}</td>
                  <td style={tdStyle('right', covColor(row.coverage_p10_p90))}>
                    <strong>{pct(row.coverage_p10_p90)}</strong>
                  </td>
                  <td style={tdStyle('right')}>{num(row.pinball_p50)}</td>
                  <td style={tdStyle('right')}>{num(row.pinball_p10)}</td>
                  <td style={tdStyle('right')}>{num(row.pinball_p90)}</td>
                  <td style={tdStyle('right', 'var(--muted)')}>{row.n_days_total ?? '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}

      {!loading && !error && !metrics?.length && (
        <div style={{ color: 'var(--muted)', fontSize: '12px' }}>no backtest data</div>
      )}

      <div style={{ marginTop: '8px', color: 'var(--muted)', fontSize: '10px', lineHeight: '1.8', fontFamily: 'var(--font-ui)' }}>
        target 80% · raw = no CQR · global = one q_hat all categories · per-cat = separate q_hat per category · pinball in ₹
      </div>
    </div>
  )
}

const thStyle = (align) => ({
  textAlign: align, padding: '5px 8px', fontWeight: 'normal',
  fontSize: '10px', color: 'var(--muted)', fontFamily: 'var(--font-ui)',
  letterSpacing: '0.03em', textTransform: 'uppercase',
})

const tdStyle = (align, color = 'var(--text)') => ({
  textAlign: align, padding: '6px 8px', color,
  fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums',
})
