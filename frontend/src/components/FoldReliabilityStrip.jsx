import { useState } from 'react'

function coverageColor(v) {
  if (v >= 0.80) return 'var(--good)'
  if (v >= 0.70) return 'var(--warn)'
  return 'var(--bad)'
}

function fmtQ(q) {
  if (q === null || q === undefined) return '—'
  const abs = Math.abs(q)
  const sign = q < 0 ? '-' : '+'
  if (abs >= 1000) return `${sign}₹${(abs / 1000).toFixed(0)}K`
  return `${sign}₹${abs.toFixed(0)}`
}

export default function FoldReliabilityStrip({ folds, loading, error }) {
  const [hovered, setHovered] = useState(null)

  if (loading) return (
    <div style={{ color: 'var(--muted)', fontSize: '11px', padding: '8px 0' }}>loading folds…</div>
  )
  if (error || !folds?.length) return null

  return (
    <div style={{ marginBottom: '10px' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        marginBottom: '4px',
      }}>
        <span style={{ color: 'var(--muted)', fontSize: '11px', fontFamily: 'var(--font-ui)', letterSpacing: '0.04em', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>
          fold coverage
        </span>
        <div style={{ display: 'flex', gap: '3px', alignItems: 'center', position: 'relative' }}>
          {folds.map((f) => (
            <div
              key={f.fold}
              onMouseEnter={() => setHovered(f)}
              onMouseLeave={() => setHovered(null)}
              style={{
                width: '22px',
                height: '18px',
                background: coverageColor(f.coverage),
                opacity: hovered?.fold === f.fold ? 1 : 0.75,
                borderRadius: '2px',
                cursor: 'default',
                transition: 'opacity 0.1s',
                position: 'relative',
              }}
            />
          ))}

          {/* Tooltip */}
          {hovered && (
            <div style={{
              position: 'absolute',
              top: '26px',
              left: '0',
              background: '#1e2228',
              border: '1px solid var(--border)',
              borderRadius: '3px',
              padding: '8px 10px',
              zIndex: 10,
              fontFamily: 'var(--font-mono)',
              fontSize: '11px',
              lineHeight: '1.8',
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
            }}>
              <div style={{ color: 'var(--muted)' }}>fold {hovered.fold} · cutoff {hovered.cutoff_date}</div>
              <div style={{ color: coverageColor(hovered.coverage) }}>
                coverage &nbsp;{(hovered.coverage * 100).toFixed(1)}%
              </div>
              <div style={{ color: hovered.q_hat >= 0 ? 'var(--warn)' : 'var(--good)' }}>
                CQR q̂ &nbsp;&nbsp;&nbsp;&nbsp;{fmtQ(hovered.q_hat)}
              </div>
              <div style={{ color: 'var(--muted)' }}>n_days &nbsp;&nbsp;{hovered.n_days}</div>
            </div>
          )}
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', gap: '10px', marginLeft: '6px' }}>
          {[['var(--good)', '≥80%'], ['var(--warn)', '70–80%'], ['var(--bad)', '<70%']].map(([color, label]) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <div style={{ width: 8, height: 8, background: color, borderRadius: '1px' }} />
              <span style={{ color: 'var(--muted)', fontSize: '10px' }}>{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
