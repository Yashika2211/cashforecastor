import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer,
} from 'recharts'
import { fmtPct } from '../format'

const LABELS = {
  overall:           'overall',
  saas_subscription: 'saas',
  d2c_ecommerce:     'd2c',
  marketplace:       'mktplc',
  food_delivery:     'food',
}

// raw -> global CQR -> per-category CQR is a refinement order, not an
// arbitrary category set (swapping it would change what it means) -- so this
// reads as one hue at increasing saturation/contrast, not three unrelated
// colors. Validated as an ordinal ramp (monotone lightness, light-end
// contrast >= 2:1 in both themes) via the dataviz skill's validator.
const SERIES = [
  { key: 'raw',     label: 'raw',              color: 'var(--accent-dim)' },
  { key: 'global',  label: 'global CQR',       color: 'var(--accent-mid)' },
  { key: 'percat',  label: 'per-category CQR', color: 'var(--accent)' },
]

const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--overlay)', border: '1px solid var(--border)', borderRadius: '3px',
      padding: '8px 12px', fontFamily: 'var(--font-mono)', fontSize: '11px', lineHeight: '1.9',
    }}>
      <div style={{ color: 'var(--muted)', marginBottom: '2px' }}>{label}</div>
      {SERIES.map(s => {
        const p = payload.find(p => p.dataKey === s.key)
        if (!p) return null
        return <div key={s.key} style={{ color: s.color }}>{s.label.padEnd(16)} {fmtPct(p.value)}</div>
      })}
    </div>
  )
}

export default function CoverageComparisonChart({ metrics }) {
  if (!metrics?.length) return null

  const data = metrics.map(m => ({
    category: LABELS[m.merchant_category] || m.merchant_category,
    raw: m.raw_coverage,
    global: m.global_coverage,
    percat: m.coverage_p10_p90,
  }))

  return (
    <div style={{ marginBottom: '14px' }}>
      <div style={{ color: 'var(--muted)', fontSize: '10px', letterSpacing: '0.05em', textTransform: 'uppercase', fontFamily: 'var(--font-ui)', marginBottom: '8px' }}>
        coverage by calibration method · target 80%
      </div>
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }} barGap={2}>
          <CartesianGrid strokeDasharray="2 4" stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="category" tick={{ fill: 'var(--muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }} axisLine={{ stroke: 'var(--border)' }} tickLine={false} />
          <YAxis domain={[0, 1]} tickFormatter={v => `${Math.round(v * 100)}%`} tick={{ fill: 'var(--muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} width={30} />
          <Tooltip content={<Tip />} cursor={{ fill: 'var(--surface2)' }} />
          <ReferenceLine y={0.8} stroke="var(--muted)" strokeDasharray="3 3" />
          {SERIES.map(s => (
            <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[2, 2, 0, 0]} isAnimationActive={false} />
          ))}
        </BarChart>
      </ResponsiveContainer>
      <div style={{ display: 'flex', gap: '14px', marginTop: '6px' }}>
        {SERIES.map(s => (
          <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{ width: 8, height: 8, borderRadius: '2px', background: s.color }} />
            <span style={{ color: 'var(--muted)', fontSize: '10px', fontFamily: 'var(--font-ui)' }}>{s.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
