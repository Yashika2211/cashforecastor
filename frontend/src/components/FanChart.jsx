import {
  ComposedChart, Area, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer,
} from 'recharts'

function fmt(v) {
  if (v === null || v === undefined) return '—'
  const abs = Math.abs(v)
  if (abs >= 1_000_000) return `₹${(v / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000)     return `₹${(v / 1_000).toFixed(1)}K`
  return `₹${v.toFixed(0)}`
}

function fmtDate(d) {
  if (!d) return ''
  return new Date(d).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' })
}

const Tip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload || {}
  return (
    <div style={{
      background: 'var(--overlay)',
      border: '1px solid var(--border)',
      borderRadius: '3px',
      padding: '8px 12px',
      fontFamily: 'var(--font-mono)',
      fontSize: '11px',
      lineHeight: '1.9',
    }}>
      <div style={{ color: 'var(--muted)', marginBottom: '2px' }}>{fmtDate(d.date)}</div>
      {d.actual  !== undefined && <div style={{ color: 'var(--text)' }}>actual  {fmt(d.actual)}</div>}
      {d.p90     !== undefined && <>
        <div style={{ color: 'var(--accent)' }}>p90     {fmt(d.p90)}</div>
        <div style={{ color: 'var(--accent)' }}>p50     {fmt(d.p50)}</div>
        <div style={{ color: 'var(--accent)' }}>p10     {fmt(d.p10)}</div>
      </>}
    </div>
  )
}

export default function FanChart({ history, forecast, forecastOrigin }) {
  const hist = (history || []).map(h => ({ date: h.date, actual: h.net_settled_amount }))
  const fcast = (forecast || []).map(f => ({ date: f.forecast_date, p10: f.p10, p50: f.p50, p90: f.p90 }))
  const data = [...hist, ...fcast]

  const tickEvery = Math.max(1, Math.floor(data.length / 9))
  const ticks = data.filter((_, i) => i % tickEvery === 0).map(d => d.date)

  const tickStyle = { fill: 'var(--muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }

  return (
    <div style={{ borderBottom: '1px solid var(--border)', paddingBottom: '12px', marginBottom: '0' }}>
      <div style={{ color: 'var(--muted)', fontSize: '10px', letterSpacing: '0.05em', textTransform: 'uppercase', marginBottom: '10px', fontFamily: 'var(--font-ui)' }}>
        net settled amount · 60-day history + 14-day forecast
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 2, right: 16, bottom: 0, left: 8 }}>
          <CartesianGrid strokeDasharray="2 4" stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="date" ticks={ticks} tickFormatter={fmtDate} tick={tickStyle} axisLine={{ stroke: 'var(--border)' }} tickLine={false} />
          <YAxis tickFormatter={fmt} tick={tickStyle} axisLine={false} tickLine={false} width={60} />
          <Tooltip content={<Tip />} />

          {/* P10–P90 shaded band */}
          <Area dataKey="p90" stroke="none" fill="var(--accent)" fillOpacity={0.10} connectNulls isAnimationActive={false} legendType="none" />
          <Area dataKey="p10" stroke="none" fill="var(--bg)"     fillOpacity={1}    connectNulls isAnimationActive={false} legendType="none" />

          {/* P90 / P10 boundary lines — very faint */}
          <Line dataKey="p90" stroke="var(--accent)" strokeWidth={0.8} strokeOpacity={0.4} dot={false} connectNulls isAnimationActive={false} legendType="none" />
          <Line dataKey="p10" stroke="var(--accent)" strokeWidth={0.8} strokeOpacity={0.4} dot={false} connectNulls isAnimationActive={false} legendType="none" />

          {/* P50 median */}
          <Line dataKey="p50" stroke="var(--accent)" strokeWidth={1.5} strokeDasharray="4 3" dot={false} connectNulls isAnimationActive={false} legendType="none" />

          {/* Actual history */}
          <Line dataKey="actual" stroke="var(--text)" strokeWidth={1.5} dot={false} connectNulls isAnimationActive={false} legendType="none" />

          {forecastOrigin && (
            <ReferenceLine x={forecastOrigin} stroke="var(--border)" strokeDasharray="3 3"
              label={{ value: 'today →', position: 'insideTopRight', fill: 'var(--muted)', fontSize: 9, fontFamily: 'var(--font-ui)' }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      <div style={{ display: 'flex', gap: '16px', paddingLeft: '68px', marginTop: '6px' }}>
        {[
          { color: 'var(--text)',   label: 'actual',      dashed: false },
          { color: 'var(--accent)', label: 'p50 forecast', dashed: true  },
          { band: true,             label: 'p10–p90 band' },
        ].map(item => (
          <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            {item.band
              ? <div style={{ width: 14, height: 7, background: 'var(--accent)', opacity: 0.2, borderRadius: '1px' }} />
              : <div style={{ width: 18, height: 0, borderTop: `1.5px ${item.dashed ? 'dashed' : 'solid'} ${item.color}` }} />
            }
            <span style={{ color: 'var(--muted)', fontSize: '10px', fontFamily: 'var(--font-ui)' }}>{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
