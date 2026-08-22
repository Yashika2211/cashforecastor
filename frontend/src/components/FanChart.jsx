import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from 'recharts'

function fmt(v) {
  if (v === null || v === undefined) return '—'
  const abs = Math.abs(v)
  if (abs >= 1_000_000) return `₹${(v / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000) return `₹${(v / 1_000).toFixed(1)}K`
  return `₹${v.toFixed(0)}`
}

function fmtDate(d) {
  if (!d) return ''
  const dt = new Date(d)
  return dt.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' })
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload || {}
  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: '3px',
      padding: '10px 14px',
      fontFamily: 'inherit',
      fontSize: '12px',
      lineHeight: '1.8',
    }}>
      <div style={{ color: 'var(--muted)', marginBottom: '4px' }}>{fmtDate(label)}</div>
      {d.actual !== undefined && (
        <div style={{ color: 'var(--text)' }}>actual &nbsp;&nbsp;{fmt(d.actual)}</div>
      )}
      {d.p90 !== undefined && (
        <>
          <div style={{ color: '#00c2a8' }}>p90 &nbsp;&nbsp;&nbsp;&nbsp;{fmt(d.p90)}</div>
          <div style={{ color: '#00c2a8' }}>p50 &nbsp;&nbsp;&nbsp;&nbsp;{fmt(d.p50)}</div>
          <div style={{ color: '#00c2a8' }}>p10 &nbsp;&nbsp;&nbsp;&nbsp;{fmt(d.p10)}</div>
        </>
      )}
    </div>
  )
}

export default function FanChart({ history, forecast, forecastOrigin }) {
  // Merge history and forecast into one timeline
  const historyPoints = (history || []).map((h) => ({
    date: h.date,
    actual: h.net_settled_amount,
  }))

  const forecastPoints = (forecast || []).map((f) => ({
    date: f.forecast_date,
    p10: f.p10,
    p50: f.p50,
    p90: f.p90,
    // band: array form for Area
    band: [f.p10, f.p90],
  }))

  const data = [...historyPoints, ...forecastPoints]

  // Tick every ~5 points to avoid clutter
  const tickEvery = Math.max(1, Math.floor(data.length / 8))
  const ticks = data.filter((_, i) => i % tickEvery === 0).map((d) => d.date)

  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: '4px',
      padding: '20px 8px 12px 8px',
    }}>
      <div style={{ color: 'var(--muted)', fontSize: '11px', marginBottom: '12px', paddingLeft: '10px' }}>
        net settled amount — 60-day history + 14-day forecast (P10/P50/P90)
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 4, right: 20, bottom: 0, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" vertical={false} />
          <XAxis
            dataKey="date"
            ticks={ticks}
            tickFormatter={fmtDate}
            tick={{ fill: '#6b6b6b', fontSize: 11, fontFamily: 'inherit' }}
            axisLine={{ stroke: 'var(--border)' }}
            tickLine={false}
          />
          <YAxis
            tickFormatter={fmt}
            tick={{ fill: '#6b6b6b', fontSize: 11, fontFamily: 'inherit' }}
            axisLine={false}
            tickLine={false}
            width={64}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Forecast band P10→P90 */}
          <Area
            dataKey="p90"
            stroke="none"
            fill="#00c2a8"
            fillOpacity={0.12}
            connectNulls
            legendType="none"
            isAnimationActive={false}
          />
          <Area
            dataKey="p10"
            stroke="none"
            fill="var(--surface)"
            fillOpacity={1}
            connectNulls
            legendType="none"
            isAnimationActive={false}
          />

          {/* P50 median line */}
          <Line
            dataKey="p50"
            stroke="#00c2a8"
            strokeWidth={1.5}
            dot={false}
            connectNulls
            strokeDasharray="4 3"
            name="P50 forecast"
            isAnimationActive={false}
          />

          {/* Actual history line */}
          <Line
            dataKey="actual"
            stroke="#e2e2e2"
            strokeWidth={1.5}
            dot={false}
            connectNulls
            name="actual"
            isAnimationActive={false}
          />

          {/* Vertical rule at forecast start */}
          {forecastOrigin && (
            <ReferenceLine
              x={forecastOrigin}
              stroke="var(--border)"
              strokeDasharray="4 3"
              label={{ value: 'forecast →', position: 'insideTopRight', fill: '#6b6b6b', fontSize: 10, fontFamily: 'inherit' }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      <div style={{ display: 'flex', gap: '20px', paddingLeft: '74px', marginTop: '8px' }}>
        <LegendDot color="#e2e2e2" label="actual" />
        <LegendDot color="#00c2a8" label="P50 forecast" dashed />
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: 14, height: 8, background: '#00c2a8', opacity: 0.25, borderRadius: 2 }} />
          <span style={{ color: 'var(--muted)', fontSize: 11 }}>P10–P90 band</span>
        </div>
      </div>
    </div>
  )
}

function LegendDot({ color, label, dashed }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
      <div style={{
        width: 20,
        height: 2,
        background: color,
        borderTop: dashed ? `2px dashed ${color}` : `2px solid ${color}`,
      }} />
      <span style={{ color: 'var(--muted)', fontSize: 11 }}>{label}</span>
    </div>
  )
}
