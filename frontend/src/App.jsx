import { useEffect, useState, useCallback } from 'react'
import MerchantSelector from './components/MerchantSelector'
import FanChart from './components/FanChart'
import BacktestPanel from './components/BacktestPanel'
import ExceptionsPanel from './components/ExceptionsPanel'
import {
  fetchMerchants, fetchForecast, fetchHistory,
  fetchBacktestMetrics, fetchBacktestFolds, fetchExceptions,
} from './api'

function useAsync(fn, deps) {
  const [state, setState] = useState({ data: null, loading: false, error: null })
  const run = useCallback(async (...args) => {
    setState({ data: null, loading: true, error: null })
    try {
      const data = await fn(...args)
      setState({ data, loading: false, error: null })
    } catch (e) {
      setState({ data: null, loading: false, error: e.message })
    }
  }, deps)
  return { ...state, run }
}

export default function App() {
  const [merchants, setMerchants]           = useState([])
  const [selectedMerchant, setSelectedMerchant] = useState(null)
  const [lastBacktestDate, setLastBacktestDate] = useState(null)

  const forecast  = useAsync(fetchForecast, [])
  const history   = useAsync(fetchHistory, [])
  const backtest  = useAsync(fetchBacktestMetrics, [])
  const folds     = useAsync(fetchBacktestFolds, [])
  const exceptions = useAsync(fetchExceptions, [])

  useEffect(() => {
    fetchMerchants()
      .then(ids => { setMerchants(ids); if (ids.length) setSelectedMerchant(ids[0]) })
      .catch(() => {})
    backtest.run()
    folds.run()
    exceptions.run()
  }, [])

  useEffect(() => {
    if (folds.data?.length) {
      const last = folds.data[folds.data.length - 1]
      setLastBacktestDate(last.cutoff_date)
    }
  }, [folds.data])

  useEffect(() => {
    if (!selectedMerchant) return
    forecast.run(selectedMerchant)
    history.run(selectedMerchant, 60)
  }, [selectedMerchant])

  const forecastOrigin = forecast.data?.forecast_origin ?? null

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', display: 'flex', flexDirection: 'column' }}>

      {/* ── Top bar ── */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 20px', height: '40px',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontFamily: 'var(--font-ui)', fontSize: '12px', letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--text)' }}>
            settlement radar
          </span>
          <span style={{ color: 'var(--border)', fontSize: '10px' }}>|</span>
          <span style={{ color: 'var(--muted)', fontSize: '11px', fontFamily: 'var(--font-ui)' }}>
            quantile · P10 / P50 / P90 · 14-day horizon
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <MerchantSelector
            merchants={merchants}
            value={selectedMerchant}
            onChange={setSelectedMerchant}
            loading={forecast.loading}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--good)' }} />
            <span style={{ color: 'var(--muted)', fontSize: '10px', fontFamily: 'var(--font-ui)' }}>model v2 + cqr</span>
          </div>
        </div>
      </div>

      {/* ── Error banner ── */}
      {forecast.error && (
        <div style={{ background: '#1a0e0e', borderBottom: '1px solid #5a1a1a', color: 'var(--bad)', padding: '6px 20px', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
          forecast error: {forecast.error}
        </div>
      )}

      {/* ── Main content ── */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 420px', minHeight: 0 }}>

        {/* Left column — fan chart + forecast table */}
        <div style={{ borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column' }}>

          {/* Fan chart */}
          <div style={{ padding: '16px 20px 12px', borderBottom: '1px solid var(--border)' }}>
            {(forecast.loading || history.loading) && !forecast.data ? (
              <div style={{ height: '280px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', fontSize: '11px' }}>loading…</div>
            ) : forecast.data ? (
              <FanChart history={history.data?.history} forecast={forecast.data?.forecast} forecastOrigin={forecastOrigin} />
            ) : (
              <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', fontSize: '11px' }}>select a merchant</div>
            )}
          </div>

          {/* Forecast table */}
          {forecast.data?.forecast && (
            <div style={{ padding: '12px 20px', overflowX: 'auto', flex: 1 }}>
              <div style={{ color: 'var(--muted)', fontSize: '10px', letterSpacing: '0.05em', textTransform: 'uppercase', fontFamily: 'var(--font-ui)', marginBottom: '8px' }}>
                14-day forward · net settled amount (₹)
              </div>
              <table style={{ borderCollapse: 'collapse', fontSize: '12px', fontFamily: 'var(--font-mono)', width: '100%' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    {['d', 'date', 'p10', 'p50', 'p90'].map((h, i) => (
                      <th key={h} style={{ textAlign: i < 2 ? 'left' : 'right', padding: '4px 8px', color: 'var(--muted)', fontWeight: 'normal', fontSize: '10px', fontFamily: 'var(--font-ui)', textTransform: 'uppercase', letterSpacing: '0.03em' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {forecast.data.forecast.map(row => (
                    <tr key={row.horizon_day} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={numCell('left', 'var(--muted)')}>{row.horizon_day}</td>
                      <td style={numCell('left')}>{row.forecast_date}</td>
                      <td style={numCell('right', 'var(--accent)', 0.6)}>{fmtNum(row.p10)}</td>
                      <td style={numCell('right', 'var(--accent)')}>{fmtNum(row.p50)}</td>
                      <td style={numCell('right', 'var(--accent)', 0.6)}>{fmtNum(row.p90)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Right column — backtest + exceptions */}
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '16px', borderBottom: '1px solid var(--border)', flex: '0 0 auto' }}>
            <BacktestPanel
              metrics={backtest.data}
              loading={backtest.loading}
              error={backtest.error}
              folds={folds.data}
              foldsLoading={folds.loading}
              foldsError={folds.error}
            />
          </div>
          <div style={{ padding: '16px', flex: 1, minHeight: 0 }}>
            <ExceptionsPanel
              flags={exceptions.data}
              loading={exceptions.loading}
              error={exceptions.error}
            />
          </div>
        </div>
      </div>

      {/* ── Footer ── */}
      <div style={{
        borderTop: '1px solid var(--border)',
        padding: '6px 20px',
        display: 'flex', gap: '20px', alignItems: 'center',
        flexShrink: 0,
      }}>
        {[
          'model: v2 + cqr calibration',
          'trained on 4,085 rows · 18 merchants',
          'backtest: 11 folds · 2,310 eval days',
          lastBacktestDate ? `last fold cutoff: ${lastBacktestDate}` : null,
        ].filter(Boolean).map(t => (
          <span key={t} style={{ color: 'var(--muted)', fontSize: '10px', fontFamily: 'var(--font-mono)' }}>{t}</span>
        ))}
      </div>

    </div>
  )
}

function fmtNum(v) {
  if (v == null) return '—'
  return Math.round(v).toLocaleString('en-IN')
}

const numCell = (align, color = 'var(--text)', opacity = 1) => ({
  textAlign: align, padding: '5px 8px', color,
  fontVariantNumeric: 'tabular-nums', opacity,
})
