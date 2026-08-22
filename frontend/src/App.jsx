import { useEffect, useState, useCallback } from 'react'
import MerchantSelector from './components/MerchantSelector'
import FanChart from './components/FanChart'
import BacktestPanel from './components/BacktestPanel'
import ExceptionsPanel from './components/ExceptionsPanel'
import {
  fetchMerchants,
  fetchForecast,
  fetchHistory,
  fetchBacktestMetrics,
  fetchExceptions,
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
  const [merchants, setMerchants] = useState([])
  const [selectedMerchant, setSelectedMerchant] = useState(null)

  const forecast = useAsync(fetchForecast, [])
  const history = useAsync(fetchHistory, [])
  const backtest = useAsync(fetchBacktestMetrics, [])
  const exceptions = useAsync(fetchExceptions, [])

  // Load static data once
  useEffect(() => {
    fetchMerchants()
      .then((ids) => {
        setMerchants(ids)
        if (ids.length > 0) setSelectedMerchant(ids[0])
      })
      .catch(() => {})
    backtest.run()
    exceptions.run()
  }, [])

  // Reload forecast + history when merchant changes
  useEffect(() => {
    if (!selectedMerchant) return
    forecast.run(selectedMerchant)
    history.run(selectedMerchant, 60)
  }, [selectedMerchant])

  const forecastOrigin = forecast.data?.forecast_origin ?? null

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', padding: '24px 32px' }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '24px',
        borderBottom: '1px solid var(--border)',
        paddingBottom: '16px',
      }}>
        <div>
          <span style={{ fontSize: '14px', color: 'var(--text)', letterSpacing: '0.05em' }}>
            cashflow forecaster
          </span>
          <span style={{ marginLeft: '10px', fontSize: '11px', color: 'var(--muted)' }}>
            quantile regression · P10 / P50 / P90 · 14-day horizon
          </span>
        </div>
        <MerchantSelector
          merchants={merchants}
          value={selectedMerchant}
          onChange={setSelectedMerchant}
          loading={forecast.loading || history.loading}
        />
      </div>

      {/* Error banner */}
      {forecast.error && (
        <div style={{
          background: '#1e0e0e',
          border: '1px solid #5a1a1a',
          color: '#e05c5c',
          padding: '10px 14px',
          borderRadius: '3px',
          marginBottom: '16px',
          fontSize: '12px',
        }}>
          forecast error: {forecast.error}
        </div>
      )}

      {/* Fan chart */}
      <div style={{ marginBottom: '20px' }}>
        {(forecast.loading || history.loading) && !forecast.data && (
          <div style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: '4px',
            height: '360px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--muted)',
            fontSize: '12px',
          }}>
            loading forecast…
          </div>
        )}
        {forecast.data && (
          <FanChart
            history={history.data?.history}
            forecast={forecast.data?.forecast}
            forecastOrigin={forecastOrigin}
          />
        )}
        {!forecast.loading && !forecast.data && !forecast.error && (
          <div style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: '4px',
            height: '200px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--muted)',
            fontSize: '12px',
          }}>
            select a merchant to see the forecast
          </div>
        )}
      </div>

      {/* Forecast table — compact day-by-day numbers */}
      {forecast.data?.forecast && (
        <div style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '4px',
          padding: '14px 16px',
          marginBottom: '20px',
          overflowX: 'auto',
        }}>
          <div style={{ color: 'var(--muted)', fontSize: '11px', marginBottom: '10px' }}>
            14-day forward forecast — net settled amount (₹)
          </div>
          <table style={{ borderCollapse: 'collapse', fontFamily: 'inherit', fontSize: '12px', width: '100%' }}>
            <thead>
              <tr style={{ color: 'var(--muted)', borderBottom: '1px solid var(--border)' }}>
                <th style={th('left')}>day</th>
                <th style={th('left')}>date</th>
                <th style={th('right')}>P10</th>
                <th style={th('right')}>P50</th>
                <th style={th('right')}>P90</th>
              </tr>
            </thead>
            <tbody>
              {forecast.data.forecast.map((row) => (
                <tr key={row.horizon_day} style={{ borderBottom: '1px solid #181818' }}>
                  <td style={td('left', 'var(--muted)')}>{row.horizon_day}</td>
                  <td style={td('left')}>{row.forecast_date}</td>
                  <td style={td('right', '#7fc8c0')}>{fmtNum(row.p10)}</td>
                  <td style={td('right', '#00c2a8')}>{fmtNum(row.p50)}</td>
                  <td style={td('right', '#7fc8c0')}>{fmtNum(row.p90)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Bottom panels */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '16px',
      }}>
        <BacktestPanel
          metrics={backtest.data}
          loading={backtest.loading}
          error={backtest.error}
        />
        <ExceptionsPanel
          flags={exceptions.data}
          loading={exceptions.loading}
          error={exceptions.error}
        />
      </div>
    </div>
  )
}

function fmtNum(v) {
  if (v === null || v === undefined) return '—'
  return v.toLocaleString('en-IN', { maximumFractionDigits: 0 })
}

const th = (align) => ({
  textAlign: align,
  padding: '5px 10px',
  fontWeight: 'normal',
  fontSize: '11px',
})

const td = (align, color = 'var(--text)') => ({
  textAlign: align,
  padding: '6px 10px',
  color,
  fontVariantNumeric: 'tabular-nums',
})
