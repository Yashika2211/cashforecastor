import { useEffect, useState } from 'react'
import MerchantSelector from './components/MerchantSelector'
import ThemeToggle from './components/ThemeToggle'
import FanChart from './components/FanChart'
import BacktestPanel from './components/BacktestPanel'
import ExceptionsPanel from './components/ExceptionsPanel'
import ReconciliationMatchesPanel from './components/ReconciliationMatchesPanel'
import ReconciliationExceptionsPanel from './components/ReconciliationExceptionsPanel'
import ReconciliationSummaryStrip from './components/ReconciliationSummaryStrip'
import {
  fetchMerchants, fetchForecast, fetchHistory,
  fetchBacktestMetrics, fetchBacktestFolds, fetchExceptions,
  fetchReconciliationSummary, fetchReconciliationMatches, fetchReconciliationExceptions,
} from './api'
import { fmtInt, fmtPct } from './format'
import { getStoredTheme, getSystemTheme, applyTheme, persistTheme } from './theme'

// Fetch wrappers are stable module-level functions and none of these calls
// depend on `run`'s identity across renders (every consumer either fires it
// once on mount or in response to a user action) -- so no memoization needed.
function useAsync(fn) {
  const [state, setState] = useState({ data: null, loading: false, error: null })
  async function run(...args) {
    setState({ data: null, loading: true, error: null })
    try {
      const data = await fn(...args)
      setState({ data, loading: false, error: null })
    } catch (e) {
      setState({ data: null, loading: false, error: e.message })
    }
  }
  return { ...state, run }
}

export default function App() {
  const [merchants, setMerchants]           = useState([])
  const [selectedMerchant, setSelectedMerchant] = useState(null)
  const [view, setView] = useState('cash')
  const [theme, setTheme] = useState(() => getStoredTheme() ?? getSystemTheme())

  function toggleTheme() {
    const next = theme === 'dark' ? 'light' : 'dark'
    applyTheme(next)
    persistTheme(next)
    setTheme(next)
  }

  // Track OS-level theme changes for as long as the user hasn't explicitly
  // overridden the default -- CSS already does this on its own via the
  // `prefers-color-scheme` media query; this just keeps the toggle button's
  // icon in sync with it.
  useEffect(() => {
    const mq = matchMedia('(prefers-color-scheme: light)')
    const onChange = () => { if (!getStoredTheme()) setTheme(mq.matches ? 'light' : 'dark') }
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  const forecast  = useAsync(fetchForecast)
  const history   = useAsync(fetchHistory)
  const backtest  = useAsync(fetchBacktestMetrics)
  const folds     = useAsync(fetchBacktestFolds)
  const exceptions = useAsync(fetchExceptions)

  const reconSummary    = useAsync(fetchReconciliationSummary)
  const reconMatches    = useAsync(fetchReconciliationMatches)
  const reconExceptions = useAsync(fetchReconciliationExceptions)

  // Fires once on mount. `backtest`/`folds`/etc. are fresh objects every
  // render (useAsync returns `{...state, run}`), so listing them here would
  // just re-run this on every state update they cause -- an infinite loop,
  // not a missing dependency.
  /* oxlint-disable react-hooks/exhaustive-deps */
  useEffect(() => {
    fetchMerchants()
      .then(ids => { setMerchants(ids); if (ids.length) setSelectedMerchant(ids[0]) })
      .catch(() => {})
    backtest.run()
    folds.run()
    exceptions.run()
    reconSummary.run()
    reconMatches.run()
    reconExceptions.run()
  }, [])
  /* oxlint-enable react-hooks/exhaustive-deps */

  // Derived from the fold data, not independent state -- no effect needed.
  const lastBacktestDate = folds.data?.length
    ? folds.data[folds.data.length - 1].cutoff_date
    : null

  // Same reasoning: re-fetch only when the selected merchant changes.
  /* oxlint-disable react-hooks/exhaustive-deps */
  useEffect(() => {
    if (!selectedMerchant) return
    forecast.run(selectedMerchant)
    history.run(selectedMerchant, 60)
  }, [selectedMerchant])
  /* oxlint-enable react-hooks/exhaustive-deps */

  const forecastOrigin = forecast.data?.forecast_origin ?? null

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', display: 'flex', flexDirection: 'column' }}>

      {/* Top bar */}
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
          <div style={{ display: 'flex', gap: '4px' }}>
            {['cash', 'books'].map(v => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`tab-btn${view === v ? ' is-active' : ''}`}
              >
                {v}
              </button>
            ))}
          </div>
          <span style={{ color: 'var(--border)', fontSize: '10px' }}>|</span>
          <span style={{ color: 'var(--muted)', fontSize: '11px', fontFamily: 'var(--font-ui)' }}>
            {view === 'cash' ? 'quantile · P10 / P50 / P90 · 14-day horizon' : 'ledger vs bank UTR settlement feed'}
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
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>
      </div>

      {/* Error banner */}
      {forecast.error && (
        <div style={{ background: 'var(--error-bg)', borderBottom: '1px solid var(--error-border)', color: 'var(--bad)', padding: '6px 20px', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
          forecast error: {forecast.error}
        </div>
      )}

      {/* Main content */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 420px', minHeight: 0 }}>

        {view === 'cash' ? (
          <>
            {/* Left column: fan chart + forecast table */}
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
                          <td style={numCell('right', 'var(--accent)', 0.6)}>{fmtInt(row.p10)}</td>
                          <td style={numCell('right', 'var(--accent)')}>{fmtInt(row.p50)}</td>
                          <td style={numCell('right', 'var(--accent)', 0.6)}>{fmtInt(row.p90)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Right column: backtest + exceptions */}
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
          </>
        ) : (
          <>
            {/* Left column: reconciliation matches */}
            <div style={{ borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', padding: '16px 20px', overflowY: 'auto' }}>
              <ReconciliationMatchesPanel
                matches={reconMatches.data}
                loading={reconMatches.loading}
                error={reconMatches.error}
              />
            </div>

            {/* Right column: summary strip + exceptions */}
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <div style={{ padding: '16px', borderBottom: '1px solid var(--border)', flex: '0 0 auto' }}>
                <ReconciliationSummaryStrip
                  summary={reconSummary.data}
                  loading={reconSummary.loading}
                  error={reconSummary.error}
                />
              </div>
              <div style={{ padding: '16px', flex: 1, minHeight: 0 }}>
                <ReconciliationExceptionsPanel
                  records={reconExceptions.data}
                  loading={reconExceptions.loading}
                  error={reconExceptions.error}
                />
              </div>
            </div>
          </>
        )}
      </div>

      {/* Footer */}
      <div style={{
        borderTop: '1px solid var(--border)',
        padding: '6px 20px',
        display: 'flex', gap: '20px', alignItems: 'center',
        flexShrink: 0,
      }}>
        {(view === 'cash' ? [
          'model: v2 + cqr calibration',
          'trained on 4,085 rows · 18 merchants',
          'backtest: 11 folds · 2,310 eval days',
          lastBacktestDate ? `last fold cutoff: ${lastBacktestDate}` : null,
        ] : [
          `ledger: ${reconSummary.data?.n_ledger_rows ?? '—'} rows · bank: ${reconSummary.data?.n_bank_rows ?? '—'} rows`,
          `match rate: ${fmtPct(reconSummary.data?.match_rate)}`,
          `throughput: ${fmtInt(reconSummary.data?.records_per_second)} rec/s`,
          reconSummary.data?.precision != null
            ? `precision ${fmtPct(reconSummary.data.precision)} · recall ${fmtPct(reconSummary.data.recall)} · false-match ${fmtPct(reconSummary.data.false_match_rate_pooled)}`
            : null,
        ]).filter(Boolean).map(t => (
          <span key={t} style={{ color: 'var(--muted)', fontSize: '10px', fontFamily: 'var(--font-mono)' }}>{t}</span>
        ))}
      </div>

    </div>
  )
}

const numCell = (align, color = 'var(--text)', opacity = 1) => ({
  textAlign: align, padding: '5px 8px', color,
  fontVariantNumeric: 'tabular-nums', opacity,
})
