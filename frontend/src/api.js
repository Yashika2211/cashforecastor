const BASE = import.meta.env.VITE_API_BASE_URL
  ? `${import.meta.env.VITE_API_BASE_URL}`
  : '/api'

async function get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const fetchMerchants      = ()                    => get('/merchants')
export const fetchForecast       = (mid)                 => get(`/forecast/${mid}`)
export const fetchHistory        = (mid, days = 60)      => get(`/history/${mid}?days=${days}`)
export const fetchBacktestMetrics = ()                   => get('/backtest-metrics')
export const fetchBacktestFolds  = ()                    => get('/backtest-folds')
export const fetchExceptions     = ()                    => get('/exceptions')

export const fetchReconciliationSummary    = ()          => get('/reconciliation-summary')
export const fetchReconciliationMatches    = ()          => get('/reconciliation-matches')
export const fetchReconciliationExceptions = ()          => get('/reconciliation-exceptions')
export const fetchReconciliationAccuracy   = ()          => get('/reconciliation-accuracy')
