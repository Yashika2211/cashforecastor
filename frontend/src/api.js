const BASE = '/api'

async function get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const fetchMerchants = () => get('/merchants')
export const fetchForecast = (merchantId) => get(`/forecast/${merchantId}`)
export const fetchHistory = (merchantId, days = 60) => get(`/history/${merchantId}?days=${days}`)
export const fetchBacktestMetrics = () => get('/backtest-metrics')
export const fetchExceptions = () => get('/exceptions')
