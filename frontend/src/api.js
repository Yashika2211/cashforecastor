// In production (Vercel), VITE_API_BASE_URL is set to the Render backend URL.
// In local dev it's unset, so the Vite proxy at /api -> localhost:8000 is used.
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

export const fetchMerchants = () => get('/merchants')
export const fetchForecast = (merchantId) => get(`/forecast/${merchantId}`)
export const fetchHistory = (merchantId, days = 60) => get(`/history/${merchantId}?days=${days}`)
export const fetchBacktestMetrics = () => get('/backtest-metrics')
export const fetchExceptions = () => get('/exceptions')
