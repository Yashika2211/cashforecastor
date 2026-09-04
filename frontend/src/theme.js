const STORAGE_KEY = 'settlement-radar-theme'

export function getStoredTheme() {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    return v === 'light' || v === 'dark' ? v : null
  } catch {
    return null // private browsing, storage disabled, etc.
  }
}

export function getSystemTheme() {
  return typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: light)').matches
    ? 'light'
    : 'dark'
}

// Only ever called with an explicit stored choice -- absent one, the CSS
// `prefers-color-scheme` media query in index.css handles the system default
// on its own, including live updates if the OS theme changes mid-session.
export function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme)
}

export function persistTheme(theme) {
  try {
    localStorage.setItem(STORAGE_KEY, theme)
  } catch {
    // theme just won't survive a reload -- not worth surfacing to the user
  }
}
