import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { getStoredTheme, applyTheme } from './theme'

// Applied before the first paint so there's no flash of the wrong theme.
// No stored choice yet -> leave data-theme unset; the CSS media query
// handles the system default (and stays live if the OS theme changes).
const stored = getStoredTheme()
if (stored) applyTheme(stored)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
