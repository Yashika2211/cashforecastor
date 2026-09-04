const RAY_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

function SunIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="3.2" stroke="currentColor" strokeWidth="1.3" />
      {RAY_ANGLES.map(a => (
        <line key={a} x1="8" y1="1.4" x2="8" y2="2.8" stroke="currentColor" strokeWidth="1.3"
          strokeLinecap="round" transform={`rotate(${a} 8 8)`} />
      ))}
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
      <path d="M13.2 9.8A5.6 5.6 0 1 1 6.2 2.8a4.6 4.6 0 0 0 7 7Z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
    </svg>
  )
}

// The icon shown is the mode a click switches TO (dark now -> shows sun).
export default function ThemeToggle({ theme, onToggle }) {
  const isDark = theme === 'dark'
  return (
    <button
      className="icon-btn"
      onClick={onToggle}
      title={isDark ? 'switch to light mode' : 'switch to dark mode'}
      aria-label="toggle color theme"
    >
      {isDark ? <SunIcon /> : <MoonIcon />}
    </button>
  )
}
