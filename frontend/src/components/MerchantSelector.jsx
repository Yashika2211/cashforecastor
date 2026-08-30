export default function MerchantSelector({ merchants, value, onChange, loading }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <span style={{ color: 'var(--muted)', fontSize: '11px', fontFamily: 'var(--font-ui)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
        merchant
      </span>
      <select
        value={value || ''}
        onChange={(e) => onChange(e.target.value)}
        disabled={loading}
        style={{
          background: 'var(--surface)',
          color: 'var(--text)',
          border: '1px solid var(--border)',
          borderRadius: '3px',
          padding: '4px 28px 4px 8px',
          fontFamily: 'var(--font-mono)',
          fontSize: '12px',
          cursor: loading ? 'not-allowed' : 'pointer',
          minWidth: '180px',
          appearance: 'none',
          backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%238B929B'/%3E%3C/svg%3E")`,
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'right 8px center',
          outline: 'none',
        }}
        onFocus={e => e.target.style.borderColor = 'var(--accent)'}
        onBlur={e => e.target.style.borderColor = 'var(--border)'}
      >
        <option value="" disabled>select…</option>
        {merchants.map((id) => (
          <option key={id} value={id}>{id}</option>
        ))}
      </select>
    </div>
  )
}
