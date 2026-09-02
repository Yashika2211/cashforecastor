export default function MerchantSelector({ merchants, value, onChange, loading }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <span style={{ color: 'var(--muted)', fontSize: '11px', fontFamily: 'var(--font-ui)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
        merchant
      </span>
      <select
        className="select-input"
        value={value || ''}
        onChange={(e) => onChange(e.target.value)}
        disabled={loading}
      >
        <option value="" disabled>select…</option>
        {merchants.map((id) => (
          <option key={id} value={id}>{id}</option>
        ))}
      </select>
    </div>
  )
}
