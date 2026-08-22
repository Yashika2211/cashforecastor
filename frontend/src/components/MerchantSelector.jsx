export default function MerchantSelector({ merchants, value, onChange, loading }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
      <label style={{ color: 'var(--muted)', whiteSpace: 'nowrap' }}>merchant</label>
      <select
        value={value || ''}
        onChange={(e) => onChange(e.target.value)}
        disabled={loading}
        style={{
          background: 'var(--surface)',
          color: 'var(--text)',
          border: '1px solid var(--border)',
          borderRadius: '3px',
          padding: '5px 10px',
          fontFamily: 'inherit',
          fontSize: '13px',
          cursor: loading ? 'not-allowed' : 'pointer',
          minWidth: '200px',
        }}
      >
        <option value="" disabled>select merchant…</option>
        {merchants.map((id) => (
          <option key={id} value={id}>{id}</option>
        ))}
      </select>
    </div>
  )
}
