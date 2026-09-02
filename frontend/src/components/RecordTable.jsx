// Shared shell for the "scrolling list of records" panels (ExceptionsPanel,
// ReconciliationMatchesPanel, ReconciliationExceptionsPanel all used to
// carry their own copy of this header/row/empty-state boilerplate).
// Callers own the domain-specific bits: column definitions, per-cell colors,
// the footnote. This owns the shape everything shares: heading + badge,
// loading/error/empty/not-run states, and the zebra-striped grid rows.

const heading = {
  color: 'var(--muted)', fontSize: '10px', letterSpacing: '0.05em', textTransform: 'uppercase',
  fontFamily: 'var(--font-ui)', marginBottom: '10px', display: 'flex', justifyContent: 'space-between',
}

const status = { color: 'var(--muted)', fontSize: '12px' }

export default function RecordTable({
  label, badge, columns, rows, rowKey,
  loading, error, emptyLabel = 'nothing to show', notRunLabel,
  footnote, maxHeight = '360px',
}) {
  const gridTemplateColumns = columns.map(c => c.width || '1fr').join(' ')

  return (
    <div>
      <div style={heading}>
        <span>{label}</span>
        {rows?.length > 0 && badge && (
          <span style={{ color: badge.tone, fontFamily: 'var(--font-mono)' }}>{badge.text(rows.length)}</span>
        )}
      </div>

      {loading && <div style={status}>loading…</div>}
      {error && <div style={{ ...status, color: 'var(--bad)' }}>error: {error}</div>}

      {!loading && !error && rows?.length > 0 && (
        <div style={{ maxHeight, overflowY: 'auto' }}>
          <div className="rt-row rt-row--head" style={{ gridTemplateColumns }}>
            {columns.map(c => (
              <span key={c.key} style={{ textAlign: c.align || 'left' }}>{c.header}</span>
            ))}
          </div>
          {rows.map((row, i) => (
            <div
              key={rowKey(row, i)}
              className={`rt-row${i % 2 ? ' rt-row--alt' : ''}`}
              style={{ gridTemplateColumns }}
            >
              {columns.map(c => (
                <span key={c.key} style={{ textAlign: c.align || 'left', ...(c.cellStyle?.(row) || null) }}>
                  {c.render(row)}
                </span>
              ))}
            </div>
          ))}
        </div>
      )}

      {!loading && !error && rows?.length === 0 && <div style={status}>{emptyLabel}</div>}
      {!loading && !error && !rows && notRunLabel && <div style={status}>{notRunLabel}</div>}

      {footnote && (
        <div style={{ marginTop: '8px', color: 'var(--muted)', fontSize: '10px', fontFamily: 'var(--font-ui)', lineHeight: '1.6' }}>
          {footnote}
        </div>
      )}
    </div>
  )
}
