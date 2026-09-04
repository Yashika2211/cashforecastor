// Compact horizontal bar list for a small categorical breakdown (match rules,
// exception reasons). Not a recharts component on purpose -- at 3-6 bars in a
// ~200px-wide panel, a plain div/flex bar reads cleaner than an axis-bearing
// chart, and it's the same "hand-rolled over library" call FoldReliabilityStrip
// already made for its mini heatmap. Value + share are always shown inline
// (not hover-only) so a zero-count bar still visibly says "0", not nothing --
// an empty track that just isn't there would look like a bug, not a fact.

export default function BreakdownBarChart({ label, items }) {
  const max = Math.max(1, ...items.map(i => i.value))
  const total = items.reduce((sum, i) => sum + i.value, 0)

  return (
    <div>
      <div style={{ color: 'var(--muted)', fontSize: '10px', letterSpacing: '0.05em', textTransform: 'uppercase', fontFamily: 'var(--font-ui)', marginBottom: '8px' }}>
        {label}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {items.map(item => (
          <div key={item.key} style={{ display: 'grid', gridTemplateColumns: '112px 1fr 62px', alignItems: 'center', gap: '8px' }}>
            <span style={{
              fontSize: '10px', color: 'var(--muted)', fontFamily: 'var(--font-ui)',
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>
              {item.label}
            </span>
            <div style={{ height: '6px', background: 'var(--surface2)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{
                width: `${(item.value / max) * 100}%`, height: '100%',
                background: item.color, borderRadius: '3px', transition: 'width 0.3s ease',
              }} />
            </div>
            <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--muted)', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
              {item.value}{total > 0 ? ` · ${Math.round((item.value / total) * 100)}%` : ''}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
