const STYLES = {
  match: { label: "Aligned", fg: "text-(--color-cyan)", dot: "bg-(--color-cyan)", bg: "bg-(--color-cyan)/10" },
  diverge: { label: "Diverges", fg: "text-(--color-diverge)", dot: "bg-(--color-diverge)", bg: "bg-(--color-diverge)/10" },
  missing_in_a: { label: "New in v4", fg: "text-(--color-brass)", dot: "bg-(--color-brass)", bg: "bg-(--color-brass)/10" },
  missing_in_b: { label: "Dropped in v4", fg: "text-(--color-brass)", dot: "bg-(--color-brass)", bg: "bg-(--color-brass)/10" },
};

export default function StatusPill({ status }) {
  const s = STYLES[status] ?? STYLES.diverge;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${s.bg} ${s.fg}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
      {s.label}
    </span>
  );
}
