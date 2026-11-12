// A source of an answer: which clause (chunk), in which document, on which page.
export default function CitationChip({ citation }) {
  return (
    <div className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-(--color-surface-2) border border-(--color-line) font-mono text-[11px]">
      <span className="w-1.5 h-1.5 rounded-full bg-(--color-brass)" />
      <span className="text-(--color-ink)">{citation.chunk_id}</span>
      <span className="text-(--color-ink-faint)">document {citation.document}</span>
      {citation.page != null && <span className="text-(--color-cyan)">p.{citation.page}</span>}
    </div>
  );
}
