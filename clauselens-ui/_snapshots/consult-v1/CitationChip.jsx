const METHOD_LABEL = {
  hybrid: "hybrid · RRF",
  semantic: "semantic",
  bm25: "keyword",
  graph: "graph",
};

export default function CitationChip({ citation }) {
  return (
    <div className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-(--color-surface-2) border border-(--color-line) font-mono text-[11px]">
      <span className="w-1.5 h-1.5 rounded-full bg-(--color-brass)" />
      <span className="text-(--color-ink)">{citation.chunk_id}</span>
      <span className="text-(--color-ink-faint)">{citation.document_id}</span>
      <span className="text-(--color-ink-faint)">
        {METHOD_LABEL[citation.retrieval_method] ?? citation.retrieval_method}
      </span>
      <span className="text-(--color-cyan)">{citation.score.toFixed(4)}</span>
    </div>
  );
}
