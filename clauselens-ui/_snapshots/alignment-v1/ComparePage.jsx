import { useMemo, useRef, useState } from "react";
import { documents, chunks, alignments } from "@/data/mockData";
import ClauseCard from "@/components/ClauseCard";
import AlignmentCanvas from "@/components/AlignmentCanvas";
import StatusPill from "@/components/StatusPill";

export default function ComparePage() {
  const containerRef = useRef(null);
  const nodeRefs = useRef({});
  const [activeId, setActiveId] = useState(null);

  const chunkById = useMemo(() => {
    const map = {};
    [...chunks.document_a, ...chunks.document_b].forEach((c) => (map[c.chunk_id] = c));
    return map;
  }, []);

  const activeAlignment = alignments.find((a) => a.id === activeId);

  const setRef = (id) => (el) => {
    if (id) nodeRefs.current[id] = el;
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-10">
      <div className="flex flex-wrap items-end justify-between gap-4 mb-8">
        <div>
          <h1 className="font-sans font-semibold text-3xl">Where the two versions agree — and where they don't</h1>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono text-(--color-ink-soft)">
          <Legend swatch="var(--color-match)" label="aligned" />
          <Legend swatch="var(--color-diverge)" label="diverges" />
          <Legend swatch="var(--color-brass)" label="added / dropped" dashed />
        </div>
      </div>

      <div ref={containerRef} className="relative grid grid-cols-2 gap-x-16">
        <AlignmentCanvas
          containerRef={containerRef}
          nodeRefs={nodeRefs}
          alignments={alignments}
          activeId={activeId}
        />

        <div className="relative z-10 flex flex-col gap-5">
          <ColumnHeader doc={documents.document_a} />
          {alignments.map((al) => (
            <ClauseCard
              key={al.id + "-a"}
              ref={setRef(al.a)}
              chunk={al.a ? chunkById[al.a] : null}
              status={al.status}
              active={activeId === al.id}
              onMouseEnter={() => setActiveId(al.id)}
              onMouseLeave={() => setActiveId(null)}
            />
          ))}
        </div>

        <div className="relative z-10 flex flex-col gap-5">
          <ColumnHeader doc={documents.document_b} align="right" />
          {alignments.map((al) => (
            <ClauseCard
              key={al.id + "-b"}
              ref={setRef(al.b)}
              chunk={al.b ? chunkById[al.b] : null}
              status={al.status}
              active={activeId === al.id}
              onMouseEnter={() => setActiveId(al.id)}
              onMouseLeave={() => setActiveId(null)}
            />
          ))}
        </div>
      </div>

      <DetailDrawer alignment={activeAlignment} />
    </div>
  );
}

function ColumnHeader({ doc, align = "left" }) {
  return (
    <div className={`mb-1 ${align === "right" ? "text-right" : ""}`}>
      <div className="font-mono text-[11px] text-(--color-ink-faint)">{doc.document_id}</div>
      <div className="font-serif-text text-xl">{doc.label}</div>
      <div className="text-xs text-(--color-ink-soft)">
        {doc.chunk_count} chunks · {doc.page_count} pages
      </div>
    </div>
  );
}

function Legend({ swatch, label, dashed }) {
  return (
    <span className="flex items-center gap-1.5">
      <svg width="16" height="2">
        <line x1="0" y1="1" x2="16" y2="1" stroke={swatch} strokeWidth="2" strokeDasharray={dashed ? "3 2" : "0"} />
      </svg>
      {label}
    </span>
  );
}

function DetailDrawer({ alignment }) {
  return (
    <div
      className={`fixed bottom-0 left-0 right-0 z-40 transition-transform duration-300 ${
        alignment ? "translate-y-0" : "translate-y-full"
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 pb-6">
        <div className="rounded-2xl bg-(--color-surface-2) border border-(--color-line) text-(--color-ink) shadow-xl p-5 flex items-start gap-4">
          {alignment && (
            <>
              <StatusPill status={alignment.status} />
              <p className="font-serif-text text-[15px] leading-snug flex-1">{alignment.note}</p>
              {alignment.score != null && (
                <div className="font-mono text-xs text-right shrink-0">
                  <div className="text-(--color-ink-faint)">match score</div>
                  <div className="text-lg text-(--color-cyan)">{alignment.score.toFixed(2)}</div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
