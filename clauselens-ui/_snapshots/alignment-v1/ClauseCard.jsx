import { forwardRef } from "react";

const BORDER = {
  match: "border-l-(--color-cyan)",
  diverge: "border-l-(--color-diverge)",
  missing_in_a: "border-l-(--color-brass)",
  missing_in_b: "border-l-(--color-brass)",
  neutral: "border-l-(--color-line)",
};

const ClauseCard = forwardRef(function ClauseCard(
  { chunk, status = "neutral", active, onMouseEnter, onMouseLeave },
  ref
) {
  if (!chunk) {
    return (
      <div className="rounded-xl border border-dashed border-(--color-line) p-4 text-sm text-(--color-ink-faint) italic">
        No corresponding clause in this document.
      </div>
    );
  }

  return (
    <div
      ref={ref}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      className={`relative rounded-xl bg-(--color-surface) border border-(--color-line) border-l-4 ${BORDER[status]} p-4 transition-shadow
        ${active ? "shadow-[0_0_0_1px_rgba(94,234,212,0.25)]" : ""}`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="font-mono text-[10px] text-(--color-ink-faint) truncate">
          {chunk.chunk_id} · p.{chunk.page_start}
          {chunk.page_end !== chunk.page_start ? `–${chunk.page_end}` : ""}
        </div>
      </div>
      {chunk.heading_path?.length > 0 && (
        <div className="text-[11px] uppercase tracking-wide text-(--color-brass) font-mono mb-1.5">
          {chunk.heading_path.join(" / ")}
        </div>
      )}
      <p className="font-serif-text text-[15px] leading-relaxed text-(--color-ink)">{chunk.text}</p>
    </div>
  );
});

export default ClauseCard;
