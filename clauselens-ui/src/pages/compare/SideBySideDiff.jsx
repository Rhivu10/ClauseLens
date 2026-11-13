import { STATUS_LABEL, STATUS_COLOR } from "./rows";
import HighlightedText from "./HighlightedText";

// The alignment view drawn as two documents laid side by side on the desk: the
// previous version on the left sheet, the new version on the right, each a small
// stack of paper. Clause pairs line up across the two sheets; each pair's match
// status is a sticky note on the outer edge, and its impact summary is a memo
// card pinned across the gap between the sheets.

const GAP = 56; // px between the two sheets

function Sheet({ side }) {
  // one stack of paper (behind the text): the top sheet plus two peeking out under it
  const pos = side === "left" ? { left: 0 } : { right: 0 };
  const tilt = side === "left" ? -1 : 1;
  return (
    <div className="absolute inset-y-0 pointer-events-none" style={{ ...pos, width: `calc(50% - ${GAP / 2}px)` }} aria-hidden="true">
      <div
        className="absolute inset-0 rounded-[3px] bg-[#f4efe7] border border-(--color-ink)/10"
        style={{ transform: `translate(${tilt * 7}px, 9px) rotate(${tilt * 0.9}deg)` }}
      />
      <div
        className="absolute inset-0 rounded-[3px] bg-[#f8f4ee] border border-(--color-ink)/10"
        style={{ transform: `translate(${tilt * 3}px, 4px) rotate(${tilt * 0.4}deg)` }}
      />
      <div className="absolute inset-0 rounded-[3px] bg-[#fffdf9] border border-(--color-ink)/10 shadow-[0_24px_48px_-24px_rgba(41,36,29,0.35)]" />
    </div>
  );
}

export default function SideBySideDiff({ documents, rows, highlight }) {
  return (
    <div className="mx-auto max-w-6xl my-4 pb-4 px-4 sm:px-16">
      <div className="relative">
      <Sheet side="left" />
      <Sheet side="right" />

      <div className="relative grid grid-cols-2" style={{ columnGap: GAP }}>
        {/* letterheads */}
        <Letterhead label={documents.document_a.label} sub="Previous version" />
        <Letterhead label={documents.document_b.label} sub="New version" />

        {rows.map((row) => {
          const accent = STATUS_COLOR[row.status];
          const paired = row.chunkA && row.chunkB;
          const noteSide = row.chunkB ? "right" : "left";
          const lit = highlight && highlight === row.change_id ? "bg-(--color-brass)/[0.06]" : "";
          return (
            <section
              key={row.id}
              id={`change-${row.change_id}`}
              className="relative col-span-2 grid grid-cols-subgrid scroll-mt-24"
            >
              <StickyNote side={noteSide} accent={accent} label={STATUS_LABEL[row.status]} />

              <Page side="left" className={lit}>
                {row.chunkA ? (
                  <>
                    <Head chunk={row.chunkA} score={row.score} accent={accent} />
                    {paired ? (
                      <HighlightedText
                        self={row.chunkA.text}
                        other={row.chunkB.text}
                        mode="del"
                        className="text-justify"
                      />
                    ) : (
                      <p className="font-serif-text text-[15px] leading-relaxed text-(--color-ink) text-justify">
                        {row.chunkA.text}
                      </p>
                    )}
                  </>
                ) : (
                  <Absent />
                )}
              </Page>

              <Page side="right" className={lit}>
                {row.chunkB ? (
                  <>
                    <Head chunk={row.chunkB} accent={accent} />
                    {paired ? (
                      <HighlightedText
                        self={row.chunkB.text}
                        other={row.chunkA.text}
                        mode="ins"
                        className="text-justify"
                      />
                    ) : (
                      <p className="font-serif-text text-[15px] leading-relaxed text-(--color-ink) text-justify">
                        {row.chunkB.text}
                      </p>
                    )}
                  </>
                ) : (
                  <Absent />
                )}
              </Page>

              {/* the impact summary: a memo card pinned across the gap */}
              <div className="col-span-2 px-8 pb-6 pt-1">
                <div className="relative mx-auto max-w-3xl rounded-[3px] border border-(--color-ink)/10 bg-(--color-surface) px-5 py-3 shadow-[0_6px_16px_-8px_rgba(41,36,29,0.3)]">
                  <span className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full bg-(--color-accent) shadow-[0_1px_2px_rgba(41,36,29,0.4)]" aria-hidden="true" />
                  <Meta row={row} />
                  {row.pending ? (
                    <p className="font-serif-text italic text-[13.5px] leading-snug text-(--color-ink-faint) flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-(--color-brass) animate-pulse" />
                      Writing and checking the impact summary…
                    </p>
                  ) : (
                    <p className="font-serif-text italic text-[13.5px] leading-snug text-(--color-ink-dim)">
                      {row.note}
                    </p>
                  )}
                  {row.needs_review && !row.pending && (
                    <p className="mt-1.5 font-mono text-[11px] text-(--color-diverge)">
                      Needs human review{row.review_reason ? ` · ${row.review_reason}` : ""}
                    </p>
                  )}
                </div>
              </div>
            </section>
          );
        })}
      </div>
      </div>
    </div>
  );
}

function Letterhead({ label, sub }) {
  return (
    <div className="px-10 pt-8 pb-4">
      <div className="font-serif text-xl tracking-wide uppercase text-(--color-ink) truncate">{label}</div>
      <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.2em] text-(--color-ink-faint)">{sub}</div>
      <div className="mt-3 h-px bg-(--color-ink)/15" />
    </div>
  );
}

function Meta({ row }) {
  const bits = [
    row.change_id,
    row.section,
    row.affected?.length ? `affects ${row.affected.join(", ")}` : null,
    row.duty_shift ? `duty ${row.duty_shift.replace("->", "→")}` : null,
    row.pending ? "summary in progress" : row.confidence ? `${row.confidence.toLowerCase()} confidence` : null,
  ].filter(Boolean);
  return (
    <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wide text-(--color-ink-faint)">
      {bits.join(" · ")}
    </div>
  );
}

function StickyNote({ side, accent, label }) {
  const isRight = side === "right";
  return (
    <span
      className={`absolute top-8 z-20 select-none px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wider leading-none
        shadow-[2px_4px_10px_rgba(41,36,29,0.22)] ${isRight ? "-right-14 rotate-[2.5deg]" : "-left-14 -rotate-[2.5deg]"}`}
      style={{
        background: `color-mix(in srgb, ${accent} 15%, var(--color-surface))`,
        color: accent,
        [isRight ? "borderRight" : "borderLeft"]: `3px solid ${accent}`,
      }}
    >
      {label}
    </span>
  );
}

function Page({ children, className = "" }) {
  return <div className={`px-10 pt-5 pb-3 transition-colors duration-700 ${className}`}>{children}</div>;
}

function Head({ chunk, score, accent, align = "left" }) {
  return (
    <div className={`mb-2 ${align === "right" ? "text-right" : ""}`}>
      <div className="font-mono text-[10px] text-(--color-ink-faint)">
        {chunk.chunk_id} · p.{chunk.page_start}
        {chunk.page_end !== chunk.page_start ? `–${chunk.page_end}` : ""}
        {score != null && (
          <span className="ml-2" style={{ color: accent }}>
            {score.toFixed(2)}
          </span>
        )}
      </div>
      {chunk.heading_path?.length > 0 && (
        <div className="mt-1 font-mono text-[11px] uppercase tracking-wide text-(--color-brass)">
          {chunk.heading_path.join(" / ")}
        </div>
      )}
    </div>
  );
}

function Absent() {
  return (
    <p className="font-serif-text italic text-[14px] text-(--color-ink-faint) py-2">
      No corresponding clause on this page.
    </p>
  );
}
