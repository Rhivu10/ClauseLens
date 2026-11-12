import { STATUS_LABEL, STATUS_COLOR } from "./rows";
import HighlightedText from "./HighlightedText";

// The alignment view drawn as an open legal book: left page = document A, right
// page = document B, meeting at a shaded central spine. The book has visible stacked page
// edges; each clause pair's match status is a sticky note poking out of the
// nearest fore-edge, and the drift note is an editor's footnote in the gutter.

const SPINE =
  "linear-gradient(90deg, transparent 0%, rgba(41,36,29,0.04) 40%, rgba(41,36,29,0.13) 50%, rgba(41,36,29,0.04) 60%, transparent 100%)";

const PAGE_EDGE =
  "repeating-linear-gradient(180deg, var(--color-surface) 0 2px, color-mix(in srgb, var(--color-line) 65%, transparent) 2px 3.5px)";

export default function SideBySideDiff({ documents, rows, highlight }) {
  return (
    <div className="relative mx-auto max-w-4xl my-2">
      {/* stacked sheets peeking behind — thickness */}
      <div className="absolute inset-0 translate-x-[6px] translate-y-[8px] rounded-lg bg-(--color-surface-2) border border-(--color-line)" />
      <div className="absolute inset-0 translate-x-[3px] translate-y-[4px] rounded-lg bg-(--color-surface) border border-(--color-line)" />

      {/* book cover */}
      <div className="relative rounded-lg bg-(--color-surface-2) p-2 border border-(--color-line) shadow-[0_34px_80px_-32px_rgba(41,36,29,0.45)]">
        {/* fore-edge page striping */}
        <div className="absolute top-3 bottom-3 -left-[3px] w-[5px] rounded-l-sm" style={{ background: PAGE_EDGE }} />
        <div className="absolute top-3 bottom-3 -right-[3px] w-[5px] rounded-r-sm" style={{ background: PAGE_EDGE }} />

        {/* the open spread */}
        <div className="relative rounded-[3px] bg-(--color-surface) border border-(--color-line)">
          {/* central spine */}
          <div
            className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-[64px] pointer-events-none"
            style={{ background: SPINE }}
          />
          <div className="absolute inset-y-0 left-1/2 w-px bg-(--color-ink)/20 pointer-events-none" />

          {/* running heads */}
          <div className="grid grid-cols-2 border-b border-(--color-line) font-mono text-[10px] uppercase tracking-[0.2em] text-(--color-ink-faint)">
            <div className="pl-10 pr-16 py-3.5 truncate">{documents.document_a.label}</div>
            <div className="pl-16 pr-10 py-3.5 text-right truncate">{documents.document_b.label}</div>
          </div>

          {rows.map((row) => {
            const accent = STATUS_COLOR[row.status];
            const paired = row.chunkA && row.chunkB;
            const noteSide = row.chunkB ? "right" : "left";
            return (
              <section
                key={row.id}
                id={`change-${row.change_id}`}
                className={`relative grid grid-cols-2 border-t border-(--color-line) first-of-type:border-t-0 scroll-mt-24 transition-colors duration-700 ${
                  highlight && highlight === row.change_id ? "bg-(--color-brass)/10" : ""
                }`}
              >
                <StickyNote side={noteSide} accent={accent} label={STATUS_LABEL[row.status]} />

                <Page side="left">
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

                <Page side="right">
                  {row.chunkB ? (
                    <>
                      <Head chunk={row.chunkB} accent={accent} align="right" />
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

                {/* editor's footnote across the gutter: the Node 5 impact summary */}
                <div className="col-span-2 px-10 pt-1 pb-6">
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
              </section>
            );
          })}
        </div>
      </div>
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
        shadow-[2px_4px_10px_rgba(41,36,29,0.22)] ${isRight ? "-right-6 rotate-[2.5deg]" : "-left-6 -rotate-[2.5deg]"}`}
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

function Page({ side, children }) {
  const pad = side === "left" ? "pl-10 pr-16" : "pl-16 pr-10";
  return <div className={`${pad} pt-7 pb-3`}>{children}</div>;
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
