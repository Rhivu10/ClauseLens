import { STATUS_LABEL, STATUS_COLOR } from "./rows";
import HighlightedText from "./HighlightedText";

// The alignment view drawn as an open book (a photo, public/book.png): left page =
// document A, right page = document B, meeting at the spine. Each clause pair's
// match status is a sticky note poking out of the nearest fore-edge, and the
// impact summary is an editor's footnote across the gutter.

// Slices of public/book.png (625 x 476): cover + page stack on each side (equal, so
// the spine stays in the middle), page tops and bottoms. The middle stretches.
const BOOK = { top: 14, side: 40, bottom: 18 };

export default function SideBySideDiff({ documents, rows, highlight }) {
  return (
    <div className="relative mx-auto max-w-4xl my-2">
      {/* the open book (public/book.png): a 9-slice frame, so the page edges, cover
          and spine stay sharp while the book grows to fit every clause */}
      <div
        className="relative drop-shadow-[0_30px_40px_rgba(41,36,29,0.22)]"
        style={{
          borderStyle: "solid",
          borderWidth: `${BOOK.top}px ${BOOK.side}px ${BOOK.bottom}px`,
          borderImage: `url(/book.png) ${BOOK.top} ${BOOK.side} ${BOOK.bottom} fill / ${BOOK.top}px ${BOOK.side}px ${BOOK.bottom}px stretch`,
        }}
      >
        <div className="relative">
          {/* running heads */}
          <div className="grid grid-cols-2 border-b border-(--color-ink)/10 font-mono text-[10px] uppercase tracking-[0.2em] text-(--color-ink-faint)">
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
                className={`relative grid grid-cols-2 border-t border-(--color-ink)/10 first-of-type:border-t-0 scroll-mt-24 transition-colors duration-700 ${
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
