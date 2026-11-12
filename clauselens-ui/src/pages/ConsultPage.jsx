import { useEffect, useRef, useState } from "react";
import { useReport } from "@/lib/api";
import CitationChip from "@/components/CitationChip";
import { EmptyState } from "@/pages/ComparePage";

// Ask about the latest report. Answers are found, not generated: the question is
// matched against every change (title, section, impact summary, parties and the
// clause text of both versions) and the best matches are shown with their
// sources. Nothing is said that is not in the report.

const SUGGESTIONS = [
  "What changed about royalties?",
  "Which changes need human review?",
  "Which duties changed?",
];

const STOP = new Set(
  "the a an and or of to in on for with by is are was were be what which who how does did do that this these those about any all changed change changes clause clauses version new old there it its from as at".split(" ")
);

const words = (text) =>
  (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9%$ ]+/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOP.has(w));

// crude stem so "royalties" finds "royalty", "terminated" finds "termination"
const stem = (w) => w.replace(/(ies|ied)$/, "y").replace(/(ing|ion|ions|ed|es|s)$/, "");

function search(question, rows) {
  const q = question.toLowerCase();
  if (/review|unsure|uncertain|low confidence/.test(q)) return { rows: rows.filter((r) => r.needs_review), how: "need human review" };
  if (/dut(y|ies)|obligation|shall|must|permission|prohibit/.test(q)) {
    const hits = rows.filter((r) => r.duty_shift);
    if (hits.length) return { rows: hits, how: "change a duty" };
  }
  const terms = [...new Set(words(question).map(stem))];
  if (!terms.length) return { rows: [], how: "" };
  const scored = rows
    .map((r) => {
      const strong = words([r.title, r.section, r.note, (r.affected || []).join(" ")].join(" ")).map(stem);
      const weak = words([r.chunkA?.text, r.chunkB?.text].join(" ")).map(stem);
      const score = terms.reduce((s, t) => s + (strong.includes(t) ? 2 : 0) + (weak.includes(t) ? 1 : 0), 0);
      return { r, score };
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score);
  return { rows: scored.slice(0, 4).map((x) => x.r), how: "match your question" };
}

function answer(question, data) {
  const { rows, how } = search(question, data.rows);
  if (!rows.length)
    return {
      role: "assistant",
      text: `No change in the report matches that. The report lists the ${data.rows.length} clauses that changed; ${data.overview.unchanged_clauses} clauses are the same in both versions.`,
    };
  return {
    role: "assistant",
    text: `${rows.length} change${rows.length === 1 ? "" : "s"} ${how}:`,
    items: rows.map((r) => ({
      id: r.change_id,
      title: r.title,
      section: r.section,
      text: r.note,
      review: r.needs_review,
      citations: [r.chunkA, r.chunkB].filter(Boolean).map((c) => ({
        chunk_id: c.chunk_id,
        document: c.chunk_id.startsWith("document_a") ? "A" : "B",
        page: c.page_start,
      })),
    })),
  };
}

export default function ConsultPage() {
  const { status, data, error } = useReport();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  if (status !== "ready") return <EmptyState status={status} error={error} />;

  function send(text) {
    const q = text ?? input;
    if (!q.trim()) return;
    setMessages((m) => [...m, { role: "user", text: q }, answer(q, data)]);
    setInput("");
  }

  const intro = {
    role: "assistant",
    text: `${data.documents.document_a.label} vs ${data.documents.document_b.label}: ${data.overview.headline}. Ask about a topic, a party or a clause — I'll show the matching changes from the report with their sources.`,
  };

  return (
    <div className="max-w-3xl mx-auto px-6 py-8 flex flex-col h-[calc(100vh-4rem)]">
      <div className="mb-5">
        <h1 className="font-serif text-3xl">Consult the report</h1>
        <p className="text-(--color-ink-dim) text-sm mt-1">
          Search the changes between the two versions. Every answer is taken from
          the report and cites the clause and page it comes from.
        </p>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-5 pr-1">
        {[intro, ...messages].map((m, i) => (
          <Message key={i} message={m} />
        ))}
      </div>

      <div className="pt-4">
        {messages.length === 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => send(s)}
                className="text-xs px-3 py-1.5 rounded-full border border-(--color-line) text-(--color-ink-dim) hover:border-(--color-brass) hover:text-(--color-brass-dim) transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
        )}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
          className="flex items-center gap-2 rounded-full border border-(--color-line) bg-(--color-surface) pl-4 pr-2 py-2 focus-within:border-(--color-brass) transition-colors"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about the changes…"
            className="flex-1 bg-transparent outline-none font-serif-text text-[15px] placeholder:text-(--color-ink-faint)"
          />
          <button
            type="submit"
            disabled={!input.trim()}
            className="w-9 h-9 rounded-full bg-(--color-brass) text-(--color-brass-ink) flex items-center justify-center hover:opacity-90 disabled:opacity-30 disabled:cursor-not-allowed shrink-0"
            aria-label="Send"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path
                d="M2 8h11m0 0L8 3m5 5l-5 5"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
}

function Message({ message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[85%] flex flex-col gap-2 ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`px-4 py-3 rounded-2xl font-serif-text text-[15px] leading-relaxed ${
            isUser
              ? "bg-(--color-brass) text-(--color-brass-ink) rounded-br-sm"
              : "bg-(--color-surface) border border-(--color-line) rounded-bl-sm"
          }`}
        >
          {message.text}
          {message.items?.map((it) => (
            <div key={it.id} className="mt-3 pt-3 border-t border-(--color-line) first-of-type:mt-3">
              <div className="font-mono text-[10.5px] uppercase tracking-wide text-(--color-ink-faint)">
                {it.id} · {it.section}
                {it.review && <span className="text-(--color-diverge)"> · needs review</span>}
              </div>
              <div className="font-medium text-[14px] mt-0.5">{it.title}</div>
              <div className="text-[14px] text-(--color-ink-dim) mt-1">{it.text}</div>
              <div className="flex flex-wrap gap-2 mt-2">
                {it.citations.map((c) => (
                  <CitationChip key={c.chunk_id} citation={c} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
