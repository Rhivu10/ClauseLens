import { Fragment, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { askConsult, getConsult, useReport } from "@/lib/api";
import CitationChip from "@/components/CitationChip";
import { HoverBorderGradient } from "@/components/ui/HoverBorderGradient";
import { EmptyState } from "@/pages/ComparePage";

// Ask about the latest report. Gemma (pipeline/consult.py) answers on Kaggle, using
// only the changes in the report, and cites them as [C012]; citations it was not
// given are removed and an answer without any is marked "not supported".
// While Gemma works, the matching changes are found here at once (a word search
// over the report). Answers are saved per report, so a question asked before is
// answered instantly; questions asked while a batch runs wait for the next batch.

const POLL_MS = 5000;

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

// "cited", "nothing" (the report has no such change: fine without a citation) or
// "unsupported"; answers saved before "kind" existed get it from their text
const NOTHING = /does ?n[o']t show|not (shown|mentioned|listed|covered|in the report)|no (listed )?changes?\b|none of the (listed )?changes|zero changes|there (are|were) no/i;
const kindOf = (a) => a.kind || (a.citations.length ? "cited" : NOTHING.test(a.answer) ? "nothing" : "unsupported");

const norm = (q) => q.trim().toLowerCase().replace(/\s+/g, " ").replace(/[?.!]+$/, "");

// The changes that match a question, shown while Gemma answers
function related(question, all) {
  const rows = all.filter((r) => r.change_id);
  const q = question.toLowerCase();
  if (/review|unsure|uncertain|low confidence/.test(q)) return rows.filter((r) => r.needs_review).slice(0, 4);
  if (/dut(y|ies)|obligation|shall|must|permission|prohibit/.test(q)) {
    const hits = rows.filter((r) => r.duty_shift);
    if (hits.length) return hits.slice(0, 4);
  }
  const terms = [...new Set(words(question).map(stem))];
  if (!terms.length) return [];
  return rows
    .map((r) => {
      const strong = words([r.title, r.section, r.note, (r.affected || []).join(" ")].join(" ")).map(stem);
      const weak = words([r.chunkA?.text, r.chunkB?.text].join(" ")).map(stem);
      const score = terms.reduce((s, t) => s + (strong.includes(t) ? 2 : 0) + (weak.includes(t) ? 1 : 0), 0);
      return { r, score };
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 4)
    .map((x) => x.r);
}

// Questions that fit this report: built from what is in it
function suggestionsFor(data) {
  const out = [];
  const rows = data.rows.filter((r) => r.change_id);
  if (rows.some((r) => r.needs_review)) out.push("Which changes need human review?");
  if (rows.some((r) => r.duty_shift)) out.push("Which duties changed, and for whom?");
  if (rows.some((r) => /delet|remov/i.test(r.type || ""))) out.push("What was removed from the new version?");
  const topic = rows.find((r) => r.title && r.title.length < 40);
  if (topic) out.push(`What changed in "${topic.title}"?`);
  out.push("What are the most important changes for me?");
  return out.slice(0, 4);
}

export default function ConsultPage() {
  const { status, data, error } = useReport();
  const [consult, setConsult] = useState(null);     // server state: answers, queued, asking, job, stages
  const [mine, setMine] = useState([]);             // questions asked on this page, in order
  const [input, setInput] = useState("");
  const [sendError, setSendError] = useState(null);
  const scrollRef = useRef(null);

  const pending = consult ? [...consult.asking, ...consult.queued] : [];
  const working = consult?.job?.state === "running" || pending.length > 0;

  useEffect(() => {
    let alive = true;
    let timer;
    const load = () =>
      getConsult()
        .then((c) => alive && setConsult(c))
        .catch(() => {})
        .finally(() => alive && (timer = setTimeout(load, POLL_MS)));
    load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);

  const answerCount = consult?.answers.length ?? 0;
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [mine.length, answerCount, pending.length]);

  if (status !== "ready") return <EmptyState status={status} error={error} />;

  async function send(text) {
    const q = (text ?? input).trim();
    if (!q) return;
    setInput("");
    setSendError(null);
    setMine((m) => (m.some((x) => norm(x) === norm(q)) ? m : [...m, q]));
    try {
      setConsult(await askConsult([q]));
    } catch (e) {
      setSendError(e.message);
    }
  }

  // The conversation: every saved answer for this report, then questions still waiting
  const answers = consult?.answers ?? [];
  const byQuestion = new Map(answers.map((a) => [norm(a.question), a]));
  const waiting = [...pending, ...mine].filter((q, i, all) => !byQuestion.has(norm(q)) && all.findIndex((x) => norm(x) === norm(q)) === i);
  const asking = new Set((consult?.asking ?? []).map(norm));
  const stageLabel = Object.fromEntries(consult?.stages ?? []);
  const job = consult?.job;
  const failed = job?.state === "error" ? job.error : null;
  const known = new Set(data.rows.map((r) => r.change_id));

  const intro = `${data.documents.document_a.label} vs ${data.documents.document_b.label}: ${data.overview.headline}. Ask anything about the changes. Gemma answers on Kaggle from the report only and cites the changes it uses. A new question takes a few minutes; matching changes appear right away.`;

  return (
    <div className="max-w-3xl mx-auto px-6 py-8 flex flex-col h-[calc(100vh-4rem)]">
      <div className="mb-5">
        <h1 className="font-serif text-3xl">Consult the report</h1>
        <p className="text-(--color-ink-dim) text-sm mt-1">
          Answers come from Gemma, using only the changes in this report. Click a
          citation like <span className="font-mono text-[12px]">C003</span> to see that clause side by side.
        </p>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-5 pr-1">
        <Bubble>{intro}</Bubble>

        {answers.map((a) => (
          <Fragment key={`a-${a.question}`}>
            <Bubble user>{a.question}</Bubble>
            <Bubble>
              <AnswerText text={a.answer} known={known} />
              {kindOf(a) === "unsupported" && (
                <div className="mt-2 text-[12.5px] text-(--color-diverge)">
                  Not supported: Gemma cited no change from the report, so treat this answer with care.
                </div>
              )}
              {a.citations.length > 0 && (
                <div className="mt-3 pt-3 border-t border-(--color-line) flex flex-wrap gap-2">
                  {a.citations.map((id) => (
                    <ChangeLink key={id} id={id} row={data.rows.find((r) => r.change_id === id)} />
                  ))}
                </div>
              )}
            </Bubble>
          </Fragment>
        ))}

        {waiting.map((q) => {
          const rows = related(q, data.rows);
          const now = asking.has(norm(q));
          return (
            <Fragment key={`w-${q}`}>
              <Bubble user>{q}</Bubble>
              <Bubble>
                <div className="flex items-center gap-2 text-[14px] text-(--color-ink-dim)">
                  {!failed && <span className="w-2 h-2 rounded-full bg-(--color-brass) animate-pulse shrink-0" />}
                  {failed
                    ? "Answering on Kaggle failed. Ask again to retry."
                    : now
                      ? `Gemma is answering on Kaggle${job?.stage && stageLabel[job.stage] ? ` · ${stageLabel[job.stage]}` : ""}${
                          job?.progress?.total ? ` · ${job.progress.done}/${job.progress.total}` : ""
                        }. This takes a few minutes.`
                      : working
                        ? "Waiting for the current batch to finish, then Gemma answers this one."
                        : "Sending to Kaggle…"}
                </div>
                {rows.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-(--color-line)">
                    <div className="font-mono text-[10.5px] uppercase tracking-wide text-(--color-ink-faint) mb-1">
                      Matching changes in the report
                    </div>
                    {rows.map((r) => (
                      <RowSummary key={r.change_id} row={r} />
                    ))}
                  </div>
                )}
              </Bubble>
            </Fragment>
          );
        })}
      </div>

      <div className="pt-4">
        {(sendError || failed) && (
          <div className="mb-3 text-[13px] text-(--color-diverge)">{sendError || `Last batch failed: ${failed}`}</div>
        )}
        {answers.length === 0 && waiting.length === 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {suggestionsFor(data).map((s) => (
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
          <HoverBorderGradient
            type="submit"
            disabled={!input.trim()}
            containerClassName="shrink-0"
            className="!p-0 w-9 h-9"
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
          </HoverBorderGradient>
        </form>
      </div>
    </div>
  );
}

function Bubble({ user = false, children }) {
  return (
    <div className={`flex ${user ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] px-4 py-3 rounded-2xl font-serif-text text-[15px] leading-relaxed ${
          user
            ? "bg-(--color-brass) text-(--color-brass-ink) rounded-br-sm"
            : "bg-(--color-surface) border border-(--color-line) rounded-bl-sm"
        }`}
      >
        {children}
      </div>
    </div>
  );
}

// The answer with each [C012] turned into a link to that change
function AnswerText({ text, known }) {
  return (
    <div className="whitespace-pre-line">
      {text.split(/(\[C\d{3,}\])/).map((part, i) => {
        const id = part.match(/^\[(C\d{3,})\]$/)?.[1];
        if (!id || !known.has(id)) return <Fragment key={i}>{part}</Fragment>;
        return (
          <Link
            key={i}
            to={`/compare?change=${id}`}
            className="font-mono text-[12px] px-1.5 py-0.5 mx-0.5 rounded border border-(--color-brass)/50 text-(--color-brass-dim) hover:bg-(--color-brass)/10 no-underline align-[1px]"
          >
            {id}
          </Link>
        );
      })}
    </div>
  );
}

function ChangeLink({ id, row }) {
  if (!row) return null;
  return (
    <Link
      to={`/compare?change=${id}`}
      className="text-[12.5px] px-2.5 py-1 rounded-full border border-(--color-line) hover:border-(--color-brass) text-(--color-ink-dim) hover:text-(--color-brass-dim) transition-colors"
    >
      <span className="font-mono text-[11px]">{id}</span> · {row.title}
    </Link>
  );
}

function RowSummary({ row }) {
  return (
    <div className="mt-2">
      <Link to={`/compare?change=${row.change_id}`} className="group block">
        <div className="font-mono text-[10.5px] uppercase tracking-wide text-(--color-ink-faint)">
          {row.change_id} · {row.section}
          {row.needs_review && <span className="text-(--color-diverge)"> · needs review</span>}
        </div>
        <div className="font-medium text-[14px] mt-0.5 group-hover:text-(--color-brass-dim)">{row.title}</div>
      </Link>
      {row.note && <div className="text-[14px] text-(--color-ink-dim) mt-1">{row.note}</div>}
      <div className="flex flex-wrap gap-2 mt-2">
        {[row.chunkA, row.chunkB].filter(Boolean).map((c) => (
          <CitationChip
            key={c.chunk_id}
            citation={{ chunk_id: c.chunk_id, document: c.chunk_id.startsWith("document_a") ? "A" : "B", page: c.page_start }}
          />
        ))}
      </div>
    </div>
  );
}
