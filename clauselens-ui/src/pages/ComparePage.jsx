import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import SideBySideDiff from "./compare/SideBySideDiff";
import { useLiveReport } from "@/lib/api";
import MagneticButton from "@/components/MagneticButton";

function Legend({ swatch, label }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="w-3 h-3 rounded-[3px]" style={{ background: `color-mix(in srgb, ${swatch} 18%, transparent)` }} />
      {label}
    </span>
  );
}

const FILTERS = [
  ["all", "All", () => true],
  ["pending", "In progress", (r) => r.pending],
  ["review", "Needs review", (r) => r.needs_review && !r.pending],
  ["diverge", "Changed", (r) => r.status === "diverge"],
  ["missing_in_a", "Added", (r) => r.status === "missing_in_a"],
  ["missing_in_b", "Removed", (r) => r.status === "missing_in_b"],
];

export function EmptyState({ status, error }) {
  return (
    <div className="max-w-xl mx-auto text-center py-24">
      {status === "loading" ? (
        <p className="font-mono text-sm text-(--color-ink-faint)">Loading the latest report…</p>
      ) : status === "running" ? (
        <>
          <h1 className="font-serif text-3xl mb-3">Comparison running on Kaggle</h1>
          <p className="text-(--color-ink-dim) mb-8">
            The changed clauses appear here as soon as Nodes 1–4 are done (about 4–7 minutes after
            the start), and each impact summary as it is written. This page updates by itself.
          </p>
          <Link to="/upload" className="text-sm text-(--color-brass) underline underline-offset-4">
            See the progress
          </Link>
        </>
      ) : (
        <>
          <h1 className="font-serif text-3xl mb-3">
            {status === "error" ? "The report could not be loaded" : "No comparison yet"}
          </h1>
          <p className="text-(--color-ink-dim) mb-8">
            {status === "error"
              ? error
              : "Upload two versions of a contract. ClauseLens runs the models on Kaggle and the results appear here."}
          </p>
          <Link to="/upload">
            <MagneticButton>Upload two PDFs</MagneticButton>
          </Link>
        </>
      )}
    </div>
  );
}

export default function ComparePage() {
  const { status, data, error, live } = useLiveReport();
  const [filter, setFilter] = useState("all");
  // ?change=C012 (from a Consult citation): show that change, scrolled into view
  const [params] = useSearchParams();
  const target = params.get("change");
  const [highlight, setHighlight] = useState(null);
  useEffect(() => {
    if (status !== "ready" || !target) return;
    setFilter("all");
    setHighlight(target);
    const t = setTimeout(() => document.getElementById(`change-${target}`)?.scrollIntoView({ behavior: "smooth", block: "start" }), 300);
    const off = setTimeout(() => setHighlight(null), 4000);
    return () => {
      clearTimeout(t);
      clearTimeout(off);
    };
  }, [status, target]);
  if (status !== "ready") return <EmptyState status={status} error={error} />;

  const { overview, documents } = data;
  const filters = FILTERS.filter(([key]) => key !== "pending" || live);
  const test = (filters.find((f) => f[0] === filter) || filters[0])[2];
  const rows = data.rows.filter(test);

  return (
    <div className="max-w-7xl mx-auto px-6 py-10">
      <div className="max-w-4xl mx-auto text-center mb-7">
        <h1 className="font-serif text-3xl">
          Where the two versions agree — and where they don't
        </h1>
        {live ? (
          <div className="mt-3 inline-flex items-center gap-2 rounded-full border border-(--color-brass)/40 bg-(--color-surface) px-4 py-1.5 text-sm">
            <span className="w-1.5 h-1.5 rounded-full bg-(--color-brass) animate-pulse" />
            {data.rows.length} changes found · impact summaries {live.done} of {live.total} ready · this page updates by itself
          </div>
        ) : (
          <p className="mt-2 text-(--color-ink-dim) text-sm">
            {overview.headline} · {overview.unchanged_clauses} clauses unchanged
          </p>
        )}
        {data.run?.cached && (
          <p className="mt-1.5 font-mono text-[11px] text-(--color-ink-faint)">
            Saved result — these two PDFs were compared on{" "}
            {new Date(data.run.finished_at).toLocaleString()}. Tick "Run again on Kaggle" on the upload page for a fresh run.
          </p>
        )}
        <div className="mt-3 flex items-center justify-center gap-5 text-xs font-mono text-(--color-ink-dim)">
          <Legend swatch="var(--color-diverge)" label="removed in B" />
          <Legend swatch="var(--color-cyan)" label="added in B" />
        </div>
        <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
          {filters.map(([key, label, f]) => (
            <button
              key={key}
              onClick={() => setFilter(key)}
              className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
                filter === key
                  ? "border-(--color-brass) text-(--color-brass-dim) bg-(--color-surface-2)"
                  : "border-(--color-line) text-(--color-ink-dim) hover:border-(--color-brass)"
              }`}
            >
              {label} <span className="font-mono text-(--color-ink-faint)">{data.rows.filter(f).length}</span>
            </button>
          ))}
        </div>
      </div>

      {rows.length ? (
        <SideBySideDiff documents={documents} rows={rows} highlight={highlight} />
      ) : (
        <p className="text-center font-serif-text italic text-(--color-ink-faint) py-12">No changes in this group.</p>
      )}
    </div>
  );
}
