import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { getJob, getSetup, startJob } from "@/lib/api";
import MagneticButton from "@/components/MagneticButton";
import ParsingOverlay from "@/components/ParsingOverlay";

function Slot({ file, onFile, label }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files?.[0];
        if (f && f.type === "application/pdf") onFile(f);
      }}
      className={`relative flex-1 min-h-[320px] rounded-2xl border transition-colors
        ${dragging ? "border-(--color-cyan) bg-(--color-cyan)/5" : "border-(--color-line) bg-(--color-surface)"}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
      />
      <div className="absolute top-4 left-4 font-mono text-[11px] uppercase tracking-[0.14em] text-(--color-ink-faint)">
        {label}
      </div>
      <div className="h-full flex flex-col items-center justify-center gap-3 p-8 text-center">
        {file ? (
          <>
            <div className="w-14 h-18 rounded-sm bg-(--color-surface-2) border border-(--color-line) flex items-center justify-center">
              <span className="font-mono text-[10px] text-(--color-ink-faint)">PDF</span>
            </div>
            <div className="font-serif-text text-lg">{file.name}</div>
            <div className="font-mono text-xs text-(--color-ink-faint)">
              {(file.size / 1024).toFixed(0)} KB · ready to compare
            </div>
            <button
              onClick={() => inputRef.current.click()}
              className="mt-2 text-sm text-(--color-brass) underline underline-offset-4"
            >
              Replace file
            </button>
          </>
        ) : (
          <>
            <svg width="40" height="40" viewBox="0 0 40 40" aria-hidden="true">
              <circle cx="17" cy="17" r="12" fill="none" stroke="var(--color-ink-faint)" strokeWidth="2" />
              <line x1="25.5" y1="25.5" x2="35" y2="35" stroke="var(--color-ink-faint)" strokeWidth="2.4" strokeLinecap="round" />
            </svg>
            <div className="text-sm text-(--color-ink-faint) max-w-[24ch]">
              Choose a PDF from your device to place it on the table
            </div>
            <button
              onClick={() => inputRef.current.click()}
              className="mt-3 px-4 py-2 rounded-full bg-(--color-surface-2) border border-(--color-line) text-sm font-medium hover:border-(--color-brass) transition-colors"
            >
              Choose file
            </button>
          </>
        )}
      </div>
    </div>
  );
}

const POLL_MS = 5000;

export default function UploadPage() {
  const navigate = useNavigate();
  const [fileA, setFileA] = useState(null);
  const [fileB, setFileB] = useState(null);
  const [setup, setSetup] = useState(null);       // {ok, username} | {ok: false, error}
  const [run, setRun] = useState(null);           // {job, stages} while a comparison is shown
  const [error, setError] = useState(null);
  const [sending, setSending] = useState(false);
  const [force, setForce] = useState(false);     // run on Kaggle even if these PDFs were compared before
  const ready = fileA && fileB && setup?.ok && !sending;

  // Kaggle tool signed in? A run already going (e.g. after a page reload)?
  useEffect(() => {
    getSetup().then(setSetup).catch((e) => setSetup({ ok: false, error: e.message }));
    getJob()
      .then((r) => r.job?.state === "running" && setRun(r))
      .catch(() => {});
  }, []);

  // Follow the run until it is done or fails
  const running = run?.job?.state === "running";
  useEffect(() => {
    if (!running) return;
    const t = setInterval(async () => {
      try {
        const r = await getJob();
        setRun(r);
        if (r.job?.state === "done") navigate("/compare");
      } catch {
        /* dev server restarting: try again on the next tick */
      }
    }, POLL_MS);
    return () => clearInterval(t);
  }, [running, navigate]);

  const handleRun = async () => {
    setError(null);
    setSending(true);
    try {
      const r = await startJob(fileA, fileB, force);
      if (r.job?.state === "done") return navigate("/compare");   // same PDFs as before: saved report
      setRun(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      {run && (
        <ParsingOverlay job={run.job} stages={run.stages} onClose={() => setRun(null)} onPreview={() => navigate("/compare")} />
      )}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="max-w-2xl mb-10"
      >
        <h1 className="font-sans font-semibold text-4xl leading-tight mb-3">
          Place two versions of the same contract on the table.
        </h1>
        <p className="text-(--color-ink-dim)">
          ClauseLens sends both PDFs to a Kaggle GPU notebook, which parses and
          aligns every clause, checks duties, and writes an impact report for
          each change.
        </p>
      </motion.div>

      {setup && !setup.ok && (
        <div className="mb-6 rounded-xl border border-(--color-diverge)/40 bg-(--color-diverge)/5 px-5 py-4 text-sm">
          <div className="font-medium text-(--color-diverge) mb-1">Kaggle is not connected</div>
          <div className="text-(--color-ink-dim) whitespace-pre-wrap">{setup.error}</div>
          <div className="text-(--color-ink-faint) mt-2">See README.md → "Running comparisons on Kaggle", then reload this page.</div>
        </div>
      )}

      <div className="flex flex-col md:flex-row gap-6">
        <Slot file={fileA} onFile={setFileA} label="Document A · previous version" />
        <div className="hidden md:flex items-center justify-center w-10">
          <span className="font-mono text-xs text-(--color-ink-faint)">vs</span>
        </div>
        <Slot file={fileB} onFile={setFileB} label="Document B · new version" />
      </div>

      <div className="mt-8 flex items-center gap-5">
        <MagneticButton
          disabled={!ready}
          onClick={handleRun}
          className="disabled:opacity-30 disabled:cursor-not-allowed"
        >
          {sending ? "Sending to Kaggle…" : "Run comparison"}
        </MagneticButton>
        <Link
          to="/compare"
          className="text-sm text-(--color-ink-dim) underline underline-offset-4 hover:text-(--color-ink)"
        >
          View the last report
        </Link>
        {setup?.ok && (
          <span className="font-mono text-[11px] text-(--color-ink-faint)">Kaggle: {setup.username}</span>
        )}
      </div>
      <label className="mt-4 flex items-center gap-2 text-sm text-(--color-ink-dim) select-none w-fit cursor-pointer">
        <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} className="accent-(--color-brass)" />
        Run again on Kaggle, even if these two PDFs were compared before
      </label>
      {error && <p className="mt-4 text-sm text-(--color-diverge) whitespace-pre-wrap">{error}</p>}
    </div>
  );
}
