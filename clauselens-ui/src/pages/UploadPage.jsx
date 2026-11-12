import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { getJob, getSetup, startJob } from "@/lib/api";
import { HoverBorderGradient } from "@/components/ui/HoverBorderGradient";
import ParsingOverlay from "@/components/ParsingOverlay";
import { Dropzone, DropzoneContent, DropzoneEmptyState } from "@/components/ui/Dropzone";

const PDF = { "application/pdf": [".pdf"] };
const MAX_PDF = 50 * 1024 * 1024;

// One document: a label, then a dropzone that takes one PDF
function Slot({ file, onFile, label }) {
  const [error, setError] = useState(null);
  return (
    <div className="flex-1 flex flex-col gap-2">
      <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-(--color-ink-faint)">{label}</div>
      <Dropzone
        accept={PDF}
        maxFiles={1}
        maxSize={MAX_PDF}
        src={file ? [file] : undefined}
        onDrop={(files) => {
          setError(null);
          onFile(files[0]);
        }}
        onError={(e) =>
          setError(/type/i.test(e.message) ? "That isn't a PDF. Choose a .pdf file." : /larger|too large/i.test(e.message) ? "That PDF is over 50 MB." : e.message)
        }
        className="min-h-[260px]"
      >
        <DropzoneEmptyState />
        <DropzoneContent />
      </Dropzone>
      {error && <p className="text-[13px] text-(--color-diverge)">{error}</p>}
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
      .then((r) => r.job?.state === "running" && r.job.type !== "consult" && setRun(r))
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
        <HoverBorderGradient disabled={!ready} onClick={handleRun}>
          {sending ? "Sending to Kaggle…" : "Run comparison"}
        </HoverBorderGradient>
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
