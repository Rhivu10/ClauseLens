import { useEffect, useState } from "react";
import Magnifier from "@/components/Magnifier";
import { motion, AnimatePresence } from "framer-motion";

// Shown while a comparison runs on Kaggle: a "detective going through the case
// files" beat — a magnifier working over a riffling stack of document pages —
// with the real pipeline stage from the Kaggle notebook underneath. A full run
// takes about 7–8 minutes for a 1-page document and ~16 minutes for a 20-page
// contract; the changed clauses show up after ~4–7 minutes (live preview).

const LENS_LOOP = 3.6; // seconds per magnifier sweep
const PAGE_EVERY_MS = 1400; // a new page slides in this often
const PAGE_SLIDE_S = 0.7; // how long the slide takes

function elapsed(since) {
  const s = Math.max(0, Math.round((Date.now() - new Date(since).getTime()) / 1000));
  const m = Math.floor(s / 60);
  return m ? `${m} min ${String(s % 60).padStart(2, "0")} s` : `${s} s`;
}

function Page({ seed }) {
  // deterministic pseudo-random line widths so each page looks different
  const rand = (n) => {
    const x = Math.sin(seed * 12.9898 + n * 78.233) * 43758.5453;
    return x - Math.floor(x);
  };
  const lines = Array.from({ length: 11 }, (_, i) => 52 + Math.round(rand(i) * 44));

  return (
    <div className="w-full h-full rounded-[3px] bg-(--color-surface) border border-(--color-line) shadow-[0_30px_80px_-30px_rgba(41,36,29,0.5)] px-8 py-9">
      <div className="h-2.5 w-1/3 rounded-full bg-(--color-brass)/45 mb-4" />
      <div className="h-px w-full bg-(--color-line) mb-5" />
      <div className="flex flex-col gap-3">
        {lines.map((w, i) => (
          <div
            key={i}
            className="h-2 rounded-full bg-(--color-ink)/12"
            style={{ width: `${w}%` }}
          />
        ))}
      </div>
    </div>
  );
}

function Lens() {
  // lens centred at (72, 72) in a 160 x 160 box, handle to the lower right
  return (
    <div className="relative" style={{ width: 160, height: 160 }}>
      <Magnifier cx={72} cy={72} r={61} shadow="0 8px 20px rgba(41,36,29,0.35)" />
    </div>
  );
}

export default function ParsingOverlay({ job, stages, onClose, onPreview }) {
  const [page, setPage] = useState(0);
  const [, tick] = useState(0);

  useEffect(() => {
    const riffle = setInterval(() => setPage((p) => p + 1), PAGE_EVERY_MS);
    const clock = setInterval(() => tick((t) => t + 1), 1000);
    return () => {
      clearInterval(riffle);
      clearInterval(clock);
    };
  }, []);

  const failed = job?.state === "error";
  const index = Math.max(0, stages.findIndex((s) => s[0] === job?.stage));
  const label = stages[index]?.[1] ?? "Working…";
  const progress = Math.round(((index + 1) / stages.length) * 100);

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-50 bg-(--color-bg) flex flex-col items-center justify-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.45 }}
      >
        <div className="font-mono-label text-[11px] uppercase tracking-[0.2em] text-(--color-ink-faint) mb-8">
          {failed ? "The run stopped" : "Examining the documents on Kaggle"}
        </div>

        {/* the case-file stack + magnifier */}
        <div className="relative w-[340px] h-[420px]">
          {/* under-stack peeking out for depth */}
          <div className="absolute inset-0 translate-x-3 translate-y-3 rounded-[3px] bg-(--color-surface-2) border border-(--color-line)" />
          <div className="absolute inset-0 translate-x-1.5 translate-y-1.5 rounded-[3px] bg-(--color-surface) border border-(--color-line)" />

          <AnimatePresence>
            <motion.div
              key={failed ? "stopped" : page}
              className="absolute inset-0"
              initial={{ x: 130, rotate: 5, opacity: 0 }}
              animate={{ x: 0, rotate: 0, opacity: 1 }}
              exit={{ x: -150, rotate: -7, opacity: 0 }}
              transition={{ duration: PAGE_SLIDE_S, ease: "easeInOut" }}
            >
              <Page seed={failed ? 0 : page} />
            </motion.div>
          </AnimatePresence>

          {!failed && (
            <motion.div
              className="absolute left-1/2 top-1/2 -ml-20 -mt-20 pointer-events-none"
              animate={{
                x: [-70, 60, -30, 55, -60, -70],
                y: [-90, -30, 40, 95, -10, -90],
                rotate: [-6, 4, -3, 6, -5, -6],
              }}
              transition={{ duration: LENS_LOOP, ease: "easeInOut", repeat: Infinity }}
            >
              <Lens />
            </motion.div>
          )}
        </div>

        {/* status + progress */}
        <div className="mt-10 w-[380px]">
          {failed ? (
            <>
              <p className="font-mono text-[13px] text-(--color-diverge) mb-2 whitespace-pre-wrap break-words">{job.error}</p>
              <div className="flex items-center gap-4 mt-4">
                <button
                  onClick={onClose}
                  className="px-4 py-2 rounded-full bg-(--color-surface-2) border border-(--color-line) text-sm font-medium hover:border-(--color-brass)"
                >
                  Back to upload
                </button>
                {job.kernel_url && (
                  <a href={job.kernel_url} target="_blank" rel="noreferrer" className="text-sm text-(--color-brass) underline underline-offset-4">
                    Open the notebook log on Kaggle
                  </a>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="flex items-center gap-2 mb-3 h-5">
                <span className="w-1.5 h-1.5 rounded-full bg-(--color-brass) animate-pulse" />
                <AnimatePresence mode="wait">
                  <motion.span
                    key={label}
                    className="font-mono text-[13px] text-(--color-ink-dim)"
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.25 }}
                  >
                    {label}
                  </motion.span>
                </AnimatePresence>
              </div>
              <div className="h-[3px] w-full rounded-full bg-(--color-surface-2) overflow-hidden">
                <motion.div
                  className="h-full bg-(--color-brass)"
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.6, ease: "easeOut" }}
                />
              </div>
              <div className="mt-3 flex items-center justify-between font-mono text-[11px] text-(--color-ink-faint)">
                <span>{job?.started_at ? elapsed(job.started_at) : ""} · usually 7–16 min · changes shown after ~4–7 min</span>
                {job?.kernel_url && (
                  <a href={job.kernel_url} target="_blank" rel="noreferrer" className="underline underline-offset-4 hover:text-(--color-ink)">
                    view on Kaggle
                  </a>
                )}
              </div>
              {job?.preview_ready && onPreview && (
                <button
                  onClick={onPreview}
                  className="mt-5 w-full px-4 py-2.5 rounded-full bg-(--color-brass) text-(--color-brass-ink) text-sm font-medium hover:opacity-90"
                >
                  See the changes found so far
                  {job.progress ? ` · ${job.progress.done} of ${job.progress.total} summaries ready` : ""} →
                </button>
              )}
              <p className="mt-4 text-[12px] text-(--color-ink-faint)">
                You can close this tab — the run keeps going on Kaggle, and the report opens here when you come back.
              </p>
            </>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
