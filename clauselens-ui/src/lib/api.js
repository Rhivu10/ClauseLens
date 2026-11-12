// Client for the ClauseLens API (server/api.js, served by the Vite dev server).
import { useEffect, useState } from "react";

async function request(url, options) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(body.error || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return body;
}

function toBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

export const getSetup = () => request("/api/setup");
export const getJob = () => request("/api/job");
export const getReport = () => request("/api/report");
export const getPartial = () => request("/api/partial");

// force: run on Kaggle even if these two PDFs were compared before (otherwise the
// saved report comes back at once, with job.state "done" and job.cached true)
export async function startJob(fileA, fileB, force = false) {
  const [a, b] = await Promise.all([toBase64(fileA), toBase64(fileB)]);
  return request("/api/job", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      document_a: { name: fileA.name, data: a },
      document_b: { name: fileB.name, data: b },
      force,
    }),
  });
}

// The latest finished report, with each alignment's chunks resolved.
// state: "loading" | "ready" | "empty" | "error"
export function useReport() {
  const [state, setState] = useState({ status: "loading", data: null, error: null });
  useEffect(() => {
    let alive = true;
    getReport()
      .then((data) => alive && setState({ status: "ready", data: withRows(data), error: null }))
      .catch((e) => alive && setState({ status: e.status === 404 ? "empty" : "error", data: null, error: e.message }));
    return () => {
      alive = false;
    };
  }, []);
  return state;
}

// Like useReport, but while a comparison runs it shows the report as it fills in
// (changes first, impact summaries as they arrive) and refreshes itself.
// status adds "running": a run is going but has no preview yet.
const LIVE_POLL_MS = 8000;

export function useLiveReport() {
  const [state, setState] = useState({ status: "loading", data: null, error: null, live: null, job: null });
  useEffect(() => {
    let alive = true;
    let timer;
    const load = async () => {
      let job = null;
      try {
        job = (await getJob()).job;
      } catch {
        /* server restarting: try again */
      }
      if (!alive) return;
      if (job?.state === "running") {
        try {
          const p = await getPartial();
          if (alive) setState({ status: "ready", data: withRows(p.ui), error: null, live: { done: p.done, total: p.total }, job });
        } catch {
          if (alive)
            setState((s) => (s.live ? { ...s, job } : { status: "running", data: null, error: null, live: null, job }));
        }
        timer = setTimeout(load, LIVE_POLL_MS);
        return;
      }
      try {
        const data = await getReport();
        if (alive) setState({ status: "ready", data: withRows(data), error: null, live: null, job });
      } catch (e) {
        if (alive) setState({ status: e.status === 404 ? "empty" : "error", data: null, error: e.message, live: null, job });
      }
    };
    load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);
  return state;
}

function withRows(data) {
  const chunkById = {};
  Object.values(data.chunks || {}).forEach((list) => list.forEach((c) => (chunkById[c.chunk_id] = c)));
  const rows = (data.alignments || []).map((al) => ({
    ...al,
    chunkA: al.a ? chunkById[al.a] : null,
    chunkB: al.b ? chunkById[al.b] : null,
  }));
  return { ...data, chunkById, rows };
}
