// ClauseLens — runs a comparison on Kaggle through the official Kaggle API (free).
//
//   1. the two PDFs + clauselens_job.json are uploaded as a new version of the
//      private dataset <user>/clauselens-input
//   2. server/kaggle/runner.py is pushed as the private notebook
//      <user>/clauselens-runner (GPU T4 x2, internet on), which starts a run
//   3. the run's log is read for "[ClauseLens] <stage>" progress markers
//   4. when the run is complete, report.json / ui_data.json are downloaded to
//      server/data/latest and the UI reads them from there
//
// Live results: while the notebook runs, its log is followed live
// (kaggle kernels logs -f). After Node 4 it prints a preview of the report
// (summaries "in progress") and during Node 5 each finished summary; they are
// kept in server/data/partial.json, which the Alignment page shows until the
// final report arrives.
//
// Consult: questions about the latest report run as a job of type "consult" on
// the same notebook (pipeline/consult.py, Gemma E4B). Answers are kept per report
// in server/data/consult/<report id>.json; questions asked while a batch runs are
// queued and go in the next batch; a question asked before is answered at once.
//
// Finished reports are also kept in server/data/cache, keyed by a fingerprint
// (SHA-256) of the two PDFs, the ClauseLens branch and runner.py. Uploading the
// same two PDFs again opens the saved report at once instead of running Kaggle
// (unless the upload asks for a fresh run).
//
// Uses the Kaggle command line tool (pip install kaggle) with the API token in
// ~/.kaggle/access_token (or ~/.kaggle/kaggle.json, or KAGGLE_API_TOKEN).

import { spawn } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import { marked } from "marked";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WORK = path.join(HERE, "work");
const DATA = path.join(HERE, "data");
const LATEST = path.join(DATA, "latest");
const STATE_FILE = path.join(DATA, "job.json");
const PARTIAL_FILE = path.join(DATA, "partial.json");
const CONSULT_DIR = path.join(DATA, "consult");
const CACHE = path.join(DATA, "cache");
const RUNNER = path.join(HERE, "kaggle", "runner.py");
const REPORT_FILES = ["report.json", "report.md", "ui_data.json", "status.json", "node5_results.jsonl"];

const PYTHON = process.env.CLAUSELENS_PYTHON || "python";
const DATASET_SLUG = process.env.CLAUSELENS_DATASET || "clauselens-input";
const KERNEL_SLUG = process.env.CLAUSELENS_KERNEL || "clauselens-runner";
// ClauseLens branch the notebook clones (e.g. to test a branch before it is merged)
const REPO_BRANCH = process.env.CLAUSELENS_REPO_BRANCH || "main";
// The CUAD adapter is in the repo (Model Finetuning/checkpoints/checkpoint-2091) and the
// runner loads it from there. Set this only to use another copy from a Kaggle dataset.
const CHECKPOINT_DATASET = process.env.CLAUSELENS_CHECKPOINT_DATASET || null;
const POLL_MS = Number(process.env.CLAUSELENS_POLL_MS) || 30_000;

// Stages, in order, with what the UI shows: the pipeline's stages (pipeline/graph.py)
// plus the Kaggle steps around them
export const STAGES = [
  ["uploading", "Uploading the PDFs to Kaggle"],
  ["dataset_processing", "Kaggle is preparing the upload"],
  ["starting", "Starting the notebook (GPU T4 x2)"],
  ["queued", "Waiting for a free Kaggle GPU"],
  ["started", "Notebook started"],
  ["installing", "Installing libraries"],
  ["node_1_parse", "Node 1 · reading both PDFs"],
  ["node_2_index", "Node 2 · indexing every clause"],
  ["node_3_loading_model", "Loading Gemma + CUAD adapter"],
  ["node_3_align", "Node 3 · aligning clauses"],
  ["node_4_deontic", "Node 4 · checking duties (shall / may / must not)"],
  ["preview_ready", "Changes found · impact summaries in progress"],
  ["node_5_loading_agents", "Node 5 · loading the Impact and Critic agents"],
  ["node_5_warmup", "Node 5 · warming up both models"],
  ["node_5_impact", "Node 5 · writing and checking impact summaries"],
  ["node_6_report", "Node 6 · assembling the report"],
  ["consult_loading_model", "Loading Gemma E4B to answer"],
  ["consult_answering", "Gemma is answering"],
  ["downloading", "Downloading the report"],
  ["done", "Report ready"],
];

let job = readJSON(STATE_FILE); // {id, state, stage, error, ...}; survives a dev-server restart
let username = null;
let timer = null;
let finishing = false; // the live log says the runner is at Node 6 or done: check status often

function readJSON(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

function writeJSON(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(value, null, 2));
}

function update(patch) {
  job = { ...job, ...patch, updated_at: new Date().toISOString() };
  writeJSON(STATE_FILE, job);
}

// Runs `python -m kaggle.cli ...`; resolves with stdout, rejects with a readable message
function kaggle(args, { cwd } = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(PYTHON, ["-m", "kaggle.cli", ...args], {
      cwd,
      env: { ...process.env, PYTHONIOENCODING: "utf-8" },
      windowsHide: true,
    });
    let out = "";
    let err = "";
    p.stdout.on("data", (d) => (out += d));
    p.stderr.on("data", (d) => (err += d));
    p.on("error", (e) =>
      reject(new Error(`Could not run "${PYTHON} -m kaggle.cli" (${e.message}). Install it with: pip install kaggle`))
    );
    p.on("close", (code) => {
      if (code === 0) return resolve(out);
      const msg = (err || out).trim();
      if (/No module named kaggle/i.test(msg)) return reject(new Error("The Kaggle tool is not installed. Run: pip install kaggle"));
      if (/Authentication required|401|Unauthorized/i.test(msg))
        return reject(new Error("Kaggle API token missing or invalid. Create one at kaggle.com/settings/api and save it as ~/.kaggle/access_token"));
      reject(new Error(msg.split("\n").slice(-6).join("\n") || `kaggle ${args[0]} ${args[1]} failed (exit ${code})`));
    });
  });
}

async function getUsername() {
  if (username) return username;
  if (process.env.KAGGLE_USERNAME) return (username = process.env.KAGGLE_USERNAME);
  const code =
    "from kaggle.api.kaggle_api_extended import KaggleApi\n" +
    "a = KaggleApi(); a.authenticate(); print(a.get_config_value('username'))";
  const out = await new Promise((resolve, reject) => {
    const p = spawn(PYTHON, ["-c", code], { windowsHide: true });
    let o = "";
    let e = "";
    p.stdout.on("data", (d) => (o += d));
    p.stderr.on("data", (d) => (e += d));
    p.on("error", (x) => reject(new Error(`Could not run ${PYTHON}: ${x.message}`)));
    p.on("close", (c) =>
      c === 0 && o.trim()
        ? resolve(o.trim())
        : reject(
            new Error(
              /No module named/i.test(e)
                ? "The Kaggle tool is not installed. Run: pip install kaggle"
                : "Kaggle API token missing or invalid. Create one at kaggle.com/settings/api and save it as ~/.kaggle/access_token"
            )
          )
    );
  });
  return (username = out.split(/\s+/).pop());
}

// Is the Kaggle tool installed and signed in? Used by the Upload page before a run.
export async function checkSetup() {
  try {
    return { ok: true, username: await getUsername() };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

export function currentJob() {
  return job;
}

// The report as it fills in during a run (null when no run is showing one)
export function partialReport() {
  const p = readJSON(PARTIAL_FILE);
  return p && job && p.job_id === job.id && job.state === "running" ? p : null;
}

export function latestReport() {
  const data = readJSON(path.join(LATEST, "ui_data.json"));
  return data && { ...data, run: readJSON(path.join(LATEST, "run.json")) };
}

// The latest finished report as a file to download:
//   md   report.md (the readable report Node 6 writes)
//   json report.json (everything, for other tools)
//   pdf  report.md as a styled page that opens the print dialog ("Save as PDF")
// Returns {filename, type, body, inline} or null when there is no report yet.
export function reportDownload(format) {
  const md = path.join(LATEST, "report.md");
  const json = path.join(LATEST, "report.json");
  if (!fs.existsSync(md) || !fs.existsSync(json)) return null;
  const docs = readJSON(json)?.documents || {};
  const names = [docs.document_a?.name, docs.document_b?.name].filter(Boolean);
  const base = `ClauseLens report - ${names.join(" vs ") || "latest"}`.replace(/[\\/:*?"<>|]+/g, "_");
  if (format === "json") return { filename: `${base}.json`, type: "application/json", body: fs.readFileSync(json) };
  if (format === "md") return { filename: `${base}.md`, type: "text/markdown; charset=utf-8", body: fs.readFileSync(md) };
  if (format === "pdf") {
    const html = marked.parse(fs.readFileSync(md, "utf8"));
    return { filename: `${base}.html`, type: "text/html; charset=utf-8", inline: true, body: printablePage(base, html) };
  }
  throw new Error(`Unknown format "${format}". Use md, json or pdf.`);
}

const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function printablePage(title, html) {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>${escapeHtml(title)}</title>
<style>
  @page { margin: 18mm 16mm; }
  body { font: 11pt/1.55 Georgia, "Times New Roman", serif; color: #29241d; max-width: 780px; margin: 32px auto; padding: 0 20px; }
  h1, h2, h3, h4 { font-family: system-ui, "Segoe UI", sans-serif; color: #29241d; line-height: 1.25; break-after: avoid; }
  h1 { font-size: 22pt; margin: 0 0 6px; } h2 { font-size: 15pt; margin-top: 28px; border-bottom: 1px solid #d8ccb6; padding-bottom: 4px; }
  h3 { font-size: 12.5pt; margin-top: 22px; color: #6a4526; } h4 { font-size: 11pt; margin: 18px 0 4px; }
  blockquote { margin: 8px 0; padding: 6px 12px; border-left: 3px solid #b08a5a; background: #f6f1e7; color: #4a4238; break-inside: avoid; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 10pt; }
  th, td { border: 1px solid #d8ccb6; padding: 5px 8px; text-align: left; vertical-align: top; } th { background: #efe7d8; }
  code { font-size: 9.5pt; } li { margin: 3px 0; }
  .bar { font: 10pt system-ui, sans-serif; background: #efe7d8; border: 1px solid #d8ccb6; border-radius: 10px; padding: 10px 14px; margin-bottom: 24px; }
  .bar button { font: inherit; margin-left: 8px; padding: 4px 12px; border-radius: 999px; border: 1px solid #6a4526; background: #6a4526; color: #f4ecdc; cursor: pointer; }
  @media print { .bar { display: none; } body { margin: 0; max-width: none; } }
</style></head>
<body>
<div class="bar">To save as PDF, choose <b>Save as PDF</b> as the printer. <button onclick="print()">Print / Save as PDF</button></div>
${html}
<script>addEventListener("load", () => setTimeout(() => print(), 300));</script>
</body></html>`;
}

// ------------------------------------------------------------------ result cache

const sha256 = (data) => crypto.createHash("sha256").update(data).digest("hex");

// The commit the notebook will clone: a new commit on the branch must not reuse
// a report made by older code. null if GitHub can't be reached (then the branch name counts).
function branchCommit() {
  return new Promise((resolve) => {
    const p = spawn("git", ["ls-remote", "https://github.com/Rhivu10/ClauseLens.git", `refs/heads/${REPO_BRANCH}`], {
      windowsHide: true,
    });
    let out = "";
    p.stdout.on("data", (d) => (out += d));
    p.on("error", () => resolve(null));
    p.on("close", (code) => resolve(code === 0 ? out.trim().split(/\s+/)[0] || null : null));
    setTimeout(() => {
      p.kill();
      resolve(null);
    }, 15_000);
  });
}

// Same PDFs (by content, not name), same code (branch commit + runner) -> same report
function cacheKey(files, commit) {
  return sha256(
    JSON.stringify({
      document_a: sha256(files.document_a.buffer),
      document_b: sha256(files.document_b.buffer),
      branch: REPO_BRANCH,
      commit,
      runner: sha256(fs.readFileSync(RUNNER)),
    })
  ).slice(0, 32);
}

// Overwrites the report files in `to` (and removes ones the new report doesn't have).
// The folder itself is kept: on Windows it can't be deleted while another program
// has it open (EPERM), which once crashed the server at the end of a run.
function copyReport(from, to) {
  fs.mkdirSync(to, { recursive: true });
  for (const f of [...REPORT_FILES, "run.json"]) {
    const src = path.join(from, f);
    const dst = path.join(to, f);
    if (fs.existsSync(src)) fs.copyFileSync(src, dst);
    else fs.rmSync(dst, { force: true });
  }
}

// The saved report under the names of this upload (the PDFs may have been renamed)
function useCached(dir, files) {
  copyReport(dir, LATEST);
  const names = { document_a: label(files.document_a.name), document_b: label(files.document_b.name) };
  const ui = readJSON(path.join(LATEST, "ui_data.json"));
  if (ui) {
    for (const side of Object.keys(names)) if (ui.documents?.[side]) ui.documents[side].label = names[side];
    writeJSON(path.join(LATEST, "ui_data.json"), ui);
  }
  const report = readJSON(path.join(LATEST, "report.json"));
  if (report) {
    for (const side of Object.keys(names)) if (report.documents?.[side]) report.documents[side].name = names[side];
    writeJSON(path.join(LATEST, "report.json"), report);
  }
}

const safeName = (name) => path.basename(name).replace(/[^\w.\- ]+/g, "_");
const label = (name) => path.basename(name).replace(/\.pdf$/i, "");

// Starts a comparison. files: {document_a: {name, buffer}, document_b: {name, buffer}}
export async function startJob(files, { force = false } = {}) {
  if (job && !["done", "error"].includes(job.state))
    throw new Error(job.type === "consult" ? "Kaggle is answering questions right now. Try again in a few minutes." : "A comparison is already running.");
  const id = `job_${new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14)}`;
  const commit = await branchCommit();
  const key = cacheKey(files, commit);
  const saved = path.join(CACHE, key);
  const meta = readJSON(path.join(saved, "run.json"));
  if (!force && meta && fs.existsSync(path.join(saved, "ui_data.json"))) {
    useCached(saved, files);
    writeJSON(path.join(LATEST, "run.json"), { ...meta, cached: true, opened_at: new Date().toISOString() });
    job = null;
    update({
      id,
      state: "done",
      stage: "done",
      error: null,
      cached: true,
      cache_key: key,
      started_at: new Date().toISOString(),
      finished_at: new Date().toISOString(),
      files: { document_a: files.document_a.name, document_b: files.document_b.name },
      repo_branch: REPO_BRANCH,
      headline: meta.headline,
      first_run_at: meta.finished_at,
    });
    return job;
  }

  const user = await getUsername();
  job = null;
  update({
    id,
    state: "running",
    stage: "uploading",
    error: null,
    started_at: new Date().toISOString(),
    files: { document_a: files.document_a.name, document_b: files.document_b.name },
    repo_branch: REPO_BRANCH,
    cached: false,
    cache_key: key,
    kernel_url: `https://www.kaggle.com/code/${user}/${KERNEL_SLUG}`,
  });
  run(user, id, (dsDir, jobFile) => {
    for (const [side, prefix] of [["document_a", "A"], ["document_b", "B"]]) {
      const file = `${prefix}_${safeName(files[side].name)}`;
      fs.writeFileSync(path.join(dsDir, file), files[side].buffer);
      jobFile[side] = { file, name: label(files[side].name) };
    }
  }).catch((e) => update({ state: "error", error: e.message }));
  return job;
}

// writeInputs(dsDir, jobFile): puts the job's input files in the dataset folder and
// fills in clauselens_job.json (the two PDFs for a comparison, report + questions for consult)
async function run(user, id, writeInputs) {
  // 1. Dataset version with the inputs + the job file
  const dsDir = path.join(WORK, "dataset");
  fs.rmSync(dsDir, { recursive: true, force: true });
  fs.mkdirSync(dsDir, { recursive: true });
  const jobFile = { job_id: id, repo_branch: REPO_BRANCH };
  writeInputs(dsDir, jobFile);
  writeJSON(path.join(dsDir, "clauselens_job.json"), jobFile);
  const datasetId = `${user}/${DATASET_SLUG}`;
  writeJSON(path.join(dsDir, "dataset-metadata.json"), {
    title: "ClauseLens Input",
    id: datasetId,
    licenses: [{ name: "CC0-1.0" }],
  });

  let exists = true;
  try {
    await kaggle(["datasets", "status", datasetId]);
  } catch {
    exists = false;
  }
  if (exists) await kaggle(["datasets", "version", "-p", dsDir, "-m", id, "-d", "-q"]);
  else await kaggle(["datasets", "create", "-p", dsDir, "-q"]); // private by default

  // 2. Wait until Kaggle has processed the upload (the notebook gets the latest ready version)
  update({ stage: "dataset_processing" });
  for (let i = 0; ; i++) {
    const s = (await kaggle(["datasets", "status", datasetId])).trim().toLowerCase();
    if (s.includes("ready")) break;
    if (s.includes("error") || s.includes("fail")) throw new Error(`Kaggle could not process the upload (${s}).`);
    if (i > 60) throw new Error("Kaggle took more than 10 minutes to process the upload.");
    await sleep(10_000);
  }

  // 3. Push the runner notebook: this starts a new run
  update({ stage: "starting" });
  const kDir = path.join(WORK, "kernel");
  fs.rmSync(kDir, { recursive: true, force: true });
  fs.mkdirSync(kDir, { recursive: true });
  fs.copyFileSync(path.join(HERE, "kaggle", "runner.py"), path.join(kDir, "runner.py"));
  writeJSON(path.join(kDir, "kernel-metadata.json"), {
    id: `${user}/${KERNEL_SLUG}`,
    title: "ClauseLens Runner",
    code_file: "runner.py",
    language: "python",
    kernel_type: "script",
    is_private: true,
    enable_gpu: true,
    enable_internet: true,
    machine_shape: "NvidiaTeslaT4", // GPU T4 x2
    dataset_sources: CHECKPOINT_DATASET ? [datasetId, CHECKPOINT_DATASET] : [datasetId],
    competition_sources: [],
    kernel_sources: [],
  });
  await kaggle(["kernels", "push", "-p", kDir]);
  update({ stage: "queued" });
  poll(user, id);
}

let currentUserForPoll = null;

function poll(user, id) {
  clearTimeout(timer);
  currentUserForPoll = user;
  const kernel = `${user}/${KERNEL_SLUG}`;
  const tick = async () => {
    if (!job || job.id !== id || job.state !== "running") return;
    followLog(kernel, id);
    try {
      const status = (await kaggle(["kernels", "status", kernel])).toLowerCase();
      if (/complete/.test(status)) return finishSafely(kernel, id);
      if (/error|cancel/.test(status)) {
        await finishSafely(kernel, id, true);
        return;
      }
    } catch (e) {
      update({ last_poll_error: e.message }); // network hiccup: keep polling
    }
    timer = setTimeout(tick, finishing ? 5_000 : POLL_MS);
  };
  timer = setTimeout(tick, 5_000);
}

// ------------------------------------------------------------------ live log
//
// Lines from runner.py:
//   [ClauseLens] <stage> {"job_id": ...}                    progress
//   [ClauseLensData <kind> <seq> <n>/<total>] <base64>      a JSON message in pieces
// kind "preview": {job_id, ui, done, total}; kind "result": {job_id, pair_id, alignment, overview, done, total}

let follower = null; // {id, proc}

function followLog(kernel, id) {
  if (follower && follower.id === id) return; // already following (restarted on exit below)
  stopFollowing();
  const pieces = new Map();
  const proc = spawn(PYTHON, ["-m", "kaggle.cli", "kernels", "logs", kernel, "-f"], {
    env: { ...process.env, PYTHONIOENCODING: "utf-8" },
    windowsHide: true,
  });
  follower = { id, proc };
  let buf = "";
  proc.stdout.on("data", (d) => {
    buf += d;
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      handleLogLine(buf.slice(0, i).trim(), id, pieces);
      buf = buf.slice(i + 1);
    }
  });
  proc.on("error", () => {});
  proc.on("close", () => {
    // The stream ends when the session ends, or right away while Kaggle still shows the
    // previous run: follow again while this run is going
    if (follower?.proc === proc) follower = null;
    if (job?.id === id && job.state === "running") setTimeout(() => followLog(kernel, id), 15_000);
  });
}

function stopFollowing() {
  if (follower) {
    const { proc } = follower;
    follower = null;
    proc.kill();
  }
}

function handleLogLine(line, id, pieces) {
  if (!job || job.id !== id || job.state !== "running") return;
  let m = line.match(/^\[ClauseLens\] (\w+) (\{.*\})$/);
  if (m) {
    try {
      if (JSON.parse(m[2]).job_id !== id) return;
      if (["node_6_report", "done", "error"].includes(m[1]) && !finishing) {
        finishing = true; // Kaggle marks the run complete a few seconds after this
        clearTimeout(timer);
        timer = setTimeout(() => poll(currentUserForPoll, id), 5_000);
      }
      if (m[1] !== "done" && m[1] !== "error") update({ stage: m[1] });
    } catch {
      /* not one of ours */
    }
    return;
  }
  m = line.match(/^\[ClauseLensData (\w+) (\d+) (\d+)\/(\d+)\] (.*)$/);
  if (!m) return;
  const [, kind, seq, n, total, part] = m;
  const got = pieces.get(seq) || [];
  got[Number(n) - 1] = part;
  pieces.set(seq, got);
  if (got.filter((x) => x !== undefined).length < Number(total)) return;
  pieces.delete(seq);
  let msg;
  try {
    msg = JSON.parse(Buffer.from(got.join(""), "base64").toString("utf8"));
  } catch {
    return;
  }
  if (msg.job_id !== id) return; // an older run's log
  if (kind === "answer") {
    if (job.type === "consult") saveAnswers(job.report_id, [msg], { done: msg.done, total: msg.total });
    return;
  }
  if (kind === "preview") {
    writeJSON(PARTIAL_FILE, { job_id: id, ui: msg.ui, done: msg.done, total: msg.total, updated_at: new Date().toISOString() });
    update({ preview_ready: true, progress: { done: msg.done, total: msg.total } });
  } else if (kind === "result") {
    const p = readJSON(PARTIAL_FILE);
    if (!p || p.job_id !== id) return;
    const i = p.ui.alignments.findIndex((a) => a.pair_id === msg.pair_id);
    if (i >= 0 && msg.alignment) p.ui.alignments[i] = msg.alignment;
    if (msg.overview) p.ui.overview = msg.overview;
    Object.assign(p, { done: msg.done, total: msg.total, updated_at: new Date().toISOString() });
    writeJSON(PARTIAL_FILE, p);
    update({ progress: { done: msg.done, total: msg.total } });
  }
}

// A problem while saving the result becomes a job error instead of crashing the server
async function finishSafely(kernel, id, failed = false) {
  try {
    await finish(kernel, id, failed);
  } catch (e) {
    update({ state: "error", error: `The run finished on Kaggle, but saving the result failed: ${e.message}` });
  }
}

async function finish(kernel, id, failed = false) {
  stopFollowing();
  finishing = false;
  if (job?.type === "consult") return finishConsult(kernel, id, failed);
  update({ stage: "downloading" });
  const outDir = path.join(WORK, "output");
  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(outDir, { recursive: true });
  try {
    await kaggle(["kernels", "output", kernel, "-p", outDir, "-o", "-q", "--file-pattern", "^((status|report|ui_data)\\.(json|md)|node5_results\\.jsonl)$"]);
  } catch (e) {
    if (!failed) throw e;
  }
  const status = readJSON(path.join(outDir, "status.json"));
  if (status?.job_id && status.job_id !== id)
    return update({ state: "error", error: "Kaggle ran an older upload. Please run the comparison again." });
  if (status?.stage === "error" || failed || !fs.existsSync(path.join(outDir, "ui_data.json")))
    return update({
      state: "error",
      error: status?.error || "The Kaggle notebook stopped with an error. Open it on Kaggle to see the log.",
    });

  const finished_at = new Date().toISOString();
  writeJSON(path.join(outDir, "run.json"), {
    job_id: id,
    files: job.files,
    repo_branch: job.repo_branch,
    started_at: job.started_at,
    finished_at,
    headline: status?.headline,
    cached: false,
  });
  copyReport(outDir, LATEST);
  if (job.cache_key) copyReport(outDir, path.join(CACHE, job.cache_key)); // for the next upload of the same PDFs
  update({ state: "done", stage: "done", finished_at, headline: status?.headline });
  fs.rmSync(PARTIAL_FILE, { force: true });
}

// ------------------------------------------------------------------ consult (AI answers)

const norm = (q) => q.trim().toLowerCase().replace(/\s+/g, " ").replace(/[?.!]+$/, "");

// The report questions are about: the latest one (its file content identifies it)
function reportId() {
  const f = path.join(LATEST, "report.json");
  return fs.existsSync(f) ? sha256(fs.readFileSync(f)).slice(0, 16) : null;
}

function consultFile(rid) {
  return path.join(CONSULT_DIR, `${rid}.json`);
}

function loadConsult(rid) {
  return { answers: [], queued: [], asking: [], ...(readJSON(consultFile(rid)) || {}) };
}

function saveAnswers(rid, answers, progress) {
  const store = loadConsult(rid);
  for (const a of answers) {
    if (!a?.question) continue;
    const entry = { question: a.question, answer: a.answer, citations: a.citations || [], supported: !!a.supported, kind: a.kind,
                    used: a.used || [], answered_at: new Date().toISOString() };
    const i = store.answers.findIndex((x) => norm(x.question) === norm(a.question));
    if (i >= 0) store.answers[i] = entry;
    else store.answers.push(entry);
    store.asking = store.asking.filter((q) => norm(q) !== norm(a.question));
  }
  writeJSON(consultFile(rid), store);
  if (progress) update({ progress });
}

// State for the Consult page: saved answers, questions waiting, the running batch
export function consultState() {
  const rid = reportId();
  if (!rid) return { report_id: null, answers: [], queued: [], asking: [], job: null };
  const store = loadConsult(rid);
  const running = job?.type === "consult" && job.report_id === rid && job.state === "running";
  return { report_id: rid, ...store, asking: running ? store.asking : [], job: job?.type === "consult" ? job : null };
}

// Ask questions about the latest report. Answered ones come back at once; the rest
// start a Kaggle batch, or wait for the next one if a batch is running.
export async function askQuestions(questions) {
  const rid = reportId();
  if (!rid) throw new Error("There is no report yet. Run a comparison first.");
  const busy = job && !["done", "error"].includes(job.state);
  if (busy && job.type !== "consult") throw new Error("A comparison is running. Ask again when it's done.");
  const store = loadConsult(rid);
  const known = new Set([...store.answers.map((a) => a.question), ...store.queued, ...store.asking].map(norm));
  for (const q of questions.map((q) => String(q).trim()).filter(Boolean)) {
    if (!known.has(norm(q))) {
      store.queued.push(q);
      known.add(norm(q));
    }
  }
  writeJSON(consultFile(rid), store);
  if (!busy && store.queued.length) await startConsultJob(rid);
  return consultState();
}

async function startConsultJob(rid) {
  const store = loadConsult(rid);
  const questions = store.queued.slice(0, 10);
  if (!questions.length) return;
  store.queued = store.queued.slice(questions.length);
  store.asking = questions;
  writeJSON(consultFile(rid), store);
  const user = await getUsername();
  const id = `ask_${new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14)}`;
  job = null;
  update({
    id,
    type: "consult",
    state: "running",
    stage: "uploading",
    error: null,
    started_at: new Date().toISOString(),
    report_id: rid,
    questions,
    progress: { done: 0, total: questions.length },
    repo_branch: REPO_BRANCH,
    kernel_url: `https://www.kaggle.com/code/${user}/${KERNEL_SLUG}`,
  });
  run(user, id, (dsDir, jobFile) => {
    fs.copyFileSync(path.join(LATEST, "report.json"), path.join(dsDir, "report.json"));
    Object.assign(jobFile, { type: "consult", report: { file: "report.json" }, questions });
  }).catch((e) => failConsult(rid, e.message));
}

function failConsult(rid, message) {
  // Questions of a failed batch go back to the queue, so they can be asked again
  const store = loadConsult(rid);
  store.queued = [...store.asking, ...store.queued];
  store.asking = [];
  writeJSON(consultFile(rid), store);
  update({ state: "error", error: message });
}

async function finishConsult(kernel, id, failed) {
  update({ stage: "downloading" });
  const outDir = path.join(WORK, "output");
  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(outDir, { recursive: true });
  try {
    await kaggle(["kernels", "output", kernel, "-p", outDir, "-o", "-q", "--file-pattern", "^(status|answers)\\.json$"]);
  } catch (e) {
    if (!failed) throw e;
  }
  const status = readJSON(path.join(outDir, "status.json"));
  const out = readJSON(path.join(outDir, "answers.json"));
  const rid = job.report_id;
  if (status?.job_id !== id || status?.stage === "error" || failed || !out?.answers)
    return failConsult(rid, status?.error || "Answering on Kaggle failed. Open the notebook log on Kaggle to see why.");
  saveAnswers(rid, out.answers);
  update({ state: "done", stage: "done", finished_at: new Date().toISOString() });
  if (loadConsult(rid).queued.length) startConsultJob(rid).catch((e) => failConsult(rid, e.message));
}

// After a dev-server restart, keep following a run that was still going
export function resume() {
  if (!job || job.state !== "running") return;
  const pushed = STAGES.findIndex((s) => s[0] === job.stage) >= STAGES.findIndex((s) => s[0] === "queued");
  if (pushed) {
    getUsername()
      .then((user) => poll(user, job.id))
      .catch((e) => update({ state: "error", error: e.message }));
  } else {
    update({ state: "error", error: "The upload was interrupted (the dev server stopped). Please run it again." });
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
