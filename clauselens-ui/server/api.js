// ClauseLens API for the UI, served by the Vite dev server (npm run dev), so no
// second process is needed.
//
//   GET  /api/setup    is the Kaggle tool installed and signed in?
//   POST /api/job      start a comparison: {document_a: {name, data}, document_b: {name, data}, force?, skipLegalCheck?}
//                      422 {code: "not_legal", files} when a PDF doesn't look like a legal document
//                      (data = the PDF as base64). The same two PDFs again return the saved
//                      report at once (state "done", cached: true) unless force is true.
//   GET  /api/job      the current comparison: state, stage, error + the list of stages
//   GET  /api/consult  saved answers for the latest report + the running batch
//   POST /api/consult  {questions: [...]}: answered at once if asked before, else on Kaggle (Gemma)
//   GET  /api/partial  the report as it fills in during a run ({ui, done, total}), 404 if none
//   GET  /api/report/download?format=md|json|pdf   the latest report as a file (pdf: a page to print / save as PDF)
//   GET  /api/report   the latest finished report (Node 6 UI data + run: when/how it was made), 404 if none yet

import { legalCheck } from "./legal.js";
import { STAGES, askQuestions, checkSetup, consultState, currentJob, latestReport, partialReport, reportDownload, resume, startJob } from "./kaggle.js";

const MAX_BYTES = 60 * 1024 * 1024;

function send(res, code, body) {
  res.statusCode = code;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(body));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const parts = [];
    let size = 0;
    req.on("data", (d) => {
      size += d.length;
      if (size > MAX_BYTES) {
        reject(new Error("The PDFs are too large (over 60 MB together)."));
        req.destroy();
      } else parts.push(d);
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(Buffer.concat(parts).toString("utf8")));
      } catch {
        reject(new Error("Invalid request body."));
      }
    });
    req.on("error", reject);
  });
}

function pdf(file, side) {
  if (!file?.name || !file?.data) throw new Error(`${side} is missing.`);
  const buffer = Buffer.from(file.data, "base64");
  if (buffer.subarray(0, 5).toString() !== "%PDF-") throw new Error(`${file.name} is not a PDF.`);
  return { name: file.name, buffer };
}

async function handle(req, res) {
  const url = req.url.split("?")[0];
  try {
    if (url === "/api/setup" && req.method === "GET") return send(res, 200, await checkSetup());
    if (url === "/api/job" && req.method === "GET") return send(res, 200, { job: currentJob(), stages: STAGES });
    if (url === "/api/job" && req.method === "POST") {
      const body = await readBody(req);
      const files = {
        document_a: pdf(body.document_a, "Document A"),
        document_b: pdf(body.document_b, "Document B"),
      };
      // Legal-document check (server/legal.js): flag a PDF with none of the marker
      // words before anything goes to Kaggle; the user can still run it anyway
      if (!body.skipLegalCheck) {
        const flagged = [];
        for (const f of Object.values(files)) if (!(await legalCheck(f.buffer)).legal) flagged.push(f.name);
        if (flagged.length)
          return send(res, 422, {
            code: "not_legal",
            files: flagged,
            error: `${flagged.join(" and ")} ${flagged.length > 1 ? "don't" : "doesn't"} look like a legal document.`,
          });
      }
      const job = await startJob(files, { force: Boolean(body.force) });
      return send(res, 202, { job, stages: STAGES });
    }
    if (url === "/api/consult" && req.method === "GET") return send(res, 200, { ...consultState(), stages: STAGES });
    if (url === "/api/consult" && req.method === "POST") {
      const body = await readBody(req);
      if (!Array.isArray(body.questions) || !body.questions.length) throw new Error("No question.");
      return send(res, 202, { ...(await askQuestions(body.questions.slice(0, 10))), stages: STAGES });
    }
    if (url === "/api/partial" && req.method === "GET") {
      const partial = partialReport();
      return partial ? send(res, 200, partial) : send(res, 404, { error: "No live results yet." });
    }
    if (url === "/api/report/download" && req.method === "GET") {
      const format = new URL(req.url, "http://localhost").searchParams.get("format") || "md";
      const file = reportDownload(format);
      if (!file) return send(res, 404, { error: "No report yet. Run a comparison first." });
      res.statusCode = 200;
      res.setHeader("Content-Type", file.type);
      res.setHeader("Content-Disposition", `${file.inline ? "inline" : "attachment"}; filename*=UTF-8''${encodeURIComponent(file.filename)}`);
      return res.end(file.body);
    }
    if (url === "/api/report" && req.method === "GET") {
      const report = latestReport();
      return report ? send(res, 200, report) : send(res, 404, { error: "No comparison has finished yet." });
    }
    return send(res, 404, { error: "Not found" });
  } catch (e) {
    return send(res, 400, { error: e.message });
  }
}

export default function clauseLensApi() {
  return {
    name: "clauselens-api",
    configureServer(server) {
      resume();
      server.middlewares.use((req, res, next) => (req.url.startsWith("/api/") ? handle(req, res) : next()));
    },
  };
}
