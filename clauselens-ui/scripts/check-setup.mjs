// ClauseLens — setup check (npm run setup)
//
// Checks everything a comparison needs on this computer and on Kaggle, and says
// exactly what to do for anything missing. Changes nothing.

import { spawnSync } from "node:child_process";

const PYTHON = process.env.CLAUSELENS_PYTHON || "python";
const CHECKPOINT = process.env.CLAUSELENS_CHECKPOINT_DATASET || null; // the adapter is in the repo by default

let problems = 0;
const ok = (msg) => console.log(`  ✓ ${msg}`);
const bad = (msg, ...fix) => {
  problems++;
  console.log(`  ✗ ${msg}`);
  for (const line of fix) console.log(`      ${line}`);
};
const run = (cmd, args) => {
  const r = spawnSync(cmd, args, { encoding: "utf8", env: { ...process.env, PYTHONIOENCODING: "utf-8" }, shell: false });
  return { code: r.error ? -1 : r.status, out: `${r.stdout || ""}${r.stderr || ""}`.trim() };
};

console.log("\nClauseLens setup check\n");

// 1. Node.js
const major = Number(process.versions.node.split(".")[0]);
if (major >= 20) ok(`Node.js ${process.versions.node}`);
else bad(`Node.js ${process.versions.node} is too old`, "Install Node.js 20 or newer from https://nodejs.org");

// 2. Python
const py = run(PYTHON, ["--version"]);
if (py.code === 0) ok(py.out);
else bad(`Python not found ("${PYTHON}")`, "Install Python 3.10+ from https://www.python.org", "or set CLAUSELENS_PYTHON to your python command");

// 3. Kaggle tool
const kg = py.code === 0 ? run(PYTHON, ["-m", "kaggle.cli", "--version"]) : { code: -1, out: "" };
const kgMissing = /No module named/i.test(kg.out);
if (py.code === 0 && !kgMissing) ok("Kaggle tool installed");
else if (py.code === 0) bad("Kaggle tool not installed", `Run: ${PYTHON} -m pip install kaggle`);

// 4. Kaggle token + username
let user = null;
if (py.code === 0 && !kgMissing) {
  const who = run(PYTHON, ["-c", "from kaggle.api.kaggle_api_extended import KaggleApi\na=KaggleApi(); a.authenticate(); print(a.get_config_value('username'))"]);
  user = who.code === 0 ? who.out.split(/\s+/).pop() : null;
  if (user) ok(`Signed in to Kaggle as ${user}`);
  else
    bad(
      "No Kaggle API token",
      "kaggle.com → Settings → API Tokens → Generate, then save the token as",
      "~/.kaggle/access_token (Windows: C:\\Users\\<you>\\.kaggle\\access_token)"
    );
}

// 5. The CUAD LoRA adapter (Node 3): in the repo, unless another copy is set
if (!CHECKPOINT) ok("CUAD adapter: loaded from the repo (Model Finetuning/checkpoints/checkpoint-2091)");
else if (user) {
  const ds = run(PYTHON, ["-m", "kaggle.cli", "datasets", "files", CHECKPOINT]);
  if (ds.code === 0 && /adapter_config\.json/.test(ds.out)) ok(`CUAD adapter found: ${CHECKPOINT}`);
  else
    bad(
      `No access to the CUAD adapter dataset ${CHECKPOINT}`,
      "Ask its owner to share it with your Kaggle account (or make it public),",
      "or upload your own copy of checkpoint-2091 and set CLAUSELENS_CHECKPOINT_DATASET=<you>/<dataset>"
    );
}

// 6. Phone verification (GPU + internet on Kaggle) can't be checked from here
console.log("  • Kaggle needs a phone-verified account for GPU + internet (kaggle.com → Settings → Phone)");

console.log(problems ? `\n${problems} thing(s) to fix, then run npm run setup again.\n` : "\nAll set. Start with: npm run dev\n");
process.exit(problems ? 1 : 0);
