# ClauseLens — UI (v2)

Frontend for ClauseLens. React + Vite + Tailwind v4 + Framer Motion. You
upload two PDFs, the ClauseLens pipeline (Nodes 1–6) runs on a free Kaggle GPU
through the official Kaggle API, and the report is shown here. There is no
mock data: the Alignment and Consult pages show the latest real report.

## Run it

```bash
npm install
npm run dev      # http://localhost:5173 (also serves the /api the pages use)
```

## Running comparisons on Kaggle (free)

One-time setup on this laptop:

1. Install the Kaggle tool: `pip install kaggle`
2. On kaggle.com → Settings → API → **Generate New Token**, and save the token
   text as `C:\Users\<you>\.kaggle\access_token` (an older `kaggle.json` works too).
3. The Kaggle account must be phone-verified (needed for GPU + internet).
4. The account must be able to read the CUAD adapter dataset
   `tdhanush24/checkpoints` (checkpoint-2091). To use another copy, set
   `CLAUSELENS_CHECKPOINT_DATASET=<owner>/<dataset>` before `npm run dev`.

`npm run setup` checks all of this (Node, Python, the Kaggle tool, the token, access
to the adapter dataset) and says what to fix.

Then `npm run dev`, open **Intake**, choose the two PDFs and press **Run comparison**.

What happens (`server/kaggle.js`):

1. The PDFs are uploaded as a new version of your private dataset `clauselens-input`.
2. `server/kaggle/runner.py` is pushed as your private notebook `clauselens-runner`
   (GPU T4 x2, internet on), which starts a run. It clones ClauseLens `main`
   and runs Nodes 1–6, the same steps as the real-contract test notebook.
3. The page shows the stage from the notebook log. A full run takes about 7–8 minutes for a
   1-page document and ~16 minutes for a 20-page contract; the changed clauses appear after
   ~4–7 minutes and the impact summaries fill in as they are written.
   You can close the tab — the run continues and the report appears on Alignment.
4. The report is downloaded to `server/data/latest/` (`report.json`, `report.md`,
   `ui_data.json`); Alignment and Consult read it from there.

**Saved results.** Every finished report is also kept in `server/data/cache/`, keyed by
a fingerprint (SHA-256) of the two PDFs' contents, the ClauseLens branch and its current commit
(looked up on GitHub when you upload), and
`server/kaggle/runner.py`. Uploading the same two PDFs again (even renamed) opens the
saved report at once without using Kaggle; the Alignment page then says "Saved result".
Tick **Run again on Kaggle** on the upload page to force a fresh run. Swapping the two
PDFs, another branch, a new commit or a changed runner counts as a new comparison. Delete
`server/data/cache/` to clear all saved results.

Each run uses Kaggle's free weekly GPU quota (about 30 hours). If a run fails,
the page shows the error and a link to the notebook log on Kaggle.

| Setting (environment variable) | Default |
|---|---|
| `CLAUSELENS_PYTHON` | `python` (the Python that has `kaggle` installed) |
| `CLAUSELENS_CHECKPOINT_DATASET` | `tdhanush24/checkpoints` |
| `CLAUSELENS_DATASET` / `CLAUSELENS_KERNEL` | `clauselens-input` / `clauselens-runner` |

## What's already built

- **Landing (`/`)** — hero with a hand-built "aperture" animation (rotating
  lens rings + a scanning sweep) and a spotlight-hover feature grid.
- **Intake (`/upload`)** — drag-and-drop two-PDF upload; runs the comparison
  on Kaggle and shows its progress.
- **Alignment (`/compare`)** — every change side by side, with the changed
  words marked, the impact summary, affected parties, confidence and a filter.
- **Consult (`/consult`)** — ask about the changes; answers are the matching
  changes from the report, each citing its clause and page.

Design tokens live in `src/index.css` (`@theme` block): a near-black base,
a warm brass + cool cyan accent duo, Newsreader for document text, Space
Grotesk for UI, IBM Plex Mono for data/labels.

## Adding real components from the sites in your reference

Everything above was hand-built (I couldn't reach ui.aceternity.com or
21st.dev from my sandbox to run their CLIs), but your project is already
set up to drop their real components straight in. Here's exactly how, in
your own VS Code terminal:

### 1. shadcn/ui (needed as the base for Aceternity + 21st.dev installs)

```bash
npx shadcn@latest init
```
When prompted: pick **Vite**, base color your choice, and let it write
`components.json`. It already sees your Tailwind v4 setup.

### 2. Pull a component from Aceternity UI

Go to https://ui.aceternity.com/components, open any component you like,
and copy the install command it shows — it looks like:

```bash
npx shadcn@latest add https://ui.aceternity.com/registry/<component-name>.json
```

That drops the component's `.jsx`/`.tsx` file straight into
`src/components/ui/`. Import and use it like any other component.

### 3. Pull a component from 21st.dev

Same pattern — browse https://21st.dev, open a component, copy its
`npx shadcn@latest add <url>` command, run it in your terminal.

### 4. Pull an animation from reactbits.dev

Go to https://reactbits.dev, pick a component (e.g. a background effect
for a second page, or a text animation), choose the **JS + Tailwind**
variant, and use the CLI command it generates — reactbits also installs
through a CLI that copies the component file into your project, same idea
as shadcn. If a component needs GSAP or Three.js, it'll tell you to
`npm install gsap` or `npm install three` first.

### 5. componentry.dev

Same shadcn-registry pattern: browse the component, copy its `npx shadcn`
install command, run it.

## A word of caution (from the same design brief you sent me)

The carousel's own advice was: **pick one signature motion moment, not
five.** This project already has its signature moment (the aperture hero).
If you pull in more components, use them for genuinely new needs (e.g. a
pricing table, a testimonial marquee) rather than stacking multiple hero
animations on top of each other — that's what starts to look AI-generated
rather than designed.

## Wiring up the real backend

Done — see "Running comparisons on Kaggle" above. The pages read the Node 6
UI data (`nodes/node_6/ui.py` in the ClauseLens repo) from `/api/report`
(`server/api.js`); `src/lib/api.js` is the client.
