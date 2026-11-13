// Is an uploaded PDF a legal document? A quick check on the laptop, before anything
// goes to Kaggle, so a wrong upload (a CV, a recipe) is caught in a second instead
// of after a full run.
//
// Deliberately lenient (agreed with the team): a PDF counts as legal if it contains
// ANY of three words that nearly every legal text uses. Only a PDF with none of them
// is flagged, and the user can still run it. A PDF without readable text (a scan)
// is never flagged: there is nothing to check.

import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";

// word stems: shall; agree, agreed, agreement; law, laws, lawful, lawfully
const MARKERS = [
  ["shall", /\bshall\b/i],
  ["agree", /\bagree(?:s|d|ment|ments)?\b/i],
  ["law", /\blaw(?:s|ful|fully)?\b/i],
];
const MIN_TEXT = 200; // characters; less than this = no readable text (e.g. a scan)

async function pdfText(buffer) {
  const task = getDocument({ data: new Uint8Array(buffer), isEvalSupported: false, verbosity: 0 });
  const doc = await task.promise;
  let text = "";
  for (let i = 1; i <= doc.numPages; i++) {
    const page = await doc.getPage(i);
    const content = await page.getTextContent();
    text += content.items.map((it) => it.str).join(" ") + "\n";
  }
  await task.destroy();
  return text;
}

// -> {readable, legal, found: ["shall", ...]}
export async function legalCheck(buffer) {
  let text = "";
  try {
    text = await pdfText(buffer);
  } catch (e) {
    if (process.env.CLAUSELENS_DEBUG) console.error("legalCheck:", e);
    return { readable: false, legal: true, found: [] }; // can't read it: don't flag
  }
  if (text.replace(/\s+/g, "").length < MIN_TEXT) return { readable: false, legal: true, found: [] };
  const found = MARKERS.filter(([, re]) => re.test(text)).map(([name]) => name);
  return { readable: true, legal: found.length > 0, found };
}
