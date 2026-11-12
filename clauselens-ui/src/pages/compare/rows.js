// Labels and colours for the alignment status of a change (from Node 6, see
// nodes/node_6/ui.py in the ClauseLens repo). The rows themselves come from the
// latest report (useReport in src/lib/api.js).

export const STATUS_LABEL = {
  match: "Aligned",
  diverge: "Changed",
  missing_in_a: "Added",
  missing_in_b: "Removed",
};

// token -> css var for the status accent
export const STATUS_COLOR = {
  match: "var(--color-cyan)",
  diverge: "var(--color-diverge)",
  missing_in_a: "var(--color-brass)",
  missing_in_b: "var(--color-brass)",
};
