// Lightweight word-level diff (LCS) for showing clause redlines.
// Returns a flat list of ops: { type: "equal" | "del" | "ins", text }

function tokenize(str) {
  // keep words and the whitespace that follows them together-ish:
  // split into word / non-word runs so punctuation diffs cleanly
  return str.match(/\s+|[^\s]+/g) ?? [];
}

export function wordDiff(a = "", b = "") {
  const A = tokenize(a);
  const B = tokenize(b);
  const n = A.length;
  const m = B.length;

  // LCS table
  const dp = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  const ops = [];
  let i = 0;
  let j = 0;
  const push = (type, text) => {
    const last = ops[ops.length - 1];
    if (last && last.type === type) last.text += text;
    else ops.push({ type, text });
  };

  while (i < n && j < m) {
    if (A[i] === B[j]) {
      push("equal", A[i]);
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      push("del", A[i]);
      i++;
    } else {
      push("ins", B[j]);
      j++;
    }
  }
  while (i < n) push("del", A[i++]);
  while (j < m) push("ins", B[j++]);

  return ops;
}

// Convenience: how much of the text (by char length) changed.
export function diffMagnitude(ops) {
  let changed = 0;
  let total = 0;
  for (const op of ops) {
    total += op.text.length;
    if (op.type !== "equal") changed += op.text.length;
  }
  return total ? changed / total : 0;
}
