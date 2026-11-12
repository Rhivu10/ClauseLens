import { wordDiff } from "@/lib/wordDiff";

// Renders `self` as plain text, but marks the words that are unique to it
// relative to `other`:
//   mode="del" → words in `self` that were removed in `other` (red)
//   mode="ins" → words in `self` that were added versus `other` (teal)
export default function HighlightedText({ self, other, mode, className = "" }) {
  const [from, to] = mode === "del" ? [self, other] : [other, self];
  const ops = wordDiff(from, to);
  return (
    <p className={`font-serif-text text-[15px] leading-relaxed text-(--color-ink) ${className}`}>
      {ops.map((op, i) => {
        if (op.type === "equal") return <span key={i}>{op.text}</span>;
        if (op.type !== mode) return null;
        const cls =
          mode === "del"
            ? "bg-(--color-diverge)/12 text-(--color-diverge) rounded-[2px]"
            : "bg-(--color-cyan)/15 text-(--color-cyan-dim) rounded-[2px]";
        return (
          <span key={i} className={cls}>
            {op.text}
          </span>
        );
      })}
    </p>
  );
}
