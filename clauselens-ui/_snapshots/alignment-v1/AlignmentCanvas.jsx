import { useEffect, useState } from "react";

const COLOR = {
  match: "var(--color-cyan)",
  diverge: "var(--color-diverge)",
  missing_in_a: "var(--color-brass)",
  missing_in_b: "var(--color-brass)",
};

export default function AlignmentCanvas({ containerRef, nodeRefs, alignments, activeId }) {
  const [paths, setPaths] = useState([]);

  useEffect(() => {
    function measure() {
      const container = containerRef.current;
      if (!container) return;
      const cRect = container.getBoundingClientRect();

      const next = alignments
        .map((al) => {
          const aEl = al.a ? nodeRefs.current[al.a] : null;
          const bEl = al.b ? nodeRefs.current[al.b] : null;
          if (!aEl || !bEl) return null;
          const aRect = aEl.getBoundingClientRect();
          const bRect = bEl.getBoundingClientRect();
          return {
            id: al.id,
            status: al.status,
            x1: aRect.right - cRect.left,
            y1: aRect.top - cRect.top + aRect.height / 2,
            x2: bRect.left - cRect.left,
            y2: bRect.top - cRect.top + bRect.height / 2,
          };
        })
        .filter(Boolean);
      setPaths(next);
    }

    measure();
    window.addEventListener("resize", measure);
    const container = containerRef.current;
    container?.addEventListener("scroll", measure);
    const interval = setInterval(measure, 400);
    return () => {
      window.removeEventListener("resize", measure);
      container?.removeEventListener("scroll", measure);
      clearInterval(interval);
    };
  }, [containerRef, nodeRefs, alignments]);

  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 1 }}>
      {paths.map((p) => {
        const isActive = activeId === p.id;
        const mid = (p.x1 + p.x2) / 2;
        const d = `M ${p.x1} ${p.y1} C ${mid} ${p.y1}, ${mid} ${p.y2}, ${p.x2} ${p.y2}`;
        const color = COLOR[p.status] ?? "var(--color-ink-faint)";
        return (
          <g key={p.id} opacity={activeId && !isActive ? 0.15 : 1}>
            <path
              d={d}
              fill="none"
              stroke={color}
              strokeWidth={isActive ? 2.5 : 1.3}
              strokeDasharray={p.status.startsWith("missing") ? "4 3" : "0"}
            />
            <circle cx={p.x1} cy={p.y1} r={isActive ? 4.5 : 3} fill={color} />
            <circle cx={p.x2} cy={p.y2} r={isActive ? 4.5 : 3} fill={color} />
          </g>
        );
      })}
    </svg>
  );
}
