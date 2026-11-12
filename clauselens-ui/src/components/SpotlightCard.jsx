import { useRef, useState } from "react";
import { cn } from "@/lib/cn";

// Mouse-follow spotlight on a card border, the effect Aceternity's
// "Card Spotlight" / "Glowing Border" components are known for.
export default function SpotlightCard({ children, className = "", spotlightColor = "var(--color-cyan)" }) {
  const ref = useRef(null);
  const [pos, setPos] = useState({ x: 50, y: 50 });
  const [opacity, setOpacity] = useState(0);

  function handleMove(e) {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    setPos({
      x: ((e.clientX - rect.left) / rect.width) * 100,
      y: ((e.clientY - rect.top) / rect.height) * 100,
    });
  }

  return (
    <div
      ref={ref}
      onMouseMove={handleMove}
      onMouseEnter={() => setOpacity(1)}
      onMouseLeave={() => setOpacity(0)}
      className={cn(
        "relative rounded-2xl border border-(--color-line) bg-(--color-surface) p-6 overflow-hidden",
        className
      )}
    >
      <div
        className="pointer-events-none absolute inset-0 transition-opacity duration-300"
        style={{
          opacity,
          background: `radial-gradient(280px circle at ${pos.x}% ${pos.y}%, color-mix(in srgb, ${spotlightColor} 15%, transparent), transparent 70%)`,
        }}
      />
      <div className="relative z-10">{children}</div>
    </div>
  );
}
