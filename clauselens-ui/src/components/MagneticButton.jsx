import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/cn";

// A button that leans gently toward the cursor within its bounds — the kind
// of "expensive-feeling" micro-interaction Aceternity/reactbits components
// are known for, built by hand here since it's a single small effect.
export default function MagneticButton({ children, className = "", ...props }) {
  const ref = useRef(null);
  const [pos, setPos] = useState({ x: 0, y: 0 });

  function handleMove(e) {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    setPos({ x: x * 0.25, y: y * 0.35 });
  }

  return (
    <motion.button
      ref={ref}
      onMouseMove={handleMove}
      onMouseLeave={() => setPos({ x: 0, y: 0 })}
      animate={{ x: pos.x, y: pos.y }}
      transition={{ type: "spring", stiffness: 150, damping: 12, mass: 0.3 }}
      className={cn(
        "relative px-6 py-3 rounded-full bg-(--color-brass) text-(--color-brass-ink) font-medium text-sm",
        "shadow-[0_0_0_1px_rgba(106,69,38,0.35)] hover:shadow-[0_0_24px_2px_rgba(106,69,38,0.28)] transition-shadow",
        className
      )}
      {...props}
    >
      {children}
    </motion.button>
  );
}
