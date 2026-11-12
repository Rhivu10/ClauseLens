// Button with a glow that travels around its border, and lights the whole border
// on hover. Adapted from Aceternity UI's hover-border-gradient to plain JSX and the
// ClauseLens colours.
//
//   variant "solid"  brass button, light text (main actions)
//   variant "soft"   paper button, dark text (secondary actions)
//   as               the element to render: "button" (default), "a", Link, ...
import { useEffect, useState } from "react";
import { motion } from "framer-motion";

const DIRECTIONS = ["TOP", "LEFT", "BOTTOM", "RIGHT"];

const VARIANTS = {
  solid: {
    inner: "bg-(--color-brass) text-(--color-brass-ink)",
    fill: "var(--color-brass)",
    glow: "232, 197, 144",          // light gold travelling over the brass edge
    highlight: "#e8c590",
  },
  soft: {
    inner: "bg-(--color-surface) text-(--color-ink)",
    fill: "var(--color-surface)",
    glow: "106, 69, 38",            // brass travelling over the paper edge
    highlight: "#6a4526",
  },
};

const moving = (rgb) => ({
  TOP: `radial-gradient(20.7% 50% at 50% 0%, rgb(${rgb}) 0%, rgba(${rgb}, 0) 100%)`,
  LEFT: `radial-gradient(16.6% 43.1% at 0% 50%, rgb(${rgb}) 0%, rgba(${rgb}, 0) 100%)`,
  BOTTOM: `radial-gradient(20.7% 50% at 50% 100%, rgb(${rgb}) 0%, rgba(${rgb}, 0) 100%)`,
  RIGHT: `radial-gradient(16.2% 41.2% at 100% 50%, rgb(${rgb}) 0%, rgba(${rgb}, 0) 100%)`,
});

export function HoverBorderGradient({
  children,
  as: Tag = "button",
  variant = "solid",
  containerClassName = "",
  className = "",
  duration = 1,
  clockwise = true,
  disabled,
  ...props
}) {
  const v = VARIANTS[variant] || VARIANTS.solid;
  const [hovered, setHovered] = useState(false);
  const [direction, setDirection] = useState("TOP");
  const map = moving(v.glow);
  const highlight = `radial-gradient(75% 181.16% at 50% 50%, ${v.highlight} 0%, rgba(255, 255, 255, 0) 100%)`;

  useEffect(() => {
    if (hovered || disabled) return;
    const t = setInterval(() => {
      setDirection((d) => {
        const i = DIRECTIONS.indexOf(d);
        return DIRECTIONS[clockwise ? (i - 1 + 4) % 4 : (i + 1) % 4];
      });
    }, duration * 1000);
    return () => clearInterval(t);
  }, [hovered, disabled, duration, clockwise]);

  return (
    <Tag
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      disabled={disabled}
      className={`relative inline-flex rounded-full border border-(--color-line) content-center items-center justify-center overflow-visible p-px w-fit h-min
        bg-(--color-brass)/15 hover:bg-(--color-brass)/10 transition duration-500 disabled:opacity-35 disabled:cursor-not-allowed ${containerClassName}`}
      {...props}
    >
      <div className={`relative z-10 w-auto px-5 py-2.5 rounded-[inherit] text-sm font-medium inline-flex items-center justify-center gap-2 ${v.inner} ${className}`}>
        {children}
      </div>
      <motion.div
        className="flex-none inset-0 overflow-hidden absolute z-0 rounded-[inherit]"
        style={{ filter: "blur(2px)", width: "100%", height: "100%" }}
        initial={{ background: map[direction] }}
        animate={{ background: hovered && !disabled ? [map[direction], highlight] : map[direction] }}
        transition={{ ease: "linear", duration }}
      />
      <div className="absolute z-[1] flex-none inset-[2px] rounded-[inherit]" style={{ background: v.fill }} />
    </Tag>
  );
}
