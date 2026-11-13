// Magnetic dock: icons grow as the cursor nears them (spring physics), with
// tooltips, badges and an active dot. Adapted from Componentry's magnetic-dock
// (componentry.dev/docs/components/magnetic-dock) to plain JSX and the ClauseLens
// colours. Made for a top bar: icons grow downward and tooltips open below.
import { useCallback, useRef, useState } from "react";
import { AnimatePresence, motion, useMotionValue, useReducedMotion, useSpring, useTransform } from "framer-motion";

const SPRING = { damping: 20, stiffness: 300, mass: 0.5 };

function DockItem({ item, mouseX, iconSize, maxScale, magneticDistance, reducedMotion }) {
  const ref = useRef(null);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);

  // distance from the cursor to this icon's centre -> scale
  const distance = useTransform(mouseX, (x) => {
    if (!ref.current) return magneticDistance + 1;
    const rect = ref.current.getBoundingClientRect();
    return x - (rect.left + rect.width / 2);
  });
  const scale = useTransform(distance, [-magneticDistance, 0, magneticDistance], [1, maxScale, 1]);
  const smoothScale = useSpring(scale, SPRING);
  const size = useTransform(smoothScale, (s) => s * iconSize);
  const y = useSpring(useTransform(smoothScale, (s) => (s - 1) * 6), SPRING);

  return (
    <motion.button
      ref={ref}
      type="button"
      aria-label={item.label}
      aria-current={item.isActive ? "page" : undefined}
      onClick={item.onClick}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="relative flex items-center justify-center rounded-xl focus:outline-none focus-visible:ring-2 focus-visible:ring-(--color-brass)"
      style={{
        width: reducedMotion ? iconSize : size,
        height: reducedMotion ? iconSize : size,
        y: reducedMotion ? 0 : y,
      }}
      whileTap={reducedMotion ? undefined : { scale: 0.9 }}
    >
      <div
        className={`relative w-full h-full rounded-xl overflow-hidden flex items-center justify-center border transition-colors duration-200 ${
          item.isActive
            ? "bg-(--color-brass) border-(--color-brass) text-(--color-brass-ink)"
            : "bg-(--color-surface) border-(--color-line) text-(--color-ink-dim)"
        }`}
        style={{
          boxShadow: hovered
            ? "0 8px 24px rgba(41,36,29,0.18), inset 0 1px 0 rgba(255,255,255,0.5)"
            : "0 3px 10px rgba(41,36,29,0.08), inset 0 1px 0 rgba(255,255,255,0.35)",
        }}
      >
        <div aria-hidden="true" className="w-[55%] h-[55%] flex items-center justify-center">
          {item.icon}
        </div>
        {/* shine */}
        <div
          className="absolute inset-0 pointer-events-none transition-opacity duration-200"
          style={{
            background: "linear-gradient(135deg, rgba(255,255,255,0.55) 0%, transparent 50%)",
            opacity: hovered ? 0.8 : 0.4,
          }}
        />
      </div>

      {/* badge */}
      <AnimatePresence initial={false}>
        {item.badge > 0 && (
          <motion.div
            initial={reducedMotion ? false : { scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            className="absolute -top-1.5 -right-1.5 min-w-[18px] h-[18px] px-1 rounded-full bg-(--color-diverge) text-white text-[10px] font-semibold flex items-center justify-center border-2 border-(--color-bg)"
          >
            {item.badge > 99 ? "99+" : item.badge}
          </motion.div>
        )}
      </AnimatePresence>

      {/* active dot */}
      <AnimatePresence initial={false}>
        {item.isActive && (
          <motion.div
            initial={reducedMotion ? false : { scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            className="absolute -bottom-2 w-1 h-1 rounded-full bg-(--color-accent)"
          />
        )}
      </AnimatePresence>

      {/* tooltip, below the icon */}
      <AnimatePresence initial={false}>
        {(hovered || focused) && (
          <motion.div
            aria-hidden="true"
            initial={reducedMotion ? false : { opacity: 0, y: -6, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.9 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            className="absolute top-full mt-3.5 left-1/2 -translate-x-1/2 px-2.5 py-1 rounded-lg bg-(--color-surface) border border-(--color-line) text-(--color-ink) text-[12.5px] whitespace-nowrap shadow-lg pointer-events-none z-50"
          >
            {item.num && <span className="font-mono text-[10px] text-(--color-accent) mr-1.5">{item.num}</span>}
            {item.label}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.button>
  );
}

export default function MagneticDock({ items, iconSize = 34, maxScale = 1.4, magneticDistance = 110, className = "" }) {
  const mouseX = useMotionValue(Infinity);
  const reducedMotion = useReducedMotion() ?? false;
  const onMove = useCallback((e) => mouseX.set(e.clientX), [mouseX]);

  return (
    <div
      onMouseMove={reducedMotion ? undefined : onMove}
      onMouseLeave={() => mouseX.set(Infinity)}
      className={`inline-flex items-start gap-2 p-1.5 rounded-2xl bg-(--color-surface-2)/70 border border-(--color-line) backdrop-blur ${className}`}
    >
      {items.map((item) => (
        <DockItem
          key={item.id}
          item={item}
          mouseX={mouseX}
          iconSize={iconSize}
          maxScale={maxScale}
          magneticDistance={magneticDistance}
          reducedMotion={reducedMotion}
        />
      ))}
    </div>
  );
}
