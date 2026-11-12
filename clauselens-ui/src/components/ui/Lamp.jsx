// Lamp: a thin bar of light that widens as it comes into view and casts two soft
// beams and a glow onto what is below it. Adapted from Aceternity UI's lamp for a
// light page: the beams fade out with masks (the original hides them behind dark
// panels), and the light is the page accent.
import { motion } from "framer-motion";

const GROW = { delay: 0.2, duration: 0.8, ease: "easeInOut" };

export function Lamp({ children, color = "136, 46, 58", width = 26, className = "" }) {
  const rgb = (a) => `rgba(${color}, ${a})`;
  // beams fade out away from the bar (from its centre, at the top inner corner)
  const fadeFrom = (x) => `radial-gradient(ellipse 100% 100% at ${x} 0%, black 25%, transparent 75%)`;

  return (
    <div className={`relative isolate flex flex-col items-center ${className}`}>
      <div className="pointer-events-none absolute inset-x-0 top-0 h-56 flex justify-center overflow-hidden" aria-hidden="true">
        {/* left and right beams, from the bar down and outwards */}
        <motion.div
          initial={{ opacity: 0.4, width: `${width / 2}rem` }}
          whileInView={{ opacity: 1, width: `${width}rem` }}
          viewport={{ once: true }}
          transition={GROW}
          className="absolute top-0 right-1/2 h-48"
          style={{
            backgroundImage: `conic-gradient(from 180deg at 100% 0%, transparent 0deg, ${rgb(0.3)} 75deg, transparent 90deg)`,
            maskImage: fadeFrom("100%"),
            WebkitMaskImage: fadeFrom("100%"),
          }}
        />
        <motion.div
          initial={{ opacity: 0.4, width: `${width / 2}rem` }}
          whileInView={{ opacity: 1, width: `${width}rem` }}
          viewport={{ once: true }}
          transition={GROW}
          className="absolute top-0 left-1/2 h-48"
          style={{
            backgroundImage: `conic-gradient(from 90deg at 0% 0%, transparent 0deg, ${rgb(0.3)} 15deg, transparent 90deg)`,
            maskImage: fadeFrom("0%"),
            WebkitMaskImage: fadeFrom("0%"),
          }}
        />
        {/* glow just under the bar */}
        <motion.div
          initial={{ width: "8rem", opacity: 0.5 }}
          whileInView={{ width: "18rem", opacity: 1 }}
          viewport={{ once: true }}
          transition={GROW}
          className="absolute top-0 h-24 -translate-y-1/2 rounded-full blur-2xl"
          style={{ background: rgb(0.35) }}
        />
      </div>

      {/* the bar itself */}
      <motion.div
        initial={{ width: "2.5rem" }}
        whileInView={{ width: `${width}rem` }}
        viewport={{ once: true }}
        transition={GROW}
        className="relative z-10 h-[3px] max-w-full rounded-full"
        style={{ background: `linear-gradient(to right, transparent, ${rgb(1)} 15%, ${rgb(1)} 85%, transparent)`, boxShadow: `0 0 12px ${rgb(0.6)}` }}
      />

      <motion.div
        initial={{ opacity: 0.6, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ ...GROW, delay: 0.3 }}
        className="relative z-10 w-full pt-6"
      >
        {children}
      </motion.div>
    </div>
  );
}
