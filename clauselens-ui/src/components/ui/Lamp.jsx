// Lamp: a thin bar of light that widens as it comes into view and pours a soft
// pool of light below it (and a faint halo above). Adapted from Aceternity UI's
// lamp for a light page: the light is drawn with radial gradients instead of
// beams hidden behind dark panels. Default colour: warm lamp gold.
import { motion } from "framer-motion";

const GROW = { delay: 0.2, duration: 0.9, ease: "easeInOut" };

export function Lamp({ children, color = "214, 150, 52", width = 30, className = "" }) {
  const rgb = (a) => `rgba(${color}, ${a})`;
  const inView = { viewport: { once: true }, transition: GROW };

  return (
    <div className={`relative isolate flex flex-col items-center pt-14 ${className}`}>
      <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center" aria-hidden="true">
        {/* faint halo above the bar */}
        <motion.div
          initial={{ width: `${width * 0.6}rem`, opacity: 0.4 }}
          whileInView={{ width: `${width * 1.4}rem`, opacity: 1 }}
          {...inView}
          className="absolute top-0 h-14 max-w-[100vw]"
          style={{ background: `radial-gradient(ellipse 50% 100% at 50% 100%, ${rgb(0.22)} 0%, transparent 75%)` }}
        />
        {/* the pool of light under the bar: brightest at its middle, fading down and out */}
        <motion.div
          initial={{ width: `${width * 0.5}rem`, opacity: 0.5 }}
          whileInView={{ width: `${width * 1.15}rem`, opacity: 1 }}
          {...inView}
          className="absolute top-14 h-48 max-w-[100vw] blur-md"
          style={{
            background: `radial-gradient(ellipse 50% 100% at 50% 0%, ${rgb(0.55)} 0%, ${rgb(0.32)} 30%, ${rgb(0.1)} 60%, transparent 80%)`,
          }}
        />
        {/* the bar */}
        <motion.div
          initial={{ width: `${width * 0.4}rem` }}
          whileInView={{ width: `${width}rem` }}
          {...inView}
          className="absolute top-14 h-[2px] max-w-[90vw] rounded-full"
          style={{ background: `rgb(${color})`, boxShadow: `0 0 10px ${rgb(0.9)}, 0 0 24px ${rgb(0.5)}` }}
        />
      </div>

      <motion.div
        initial={{ opacity: 0.6, y: 14 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ ...GROW, delay: 0.35 }}
        className="relative z-10 w-full pt-14"
      >
        {children}
      </motion.div>
    </div>
  );
}
