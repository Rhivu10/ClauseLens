import { useEffect, useRef, useState } from "react";
import { motion, useMotionValue, useTransform, animate, AnimatePresence } from "framer-motion";
import DocumentLensMark from "@/components/DocumentLensMark";
import Magnifier from "@/components/Magnifier";

const WORD = "ClauseLens";
const LENS_RADIUS = 42;
const RING_BOX = 112; // bounding box for the ring SVG — bigger than the lens circle so the handle has room to stick out
const SWEEP_DURATION = 1.8;
const MAG_SCALE = 1.5;
const LETTER_SPACING = "0.02em";

function Letters() {
  return WORD.split("").map((ch, i) => (
    <span key={i} className={i >= 6 ? "text-(--color-brass)" : "text-(--color-ink)"}>
      {ch}
    </span>
  ));
}

export default function IntroSplash({ onDone }) {
  const [visible, setVisible] = useState(true);
  const [sweepDone, setSweepDone] = useState(false);
  const [width, setWidth] = useState(0);
  const textRef = useRef(null);

  const lensX = useMotionValue(-LENS_RADIUS);
  const lensScale = useMotionValue(1);
  const lensOpacity = useMotionValue(1);

  // real optical loupe: scale-transform the duplicate text around the lens's
  // own live position, and clip it to a circle at that same position — the
  // standard CSS technique for a magnifier that shows real content.
  const clipPath = useTransform(lensX, (v) => `circle(${LENS_RADIUS}px at ${v}px 50%)`);
  const transformOrigin = useTransform(lensX, (v) => `${v}px center`);
  const ringLeft = useTransform(lensX, (v) => v - LENS_RADIUS);
  const ringVisualLeft = useTransform(lensX, (v) => v - RING_BOX / 2);

  useEffect(() => {
    if (textRef.current) setWidth(textRef.current.getBoundingClientRect().width);
  }, []);

  useEffect(() => {
    if (!width) return;
    const controls = animate(lensX, width + LENS_RADIUS * 2, {
      duration: SWEEP_DURATION,
      ease: "easeInOut",
      onComplete: () => {
        setSweepDone(true);
        animate(lensScale, 2.8, { duration: 0.55, ease: "easeIn" });
        animate(lensOpacity, 0, { duration: 0.45, delay: 0.1 });
      },
    });
    return () => controls.stop();
  }, [width]);

  useEffect(() => {
    const t = setTimeout(() => setVisible(false), (SWEEP_DURATION + 1.2) * 1000);
    return () => clearTimeout(t);
  }, []);

  return (
    <AnimatePresence onExitComplete={onDone}>
      {visible && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-(--color-bg)"
          initial={{ clipPath: "circle(150% at 50% 50%)" }}
          exit={{ clipPath: "circle(0% at 50% 50%)" }}
          transition={{ duration: 0.9, ease: [0.76, 0, 0.24, 1] }}
        >
          <div className="flex items-center gap-4">
            {/* settled logo mark — appears once the lens has swept through */}
            <motion.div
              initial={{ opacity: 0, scale: 0.4 }}
              animate={sweepDone ? { opacity: 1, scale: 1 } : {}}
              transition={{ type: "spring", stiffness: 260, damping: 20 }}
            >
              <DocumentLensMark className="w-20 h-20" />
            </motion.div>

            <div className="relative flex items-center" style={{ height: 72 }}>
              {/* base text, always visible */}
              <div
                ref={textRef}
                className="font-sans font-semibold text-4xl md:text-5xl tracking-tight whitespace-nowrap flex items-center h-full"
                style={{ letterSpacing: LETTER_SPACING }}
              >
                <Letters />
              </div>

              {/* opaque disc that blocks the base text behind the lens, so only the magnified copy shows through the glass */}
              {width > 0 && !sweepDone && (
                <motion.div
                  className="absolute rounded-full bg-(--color-bg) pointer-events-none"
                  style={{
                    left: ringLeft,
                    top: "50%",
                    translateY: "-50%",
                    width: LENS_RADIUS * 2,
                    height: LENS_RADIUS * 2,
                  }}
                />
              )}

              {/* magnified duplicate of the SAME text, clipped to the lens circle */}
              {width > 0 && !sweepDone && (
                <motion.div
                  className="absolute inset-0 whitespace-nowrap font-sans font-semibold text-4xl md:text-5xl tracking-tight pointer-events-none flex items-center h-full"
                  style={{ clipPath, transformOrigin, scale: MAG_SCALE, letterSpacing: LETTER_SPACING }}
                >
                  <Letters />
                </motion.div>
              )}

              {/* the magnifying glass riding along the sweep */}
              <motion.div
                className="absolute pointer-events-none"
                style={{
                  left: ringVisualLeft,
                  top: "50%",
                  translateY: "-50%",
                  width: RING_BOX,
                  height: RING_BOX,
                  scale: lensScale,
                  opacity: lensOpacity,
                }}
              >
                <Magnifier cx={RING_BOX / 2} cy={RING_BOX / 2} r={LENS_RADIUS} />
              </motion.div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}