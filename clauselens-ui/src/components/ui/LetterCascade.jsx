// Letter cascade: on hover (or click), every letter tilts back and a copy flips up
// from below in a wave. Adapted from Componentry's letter-cascade to plain JSX;
// words stay whole but the phrase can wrap between them. Screen readers get the
// plain text.
import { useCallback, useRef } from "react";
import { motion, stagger, useAnimate } from "framer-motion";

export function LetterCascade({
  text,
  className = "",
  letterClassName = "",
  staggerDuration = 0.03,
  staggerFrom = "first",
  stiffness = 220,
  damping = 16,
  triggerOnClick = false,
  onComplete,
}) {
  const [scope, animate] = useAnimate();
  const busy = useRef(false);

  const trigger = useCallback(() => {
    if (busy.current) return;
    busy.current = true;
    const opts = { type: "spring", stiffness, damping, delay: stagger(staggerDuration, { from: staggerFrom }) };

    // front tilts back while the echo flips in from below, then both reset at once
    animate(".cascade-front", { rotateX: 90, opacity: 0, y: -6, filter: "blur(4px)" }, opts).then(() =>
      animate(".cascade-front", { rotateX: 0, opacity: 1, y: 0, filter: "blur(0px)" }, { duration: 0 }).then(() => {
        busy.current = false;
        onComplete?.();
      })
    );
    animate(".cascade-echo", { rotateX: 0, opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }, opts).then(() =>
      animate(".cascade-echo", { rotateX: -90, opacity: 0, y: 6, scale: 0.8, filter: "blur(4px)" }, { duration: 0 })
    );
  }, [animate, staggerDuration, staggerFrom, stiffness, damping, onComplete]);

  const words = text.split(" ");
  return (
    <span
      ref={scope}
      className={`cursor-default ${className}`}
      {...(triggerOnClick ? { onClick: trigger } : { onMouseEnter: trigger })}
    >
      <span className="sr-only">{text}</span>
      {words.map((word, w) => (
        <span key={w} aria-hidden="true">
          <span className="inline-block whitespace-nowrap">
            {word.split("").map((letter, i) => (
              <span key={i} className="relative inline-block" style={{ perspective: "500px" }}>
                <motion.span
                  className={`cascade-front inline-block ${letterClassName}`}
                  style={{ transformOrigin: "bottom center", backfaceVisibility: "hidden" }}
                >
                  {letter}
                </motion.span>
                <motion.span
                  className={`cascade-echo absolute inset-0 inline-block ${letterClassName}`}
                  style={{
                    rotateX: -90,
                    opacity: 0,
                    y: 6,
                    scale: 0.8,
                    filter: "blur(4px)",
                    transformOrigin: "top center",
                    backfaceVisibility: "hidden",
                  }}
                >
                  {letter}
                </motion.span>
              </span>
            ))}
          </span>
          {w < words.length - 1 ? " " : null}
        </span>
      ))}
    </span>
  );
}
