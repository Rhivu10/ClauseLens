// Link preview: hovering (or focusing) the link pops up a picture of the page it
// opens, which leans after the cursor. Adapted from Aceternity UI's link-preview
// to plain JSX with framer-motion only (no Radix); it shows a saved image
// (imageSrc) instead of fetching a live screenshot from a third-party service.
import { useRef, useState } from "react";
import { AnimatePresence, motion, useMotionValue, useSpring } from "framer-motion";

export function LinkPreview({ children, url, imageSrc, width = 240, height = 120, className = "", ...props }) {
  const [open, setOpen] = useState(false);
  const timer = useRef(null);
  const x = useMotionValue(0);
  const translateX = useSpring(x, { stiffness: 100, damping: 15 });

  const show = () => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setOpen(true), 50);
  };
  const hide = () => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setOpen(false), 100);
  };
  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    x.set((e.clientX - r.left - r.width / 2) / 2);
  };

  return (
    <span className="relative inline-block" onMouseEnter={show} onMouseLeave={hide}>
      {/* load the picture early so it is there on the first hover */}
      <link rel="preload" as="image" href={imageSrc} />
      <a href={url} onMouseMove={onMove} onFocus={show} onBlur={hide} className={className} {...props}>
        {children}
      </a>
      <AnimatePresence>
        {open && (
          <motion.span
            initial={{ opacity: 0, y: 20, scale: 0.6 }}
            animate={{ opacity: 1, y: 0, scale: 1, transition: { type: "spring", stiffness: 260, damping: 20 } }}
            exit={{ opacity: 0, y: 20, scale: 0.6 }}
            className="absolute bottom-full left-1/2 z-50 mb-3 block origin-bottom"
            style={{ x: translateX, marginLeft: -width / 2 - 4 }}
          >
            <a
              href={url}
              target={props.target}
              rel={props.rel}
              tabIndex={-1}
              aria-hidden="true"
              className="block rounded-xl border-2 border-transparent bg-(--color-surface) p-1 shadow-xl hover:border-(--color-line)"
              style={{ fontSize: 0 }}
            >
              <img src={imageSrc} width={width} height={height} alt="" className="rounded-lg" />
            </a>
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  );
}
