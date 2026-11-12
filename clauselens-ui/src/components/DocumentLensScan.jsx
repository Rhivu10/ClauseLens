import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { motion, useMotionValue, useTransform, animate } from "framer-motion";

// Ambient hero animation: a magnifier lens reads the app's introduction the way
// an eye does — travelling word by word along each justified line of a document
// sheet, then dropping to the next. Text under the glass is sharp; everything
// else is a soft blurred ghost.

const R = 46; // lens radius, px
const LOOP_SECONDS = 34;
const READ = 0.92; // fraction of the loop spent reading (rest = glide back to top)

const TITLE = "How ClauseLens reads a contract";
const BODY =
  "ClauseLens reads two versions of the same contract at once — the prior draft and the redline — parsing each with layout-aware extraction, so headings, numbering and defined terms survive intact. Every clause is indexed and aligned to its counterpart across both documents, producing an exact map of what changed: what was added, what was quietly dropped, and what shifted just enough to move the risk.";

const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
const lerp = (a, b, t) => a + (b - a) * t;

// Where is the lens at loop-progress p, given measured line geometry?
function readingPos(p, g) {
  if (!g || !g.ys.length) return { x: 0, y: 0, lift: 1 };
  const N = g.ys.length;

  if (p >= READ) {
    const t = ease((p - READ) / (1 - READ));
    return {
      x: lerp(g.lefts[N - 1] + g.widths[N - 1], g.lefts[0], t),
      y: lerp(g.ys[N - 1], g.ys[0], t),
      lift: Math.sin(t * Math.PI), // 0 → 1 → 0: lens rises off the page
    };
  }

  const scaled = (p / READ) * N;
  const i = Math.min(Math.floor(scaled), N - 1);
  const local = scaled - i;

  if (local <= 0.84 || i === N - 1) {
    const t = Math.min(local / 0.84, 1);
    return { x: g.lefts[i] + t * g.widths[i], y: g.ys[i], lift: 0 };
  }

  const t = ease((local - 0.84) / 0.16);
  return {
    x: lerp(g.lefts[i] + g.widths[i], g.lefts[i + 1], t),
    y: lerp(g.ys[i], g.ys[i + 1], t),
    lift: 0.4 * Math.sin(t * Math.PI),
  };
}

function Sheet({ sharp, bodyRef }) {
  return (
    <div
      className={
        sharp
          ? "px-9 py-12 md:px-10 md:py-16"
          : "px-9 py-12 md:px-10 md:py-16 rounded-[3px] bg-(--color-surface) border border-(--color-line) shadow-[0_28px_70px_-24px_rgba(41,36,29,0.35)]"
      }
    >
      <div
        className={`font-mono text-[11px] tracking-[0.22em] uppercase mb-3 ${
          sharp ? "opacity-0" : "text-(--color-ink-faint)"
        }`}
      >
        {TITLE}
      </div>
      <div className={`h-px mb-6 ${sharp ? "opacity-0" : "bg-(--color-line)"}`} />
      <p
        ref={bodyRef}
        className="font-serif-text text-lg md:text-xl leading-[1.9] text-justify"
        style={sharp ? { color: "var(--color-ink)" } : { color: "var(--color-ink-dim)" }}
      >
        {BODY}
      </p>
    </div>
  );
}

export default function DocumentLensScan({ className = "" }) {
  const wrapRef = useRef(null);
  const bodyEl = useRef(null);
  const geomRef = useRef(null);
  const [ready, setReady] = useState(false);

  const progress = useMotionValue(0);

  useLayoutEffect(() => {
    function measure() {
      const wrap = wrapRef.current;
      const body = bodyEl.current;
      if (!wrap || !body) return;
      const wr = wrap.getBoundingClientRect();

      // one rect per rendered (justified) line of the paragraph
      const range = document.createRange();
      range.selectNodeContents(body);
      const rects = Array.from(range.getClientRects()).filter((r) => r.width > 6 && r.height > 6);

      const lines = [];
      rects.forEach((r) => {
        if (!lines.some((l) => Math.abs(l.top - r.top) < 5)) lines.push(r);
      });
      if (!lines.length) return;

      geomRef.current = {
        lefts: lines.map((r) => r.left - wr.left),
        widths: lines.map((r) => r.width),
        ys: lines.map((r) => r.top - wr.top + r.height / 2),
      };
      setReady(true);
    }

    measure();
    const t1 = setTimeout(measure, 400);
    document.fonts?.ready.then(measure).catch(() => {});
    window.addEventListener("resize", measure);
    return () => {
      clearTimeout(t1);
      window.removeEventListener("resize", measure);
    };
  }, []);

  useEffect(() => {
    const controls = animate(progress, 1, {
      duration: LOOP_SECONDS,
      ease: "linear",
      repeat: Infinity,
    });
    return () => controls.stop();
  }, [progress]);

  const cx = useTransform(progress, (p) => readingPos(p, geomRef.current).x);
  const cy = useTransform(progress, (p) => readingPos(p, geomRef.current).y);
  const lift = useTransform(progress, (p) => readingPos(p, geomRef.current).lift);

  const mask = useTransform(
    [cx, cy],
    ([x, y]) =>
      `radial-gradient(circle ${R * 1.2}px at ${x}px ${y}px, #000 ${R * 0.5}px, rgba(0,0,0,0.45) ${R * 0.78}px, transparent ${R * 0.98}px)`
  );
  const sharpOpacity = useTransform(lift, (l) => 1 - l * 0.8);
  const lensScale = useTransform(lift, (l) => 1 + l * 0.5);
  const lensOpacity = useTransform(lift, (l) => 1 - l * 0.55);

  return (
    <div className={`overflow-hidden pointer-events-none ${className}`} aria-hidden="true">
      <div
        ref={wrapRef}
        className="absolute top-1/2 -translate-y-1/2 left-[50%] right-[6%] max-w-[460px]"
      >
        {/* the document sheet — soft blurred ghost text */}
        <div style={{ filter: "blur(2.4px)", opacity: 0.9 }}>
          <Sheet bodyRef={bodyEl} />
        </div>

        {/* sharp copy of the text, revealed only under the glass */}
        {ready && (
          <motion.div
            className="absolute inset-0"
            style={{ maskImage: mask, WebkitMaskImage: mask, opacity: sharpOpacity }}
          >
            <Sheet sharp />
          </motion.div>
        )}

        {/* warm glow riding with the lens */}
        {ready && (
          <motion.div
            className="absolute rounded-full"
            style={{
              left: cx,
              top: cy,
              x: "-50%",
              y: "-50%",
              width: R * 4.4,
              height: R * 4.4,
              background: "radial-gradient(circle, rgba(106,69,38,0.20) 0%, transparent 62%)",
            }}
          />
        )}

        {/* the lens */}
        {ready && (
          <motion.div
            className="absolute"
            style={{
              left: cx,
              top: cy,
              x: "-50%",
              y: "-50%",
              width: R * 2,
              height: R * 2,
              scale: lensScale,
              opacity: lensOpacity,
            }}
          >
            <svg viewBox="0 0 220 220" width="100%" height="100%">
              <defs>
                <radialGradient id="scanGlass" cx="38%" cy="34%" r="70%">
                  <stop offset="0%" stopColor="#ffffff" stopOpacity="0.42" />
                  <stop offset="45%" stopColor="#ffffff" stopOpacity="0.06" />
                  <stop offset="100%" stopColor="var(--color-brass)" stopOpacity="0.12" />
                </radialGradient>
              </defs>
              <circle cx="110" cy="110" r="102" fill="url(#scanGlass)" />
              <circle cx="110" cy="110" r="102" fill="none" stroke="var(--color-brass)" strokeWidth="7" />
              <circle cx="110" cy="110" r="95" fill="none" stroke="#ffffff" strokeOpacity="0.4" strokeWidth="1.8" />
              <line
                x1={110 + 102 * 0.707}
                y1={110 + 102 * 0.707}
                x2={110 + 102 * 0.707 + 48}
                y2={110 + 102 * 0.707 + 48}
                stroke="var(--color-brass)"
                strokeWidth="15"
                strokeLinecap="round"
              />
            </svg>
          </motion.div>
        )}
      </div>

      {/* left scrim so the headline stays readable */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(90deg, var(--color-bg) 0%, color-mix(in srgb, var(--color-bg) 82%, transparent) 26%, transparent 46%)",
        }}
      />
    </div>
  );
}
