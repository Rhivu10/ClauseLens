import { useEffect, useLayoutEffect, useRef, useState } from "react";
import Magnifier from "@/components/Magnifier";
import { LiquidGlass } from "@/components/ui/LiquidGlass";
import { motion, useAnimationFrame, useMotionValue, useTransform, animate } from "framer-motion";

// Ambient hero animation: a magnifier lens reads the app's introduction the way
// an eye does — travelling word by word along each justified line of a document
// sheet, then dropping to the next. Text under the glass is sharp; everything
// else is a soft blurred ghost.
//
// Click the glass to pick it up: it stops reading and can be dragged anywhere on
// the page, bending whatever is behind it (liquid glass); double-click lets it
// read again (the lens's tooltip says so).

const R = 46; // lens radius, px
const GLASS = Math.round(R * 0.885); // the see-through part inside the brass rim
const MAG = 1.6;                      // how much the glass magnifies
const LOOP_SECONDS = 34;
const READ = 0.92; // fraction of the loop spent reading (rest = glide back to top)

const TITLE = "How ClauseLens reads a contract";
const BODY_BEFORE =
  "ClauseLens reads two versions of the same contract at once — the prior draft and the redline — parsing each with layout-aware extraction, so headings, numbering and defined terms survive intact. Every clause is indexed and aligned to its counterpart across both documents, producing ";
const BODY_MARK = "an exact map of what changed";   // highlighted, like a marked-up clause
const BODY_AFTER =
  ": what was added, what was quietly dropped, and what shifted just enough to move the risk.";

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

// The document: a policy sheet in a leather folder, with a wax seal, twine and a
// fountain pen (public/policy-folder.webp: a picture with its printed text
// removed). Our text sits where the picture's text was: right of the twine, with
// the body flowing around the pen. Percentages are of the picture (1079 x 1457).
// (padding percentages are of the width: top 27% of the height = 36.5% of the width)
const PAPER = { aspect: 1079 / 1457, left: "22%", right: "9%", top: "36.5%" };

function Sheet({ sharp, bodyRef }) {
  const ink = sharp ? "var(--color-ink)" : "var(--color-ink-dim)";
  const hide = {};
  return (
    <div className="absolute inset-0" style={{ paddingLeft: PAPER.left, paddingRight: PAPER.right, paddingTop: PAPER.top }}>
      <div className="h-full">
        {/* keeps the text off the pen: an empty float at the pen's place (from 62% of the
            picture down, right of 76%), so the lines below it get shorter */}
        <div className="float-right w-0" style={{ height: "48%" }} />
        <div className="float-right clear-right h-full" style={{ width: "25%" }} />

        <div className="text-center" style={hide}>
          <div className="font-serif font-semibold text-[3.1cqw] tracking-[0.02em] uppercase whitespace-nowrap text-(--color-ink)">
            {TITLE}
          </div>
          <div className="mt-1 font-serif-text text-[3.1cqw] text-(--color-ink-dim)">
            Version 2.0 | ClauseLens
          </div>
          <div className="mt-3 h-px bg-(--color-ink)/60" />
        </div>
        <div className="mt-3 mb-1 font-serif font-semibold text-[3.85cqw] text-(--color-ink)" style={hide}>
          Clause 1: How it reads
        </div>
        <p
          ref={bodyRef}
          className="font-serif-text text-[3.25cqw] leading-[1.6] text-left"
          style={{ color: ink }}
        >
          {BODY_BEFORE}
          <span className="rounded-[3px] px-0.5 font-semibold" style={{ background: "rgba(136,46,58,0.2)" }}>
            {BODY_MARK}
          </span>
          {BODY_AFTER}
        </p>
      </div>
    </div>
  );
}

export default function DocumentLensScan({ className = "" }) {
  const rootRef = useRef(null);
  const wrapRef = useRef(null);
  const bodyEl = useRef(null);
  const geomRef = useRef(null);
  const offRef = useRef({ x: 0, y: 0 });     // the sheet's top-left inside the root
  const dragRef = useRef(null);               // {dx, dy} from the pointer to the lens centre while dragging
  const [ready, setReady] = useState(false);
  const [held, setHeld] = useState(false);    // picked up: stays put until dragged, double-click lets it read again
  const [dragging, setDragging] = useState(false);
  const heldRef = useRef(false);

  // the lens centre in root coordinates, and how far it is lifted off the page (0 = reading)
  const x = useMotionValue(-200);
  const y = useMotionValue(-200);
  const lift = useMotionValue(0);

  useLayoutEffect(() => {
    function measure() {
      const root = rootRef.current;
      const wrap = wrapRef.current;
      const body = bodyEl.current;
      if (!root || !wrap || !body) return;
      const rr = root.getBoundingClientRect();
      const wr = wrap.getBoundingClientRect();
      offRef.current = { x: wr.left - rr.left, y: wr.top - rr.top };

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

  // Reading: every frame the loop moves the lens along the lines; picked up, it pauses
  const elapsed = useRef(0);
  useAnimationFrame((_, delta) => {
    if (heldRef.current || !geomRef.current) return;
    elapsed.current += Math.min(delta, 100) / 1000;
    const r = readingPos((elapsed.current / LOOP_SECONDS) % 1, geomRef.current);
    x.set(r.x + offRef.current.x);
    y.set(r.y + offRef.current.y);
    lift.set(r.lift);
  });

  function moveTo(clientX, clientY) {
    const rr = rootRef.current.getBoundingClientRect();
    const d = dragRef.current || { dx: 0, dy: 0 };
    x.set(Math.min(rr.width - R, Math.max(R, clientX - rr.left + d.dx)));
    y.set(Math.min(rr.height - R, Math.max(R, clientY - rr.top + d.dy)));
  }

  function onPointerDown(e) {
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    const rr = rootRef.current.getBoundingClientRect();
    dragRef.current = { dx: x.get() - (e.clientX - rr.left), dy: y.get() - (e.clientY - rr.top) };
    if (!heldRef.current) {
      heldRef.current = true;
      setHeld(true);
      animate(lift, 0, { duration: 0.2 });
    }
    setDragging(true);
  }

  function onPointerMove(e) {
    if (dragging) moveTo(e.clientX, e.clientY);
  }

  function onPointerUp() {
    dragRef.current = null;
    setDragging(false);
  }

  // Double-click: put it back on the page to read from where the loop left off
  function release() {
    heldRef.current = false;
    setHeld(false);
  }

  // The magnified view: a sharp copy of the document, scaled up around the lens
  // centre and clipped to the glass (the clip is scaled too, so its radius is /MAG)
  const clipPath = useTransform([x, y], ([lx, ly]) =>
    `circle(${(GLASS - 1) / MAG}px at ${lx - offRef.current.x}px ${ly - offRef.current.y}px)`
  );
  const origin = useTransform([x, y], ([lx, ly]) => `${lx - offRef.current.x}px ${ly - offRef.current.y}px`);
  const sharpOpacity = useTransform(lift, (l) => 1 - l * 0.8);
  const lensScale = useTransform(lift, (l) => 1 + l * 0.5);
  const lensOpacity = useTransform(lift, (l) => 1 - l * 0.55);

  return (
    <div ref={rootRef} className={`overflow-hidden pointer-events-none ${className}`}>
      <div
        ref={wrapRef}
        aria-hidden="true"
        className="@container absolute top-1/2 -translate-y-1/2 left-[56%] right-[7%] max-w-[350px]"
      >
        {/* the document: the folder picture, with soft blurred ghost text on it */}
        <div
          className="relative bg-contain bg-no-repeat bg-center drop-shadow-[0_28px_40px_rgba(41,36,29,0.3)]"
          style={{ aspectRatio: PAPER.aspect, backgroundImage: "url(/policy-folder.webp)", opacity: 0.86 }}
        />
        <div className="absolute inset-0" style={{ filter: "blur(2px)", opacity: 0.85 }}>
          <Sheet bodyRef={bodyEl} />
        </div>

        {/* sharp copy of the text, revealed only under the glass */}
        {ready && (
          <motion.div
            className="absolute inset-0 bg-contain bg-no-repeat bg-center"
            style={{ clipPath, WebkitClipPath: clipPath, transformOrigin: origin, scale: MAG, opacity: sharpOpacity, backgroundImage: "url(/policy-folder.webp)" }}
          >
            <Sheet sharp />
          </motion.div>
        )}
      </div>

      {/* warm glow riding with the lens */}
      {ready && (
        <motion.div
          className="absolute rounded-full"
          style={{
            left: x,
            top: y,
            x: "-50%",
            y: "-50%",
            width: R * 4.4,
            height: R * 4.4,
            background: "radial-gradient(circle, rgba(58,31,31,0.20) 0%, transparent 62%)",
          }}
        />
      )}

      {/* the lens: click to pick it up, drag it anywhere; while held its glass bends the page behind it */}
      {ready && (
        <motion.div
          role="button"
          tabIndex={-1}
          title={held ? "Drag the glass · double-click to let it read again" : "Pick up the magnifying glass"}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onDoubleClick={release}
          className={`absolute z-20 pointer-events-auto touch-none select-none ${dragging ? "cursor-grabbing" : "cursor-grab"}`}
          style={{
            left: x,
            top: y,
            x: "-50%",
            y: "-50%",
            width: R * 2,
            height: R * 2,
            scale: lensScale,
            opacity: lensOpacity,
          }}
        >
          <motion.div
            className="relative w-full h-full"
            animate={{ scale: dragging ? 1.06 : 1 }}
            transition={{ type: "spring", stiffness: 300, damping: 22 }}
          >
          {held && (
            <LiquidGlass
              radius={GLASS}
              scale={26}
              aberration={8}
              bevelDepth={18}
              frost={0.3}
              tint={0.06}
              className="absolute"
              style={{ left: R - GLASS, top: R - GLASS, width: GLASS * 2, height: GLASS * 2 }}
            />
          )}
            <Magnifier cx={R} cy={R} r={R} shadow="0 8px 20px rgba(41,36,29,0.3)" />
          </motion.div>
        </motion.div>
      )}

    </div>
  );
}
