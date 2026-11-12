// Canvas text: the words are filled with a base colour and slow wavy lines flow
// through them. Adapted from Aceternity UI's canvas-text to plain JSX; colours
// may be CSS variables ("var(--color-brass)"). Screen readers get the plain text.
import { useEffect, useRef, useState } from "react";

function resolveColor(color) {
  if (!color.startsWith("var(")) return color;
  const name = color.slice(4, -1).trim();
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || color;
}

export function CanvasText({
  text,
  className = "",
  color = "var(--color-cyan)",          // the letters
  colors = ["var(--color-brass)"],       // the flowing lines, used in turn
  animationDuration = 20,
  lineWidth = 1.5,
  lineGap = 4,
  curveIntensity = 60,
}) {
  const canvasRef = useRef(null);
  const textRef = useRef(null);
  const [box, setBox] = useState({ width: 0, height: 0, font: "", letterSpacing: "0px" });

  // Size and font of the hidden text, so the canvas matches it exactly
  useEffect(() => {
    const el = textRef.current;
    if (!el) return;
    const update = () => {
      const rect = el.getBoundingClientRect();
      const cs = getComputedStyle(el);
      setBox({
        width: Math.ceil(rect.width) || 400,
        height: Math.ceil(rect.height) || 100,
        font: `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`,
        letterSpacing: cs.letterSpacing === "normal" ? "0px" : cs.letterSpacing,
      });
    };
    update();
    document.fonts?.ready.then(update).catch(() => {});
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [text]);

  const colorKey = [color, ...colors].join("|");
  useEffect(() => {
    const canvas = canvasRef.current;
    const { width, height, font, letterSpacing } = box;
    if (!canvas || !width || !font) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    const fill = resolveColor(color);
    const lines = colors.map(resolveColor);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;

    ctx.font = font;
    if ("letterSpacing" in ctx) ctx.letterSpacing = letterSpacing;
    const m = ctx.measureText(text);
    const baselineY = (height + m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2;
    const numLines = Math.floor(height / lineGap) + 10;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const start = performance.now();
    let raf = 0;

    const draw = (now) => {
      const phase = reduced ? 0 : ((now - start) / 1000 / animationDuration) * Math.PI * 2;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      // the letters as a mask, filled with the base colour
      ctx.globalCompositeOperation = "source-over";
      ctx.font = font;
      if ("letterSpacing" in ctx) ctx.letterSpacing = letterSpacing;
      ctx.textBaseline = "alphabetic";
      ctx.textAlign = "left";
      ctx.fillStyle = "#000";
      ctx.fillText(text, 0, baselineY);
      ctx.globalCompositeOperation = "source-in";
      ctx.fillStyle = fill;
      ctx.fillRect(0, 0, width, height);

      // wavy lines, drawn only where there are letters
      ctx.globalCompositeOperation = "source-atop";
      const c1 = Math.sin(phase) * curveIntensity;
      const c2 = Math.sin(phase + 0.5) * curveIntensity * 0.6;
      ctx.lineWidth = lineWidth;
      for (let i = 0; i < numLines; i++) {
        const y = i * lineGap;
        ctx.strokeStyle = lines[i % lines.length];
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.bezierCurveTo(width * 0.33, y + c1, width * 0.66, y + c2, width, y);
        ctx.stroke();
      }
      if (!reduced) raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [box, text, colorKey, animationDuration, lineWidth, lineGap, curveIntensity]);

  return (
    <span className={`relative inline-block ${className}`}>
      <span className="sr-only">{text}</span>
      <span ref={textRef} className="invisible inline-block" aria-hidden="true">
        {text}
      </span>
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        className="pointer-events-none absolute top-0 left-0"
        style={{ width: box.width || "auto", height: box.height || "auto" }}
      />
    </span>
  );
}
