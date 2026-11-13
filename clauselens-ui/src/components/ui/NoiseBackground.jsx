// Noise background: soft colour blobs drift slowly behind a grainy texture.
// Adapted from Aceternity UI's noise-background to plain JSX; the grain is an
// inline SVG (no image download).
import { useEffect, useRef } from "react";
import { motion, useAnimationFrame, useMotionTemplate, useMotionValue, useSpring, useTransform } from "framer-motion";

const NOISE =
  "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.5  0 0 0 0 0.5  0 0 0 0 0.5  0 0 0 1 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>\")";

function GradientLayer({ springX, springY, color, opacity, multiplier }) {
  const x = useTransform(springX, (v) => v * multiplier);
  const y = useTransform(springY, (v) => v * multiplier);
  const background = useMotionTemplate`radial-gradient(circle at ${x}px ${y}px, ${color} 0%, transparent 50%)`;
  return <motion.div className="absolute inset-0" style={{ opacity, background }} />;
}

const randomVelocity = (speed) => {
  const angle = Math.random() * Math.PI * 2;
  const m = speed * (0.5 + Math.random() * 0.5);
  return { x: Math.cos(angle) * m, y: Math.sin(angle) * m };
};

export function NoiseBackground({
  children,
  className = "",
  containerClassName = "",
  gradientColors = ["rgb(136, 46, 58)", "rgb(104, 124, 134)", "rgb(214, 160, 140)"],
  noiseIntensity = 0.25,
  speed = 0.1,
  animating = true,
}) {
  const ref = useRef(null);
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const springX = useSpring(x, { stiffness: 100, damping: 30 });
  const springY = useSpring(y, { stiffness: 100, damping: 30 });
  const stripX = useTransform(springX, (v) => v * 0.1 - 50);
  const velocity = useRef(randomVelocity(speed));
  const lastTurn = useRef(0);

  useEffect(() => {
    const r = ref.current?.getBoundingClientRect();
    if (r) {
      x.set(r.width / 2);
      y.set(r.height / 2);
    }
  }, [x, y]);

  useAnimationFrame((time) => {
    if (!animating || !ref.current) return;
    const { width, height } = ref.current.getBoundingClientRect();
    if (time - lastTurn.current > 1500 + Math.random() * 1500) {
      velocity.current = randomVelocity(speed);
      lastTurn.current = time;
    }
    const pad = 8;
    let nx = x.get() + velocity.current.x * 16;
    let ny = y.get() + velocity.current.y * 16;
    if (nx < pad || nx > width - pad || ny < pad || ny > height - pad) {
      velocity.current = randomVelocity(speed);
      lastTurn.current = time;
      nx = Math.max(pad, Math.min(width - pad, nx));
      ny = Math.max(pad, Math.min(height - pad, ny));
    }
    x.set(nx);
    y.set(ny);
  });

  return (
    <div
      ref={ref}
      className={`group relative overflow-hidden rounded-2xl bg-(--color-surface-2) p-2 shadow-[inset_0_0.5px_1px_rgba(41,36,29,0.25),0_1px_0_rgba(255,255,255,0.7)] ${containerClassName}`}
    >
      <GradientLayer springX={springX} springY={springY} color={gradientColors[0]} opacity={0.45} multiplier={1} />
      <GradientLayer springX={springX} springY={springY} color={gradientColors[1]} opacity={0.35} multiplier={0.7} />
      <GradientLayer springX={springX} springY={springY} color={gradientColors[2] || gradientColors[0]} opacity={0.3} multiplier={1.2} />

      {/* coloured strip along the top edge */}
      <motion.div
        className="absolute inset-x-0 top-0 h-1 opacity-80 blur-sm"
        style={{ background: `linear-gradient(to right, ${gradientColors.join(", ")})`, x: animating ? stripX : 0 }}
      />

      {/* grain */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{ backgroundImage: NOISE, opacity: noiseIntensity, mixBlendMode: "overlay" }}
      />

      <div className={`relative z-10 ${className}`}>{children}</div>
    </div>
  );
}
