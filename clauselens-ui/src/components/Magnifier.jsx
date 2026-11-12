// The magnifying glass (public/magnifier.png: brass rim, see-through glass, wooden
// handle), drawn so its lens is a circle of radius r centred at (cx, cy) inside the
// parent, which must be positioned. The handle sticks out to the lower right.
//
// Lens geometry in the image, as fractions of its size (640 x 595): centre at
// (0.323, 0.351), outer rim radius 0.3235 of the width.
const CX = 0.323;
const CY = 0.351;
const RIM = 0.3235;
const ASPECT = 595 / 640;

export default function Magnifier({ cx, cy, r, shadow = "0 6px 16px rgba(41,36,29,0.35)" }) {
  const width = r / RIM;
  return (
    <img
      src="/magnifier.png"
      alt=""
      aria-hidden="true"
      draggable="false"
      className="absolute pointer-events-none select-none"
      style={{
        left: cx - CX * width,
        top: cy - CY * width * ASPECT,
        width,
        maxWidth: "none",
        filter: `drop-shadow(${shadow})`,
      }}
    />
  );
}
