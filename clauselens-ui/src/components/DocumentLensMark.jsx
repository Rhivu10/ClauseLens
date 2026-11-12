// The ClauseLens mark: the emblem of the ClauseLens logo (a "C" around a contract
// page with the scales of justice, a pen nib and a swoosh). Used in the nav and
// the intro's settled logo; the favicon is the same image (public/favicon-*.png).
// The full logo with the name and tagline is public/clauselens-logo.png.
export default function DocumentLensMark({ className = "" }) {
  return <img src="/clauselens-mark.png" alt="" aria-hidden="true" draggable="false" className={`object-contain ${className}`} />;
}
