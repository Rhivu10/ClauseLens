// The ClauseLens mark: a document (with a folded corner and text lines)
// being examined by a lens, with the line under the glass drawn bolder to
// suggest a clause being read. Used in the nav, the intro's settled logo,
// and the favicon.
export default function DocumentLensMark({ className = "" }) {
  return (
    <svg viewBox="0 0 40 40" className={className} aria-hidden="true">
      {/* document */}
      <rect x="7" y="6" width="18" height="24" rx="2.2" fill="var(--color-surface)" stroke="var(--color-ink)" strokeWidth="1.6" />
      {/* folded corner */}
      <path d="M19.5 6 H25 V11.5 Z" fill="var(--color-line)" />
      <path d="M19.5 6 L25 11.5" stroke="var(--color-ink)" strokeWidth="1.1" strokeLinejoin="round" />
      {/* text lines */}
      <line x1="10.5" y1="14.5" x2="19.5" y2="14.5" stroke="var(--color-ink-faint)" strokeWidth="1.5" strokeLinecap="round" />
      <line x1="10.5" y1="18.5" x2="21.5" y2="18.5" stroke="var(--color-ink-faint)" strokeWidth="1.5" strokeLinecap="round" />
      <line x1="10.5" y1="22.5" x2="17" y2="22.5" stroke="var(--color-ink-faint)" strokeWidth="1.5" strokeLinecap="round" />

      {/* lens over the bottom-right of the page */}
      <circle cx="23.5" cy="25" r="8.6" fill="var(--color-surface)" stroke="var(--color-brass)" strokeWidth="2.2" />
      {/* the clause line highlighted under the glass */}
      <line x1="18.5" y1="25" x2="28" y2="25" stroke="var(--color-brass)" strokeWidth="1.8" strokeLinecap="round" />
      {/* glass shine */}
      <path d="M19 20.5 Q20.5 18.3 23.5 18" stroke="white" strokeWidth="1.3" strokeLinecap="round" fill="none" opacity="0.7" />
      {/* handle */}
      <line x1="29.6" y1="31.1" x2="34" y2="35.5" stroke="var(--color-brass)" strokeWidth="2.6" strokeLinecap="round" />
    </svg>
  );
}