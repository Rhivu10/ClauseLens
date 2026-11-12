import { NavLink } from "react-router-dom";
import DocumentLensMark from "@/components/DocumentLensMark";

const links = [
  { to: "/", label: "Intake", num: "01" },
  { to: "/compare", label: "Alignment", num: "02" },
  { to: "/consult", label: "Consult", num: "03" },
];

export default function Nav() {
  return (
    <header className="border-b border-(--color-line) bg-(--color-bg)/85 backdrop-blur sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <DocumentLensMark className="w-8 h-8" />
          <div className="leading-none">
            <div className="font-semibold tracking-tight text-lg">ClauseLens</div>
            <div className="text-[10px] font-mono-label uppercase text-(--color-ink-faint)">
              policy diff · impact analysis
            </div>
          </div>
        </div>
        <nav className="flex items-center gap-1">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) =>
                `group flex items-center gap-2 px-3 py-2 rounded-full text-sm transition-colors ${
                  isActive
                    ? "bg-(--color-surface-2) text-(--color-ink)"
                    : "text-(--color-ink-dim) hover:text-(--color-ink)"
                }`
              }
            >
              <span className="font-mono text-[10px] text-(--color-brass)">{l.num}</span>
              {l.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}