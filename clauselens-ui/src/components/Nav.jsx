import { useLocation, useNavigate } from "react-router-dom";
import DocumentLensMark from "@/components/DocumentLensMark";
import MagneticDock from "@/components/ui/MagneticDock";

const Icon = ({ children }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-full h-full">
    {children}
  </svg>
);

const links = [
  {
    to: "/",
    match: ["/", "/upload"],
    label: "Intake",
    num: "01",
    icon: (
      <Icon>
        <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
        <path d="M14 3v5h5M12 17v-6m-3 3l3-3 3 3" />
      </Icon>
    ),
  },
  {
    to: "/compare",
    match: ["/compare"],
    label: "Alignment",
    num: "02",
    icon: (
      <Icon>
        <rect x="3" y="4" width="7" height="16" rx="1.5" />
        <rect x="14" y="4" width="7" height="16" rx="1.5" />
        <path d="M10 9h4m-4 6h4" />
      </Icon>
    ),
  },
  {
    to: "/consult",
    match: ["/consult"],
    label: "Consult",
    num: "03",
    icon: (
      <Icon>
        <path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z" />
        <path d="M9 10h6m-6 3.5h4" />
      </Icon>
    ),
  },
];

export default function Nav() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const items = links.map((l) => ({
    id: l.to,
    label: l.label,
    num: l.num,
    icon: l.icon,
    isActive: l.match.includes(pathname),
    onClick: () => navigate(l.to),
  }));

  return (
    <header className="shadow-[inset_0_3px_0_var(--color-accent)] border-b border-(--color-line) bg-(--color-bg)/85 backdrop-blur sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <DocumentLensMark className="w-10 h-10" />
          <div className="leading-none">
            <div className="font-semibold tracking-tight text-lg">ClauseLens</div>
            <div className="text-[10px] font-mono-label uppercase text-(--color-ink-faint)">
              policy diff · impact analysis
            </div>
          </div>
        </div>
        <nav aria-label="Pages">
          <MagneticDock items={items} />
        </nav>
      </div>
    </header>
  );
}
