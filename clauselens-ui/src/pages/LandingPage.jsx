import { Link } from "react-router-dom";
import DocumentLensScan from "@/components/DocumentLensScan";
import MagneticButton from "@/components/MagneticButton";
import { CanvasText } from "@/components/ui/CanvasText";

export default function LandingPage() {
  return (
    <section className="relative min-h-[calc(100vh-4rem)] flex items-center overflow-hidden">
      <DocumentLensScan className="absolute inset-0" />
      <div className="relative z-10 w-full max-w-7xl mx-auto px-6">
        <div className="max-w-2xl">
          <h1 className="font-sans font-semibold text-5xl md:text-6xl leading-[1.05] tracking-tight mb-6 [text-shadow:0_2px_18px_rgba(41,36,29,0.22)]">
            Read two versions
            <br />
            of a policy{" "}
            <CanvasText
              text="at once"
              color="var(--color-cyan)"
              colors={[
                "rgba(106, 69, 38, 0.95)",
                "rgba(106, 69, 38, 0.8)",
                "rgba(106, 69, 38, 0.65)",
                "rgba(106, 69, 38, 0.5)",
                "rgba(106, 69, 38, 0.35)",
                "rgba(106, 69, 38, 0.2)",
              ]}
              lineGap={4}
              animationDuration={20}
            />
            .
          </h1>
          <p className="text-(--color-ink-dim) text-lg max-w-lg mb-9 [text-shadow:0_1px_10px_rgba(41,36,29,0.14)]">
            ClauseLens parses, indexes and aligns two policy documents clause by
            clause — so you see exactly what changed, and can ask why.
          </p>
          <div className="flex items-center gap-5">
            <Link to="/upload">
              <MagneticButton>Start a comparison</MagneticButton>
            </Link>
            <a
              href="https://github.com/Rhivu10/ClauseLens"
              target="_blank"
              rel="noreferrer"
              className="text-sm text-(--color-ink-dim) underline underline-offset-4 hover:text-(--color-ink)"
            >
              View the repo
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
