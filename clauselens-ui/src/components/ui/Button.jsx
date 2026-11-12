// The ClauseLens button: a pill inside a noise-background frame whose colours
// drift slowly (NoiseBackground, adapted from Aceternity UI).
//
//   variant "solid"  dark wine-brown pill, light text (main actions)
//   variant "soft"   paper pill, dark text (secondary actions)
//   as               the element to render: "button" (default), Link, "a", ...
import { NoiseBackground } from "@/components/ui/NoiseBackground";

const PILL = {
  solid:
    "bg-linear-to-r from-(--color-brass) via-(--color-brass) to-[#55302f] text-(--color-brass-ink) shadow-[inset_0_1px_0_rgba(255,255,255,0.12),0_1px_1px_rgba(41,36,29,0.3)]",
  soft:
    "bg-linear-to-r from-(--color-surface) via-(--color-surface) to-white text-(--color-ink) shadow-[inset_0_2px_0_rgba(255,255,255,0.9),0_0.5px_1px_rgba(41,36,29,0.3)]",
};

export function Button({
  as: Tag = "button",
  variant = "solid",
  containerClassName = "",
  className = "",
  disabled,
  children,
  ...props
}) {
  return (
    <NoiseBackground
      animating={!disabled}
      containerClassName={`w-fit p-1.5 rounded-full ${disabled ? "opacity-40" : ""} ${containerClassName}`}
    >
      <Tag
        disabled={disabled}
        className={`h-full w-full inline-flex items-center justify-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium cursor-pointer transition-all duration-100 active:scale-[0.98] disabled:cursor-not-allowed disabled:active:scale-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-(--color-accent) ${PILL[variant] || PILL.solid} ${className}`}
        {...props}
      >
        {children}
      </Tag>
    </NoiseBackground>
  );
}
