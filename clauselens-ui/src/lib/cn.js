import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

// Standard shadcn/aceternity-style class combiner — merges Tailwind classes
// intelligently so later conditional classes correctly override earlier ones.
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}
