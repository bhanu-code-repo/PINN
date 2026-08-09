import { clsx } from "clsx";
import type { LucideIcon } from "lucide-react";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "default" | "brand" | "success" | "warning" | "danger";
  icon?: LucideIcon;
}

const variants = {
  default: "bg-slate-50 text-slate-600 border-slate-200",
  brand: "bg-brand-muted text-brand border-brand-200",
  success: "bg-emerald-50 text-emerald-700 border-emerald-200",
  warning: "bg-amber-50 text-amber-700 border-amber-200",
  danger: "bg-red-50 text-red-700 border-red-200",
};

export function Badge({ children, variant = "default", icon: Icon }: BadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium",
        variants[variant],
      )}
    >
      {Icon && <Icon className="w-3.5 h-3.5 mr-1.5" />}
      {children}
    </span>
  );
}
