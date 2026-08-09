import type { ReactNode } from "react";

interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: ReactNode;
}

export function PageHeader({ title, description, actions }: PageHeaderProps) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between py-6 mb-4 border-b border-slate-200">
      <div>
        <h1 className="text-2xl font-semibold text-brand tracking-tight">
          {title}
        </h1>
        {description && (
          <p className="mt-1 text-sm text-slate-500">{description}</p>
        )}
      </div>
      {actions && (
        <div className="mt-4 sm:mt-0 flex items-center gap-3">{actions}</div>
      )}
    </div>
  );
}
