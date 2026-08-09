import { Link } from "react-router-dom";
import { Library, FileText, HardDrive, ShieldCheck, ChevronRight } from "lucide-react";
import { useDashboardStats } from "@/hooks/use-dashboard";
import { PageHeader } from "@/components/shared/page-header";
import { Spinner } from "@/components/ui/spinner";
import { Badge } from "@/components/ui/badge";

export function DashboardPage() {
  const { data: stats, isLoading } = useDashboardStats();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Spinner size="lg" />
      </div>
    );
  }

  if (!stats) return null;

  return (
    <div className="animate-page">
      <PageHeader
        title="System Overview"
        description="High-level metrics across the PINN knowledge base."
      />

      {/* Typography-driven metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-10 mb-14 py-2">
        <MetricBlock
          icon={Library}
          label="Collections"
          value={stats.collections}
        />
        <MetricBlock
          icon={FileText}
          label="Documents"
          value={stats.documents}
        />
        <MetricBlock
          icon={HardDrive}
          label="Registered Files"
          value={stats.files}
        />
        <MetricBlock
          icon={ShieldCheck}
          label="Restricted"
          value={stats.restricted}
          accent
        />
      </div>

      {/* Recent tables side by side */}
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
        {/* Recent Collections */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-medium text-brand flex items-center gap-2">
              <Library size={16} className="text-slate-400" />
              Recent Collections
            </h3>
            <Link
              to="/collections"
              className="text-xs font-medium text-slate-500 hover:text-brand transition-colors flex items-center gap-1"
            >
              View all <ChevronRight size={12} />
            </Link>
          </div>
          {stats.recent_collections.length === 0 ? (
            <p className="py-8 text-center text-sm text-slate-400">
              No collections yet
            </p>
          ) : (
            <table className="min-w-full text-left text-sm">
              <thead className="border-b border-slate-200 text-slate-500">
                <tr>
                  <th className="font-medium py-3 px-2">Name</th>
                  <th className="font-medium py-3 px-2">Access</th>
                  <th className="font-medium py-3 px-2 text-right">Docs</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {stats.recent_collections.map((c) => (
                  <tr
                    key={c.id}
                    className="hover:bg-slate-50 transition-colors group"
                  >
                    <td className="py-3.5 px-2">
                      <Link
                        to={`/collections/${c.id}`}
                        className="font-medium text-slate-900 hover:text-brand transition-colors"
                      >
                        {c.name}
                      </Link>
                    </td>
                    <td className="py-3.5 px-2">
                      <Badge
                        variant={c.access === "public" ? "success" : "warning"}
                      >
                        {c.access}
                      </Badge>
                    </td>
                    <td className="py-3.5 px-2 text-right text-slate-500">
                      {c.doc_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {/* Recent Documents */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-medium text-brand flex items-center gap-2">
              <FileText size={16} className="text-slate-400" />
              Recent Documents
            </h3>
            <Link
              to="/documents"
              className="text-xs font-medium text-slate-500 hover:text-brand transition-colors flex items-center gap-1"
            >
              View all <ChevronRight size={12} />
            </Link>
          </div>
          {stats.recent_documents.length === 0 ? (
            <p className="py-8 text-center text-sm text-slate-400">
              No documents yet
            </p>
          ) : (
            <table className="min-w-full text-left text-sm">
              <thead className="border-b border-slate-200 text-slate-500">
                <tr>
                  <th className="font-medium py-3 px-2">Name</th>
                  <th className="font-medium py-3 px-2 text-right">Nodes</th>
                  <th className="font-medium py-3 px-2 text-right">Tokens</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {stats.recent_documents.map((d) => (
                  <tr
                    key={d.doc_id}
                    className="hover:bg-slate-50 transition-colors group"
                  >
                    <td className="py-3.5 px-2">
                      <Link
                        to={`/documents/${d.doc_id}`}
                        className="font-medium text-slate-900 hover:text-brand transition-colors"
                      >
                        {d.doc_name}
                      </Link>
                    </td>
                    <td className="py-3.5 px-2 text-right text-slate-500">
                      {d.node_count}
                    </td>
                    <td className="py-3.5 px-2 text-right text-slate-500">
                      {d.total_tokens.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>
    </div>
  );
}

function MetricBlock({
  icon: Icon,
  label,
  value,
  accent,
}: {
  icon: typeof Library;
  label: string;
  value: number | string;
  accent?: boolean;
}) {
  return (
    <div className="border-l-2 border-slate-200 pl-5">
      <p className="text-sm font-medium text-slate-500 mb-1 flex items-center gap-1.5">
        <Icon size={14} className="text-slate-400" />
        {label}
      </p>
      <div
        className={`text-4xl font-light ${accent ? "text-amber-600" : "text-brand"}`}
      >
        {typeof value === "number" ? value.toLocaleString() : value}
      </div>
    </div>
  );
}
