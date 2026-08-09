import { Link } from "react-router-dom";
import { Library, FileText, HardDrive, ShieldCheck } from "lucide-react";
import { useDashboardStats } from "@/hooks/use-dashboard";
import { PageHeader } from "@/components/shared/page-header";
import { StatCard } from "@/components/shared/stat-card";
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
    <div>
      <PageHeader
        title="Dashboard"
        description="Knowledge base overview"
      />

      {/* Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-8">
        <StatCard icon={Library} label="Collections" value={stats.collections} />
        <StatCard icon={FileText} label="Documents" value={stats.documents} />
        <StatCard icon={HardDrive} label="Registered Files" value={stats.files} />
        <StatCard icon={ShieldCheck} label="Restricted" value={stats.restricted} />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Recent Collections */}
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-400">
            Recent Collections
          </h2>
          <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
            {stats.recent_collections.length === 0 ? (
              <p className="px-4 py-6 text-center text-sm text-slate-400">
                No collections yet
              </p>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-100 text-left">
                    <th className="px-4 py-2.5 font-medium text-slate-500">Name</th>
                    <th className="px-4 py-2.5 font-medium text-slate-500">Access</th>
                    <th className="px-4 py-2.5 font-medium text-slate-500 text-right">Docs</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.recent_collections.map((c) => (
                    <tr key={c.id} className="border-b border-slate-50 last:border-0 hover:bg-slate-50/50 transition-colors">
                      <td className="px-4 py-2.5">
                        <Link to={`/collections/${c.id}`} className="font-medium text-slate-800 hover:text-brand">
                          {c.name}
                        </Link>
                      </td>
                      <td className="px-4 py-2.5">
                        <Badge variant={c.access === "public" ? "success" : "warning"}>
                          {c.access}
                        </Badge>
                      </td>
                      <td className="px-4 py-2.5 text-right text-slate-500">{c.doc_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>

        {/* Recent Documents */}
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-400">
            Recent Documents
          </h2>
          <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
            {stats.recent_documents.length === 0 ? (
              <p className="px-4 py-6 text-center text-sm text-slate-400">
                No documents yet
              </p>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-100 text-left">
                    <th className="px-4 py-2.5 font-medium text-slate-500">Name</th>
                    <th className="px-4 py-2.5 font-medium text-slate-500 text-right">Nodes</th>
                    <th className="px-4 py-2.5 font-medium text-slate-500 text-right">Tokens</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.recent_documents.map((d) => (
                    <tr key={d.doc_id} className="border-b border-slate-50 last:border-0 hover:bg-slate-50/50 transition-colors">
                      <td className="px-4 py-2.5">
                        <Link to={`/documents/${d.doc_id}`} className="font-medium text-slate-800 hover:text-brand">
                          {d.doc_name}
                        </Link>
                      </td>
                      <td className="px-4 py-2.5 text-right text-slate-500">{d.node_count}</td>
                      <td className="px-4 py-2.5 text-right text-slate-500">{d.total_tokens.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
