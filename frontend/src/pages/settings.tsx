import { useQuery } from "@tanstack/react-query";
import { Settings, Server, Database, FolderOpen, Users } from "lucide-react";
import { apiFetch } from "@/api/client";
import { PageHeader } from "@/components/shared/page-header";
import { Spinner } from "@/components/ui/spinner";
import { useRagInfo } from "@/hooks/use-rag";

interface SystemSettings {
  knowledge_store_dir: string;
  knowledge_sources_dir: string;
  registry_db: string;
  collections_db: string;
  users_db: string;
  host: string;
  port: string;
}

export function SettingsPage() {
  const { data: settings, isLoading } = useQuery({
    queryKey: ["settings"],
    queryFn: () => apiFetch<SystemSettings>("/settings"),
  });
  const { data: ragInfo } = useRagInfo();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Spinner size="lg" />
      </div>
    );
  }

  if (!settings) return null;

  return (
    <div className="animate-page">
      <PageHeader
        title="Settings"
        description="System configuration and status."
      />

      {/* System metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-8 mb-10">
        <div className="border-l-2 border-slate-200 pl-5">
          <p className="text-sm font-medium text-slate-500 flex items-center gap-1.5">
            <Server size={14} className="text-slate-400" />
            Server
          </p>
          <p className="text-2xl font-light text-brand mt-1">
            {settings.host}:{settings.port}
          </p>
        </div>
        <div className="border-l-2 border-slate-200 pl-5">
          <p className="text-sm font-medium text-slate-500 flex items-center gap-1.5">
            <Database size={14} className="text-slate-400" />
            Documents Loaded
          </p>
          <p className="text-2xl font-light text-brand mt-1">
            {ragInfo?.doc_count ?? "—"}
          </p>
        </div>
        <div className="border-l-2 border-emerald-200 pl-5">
          <p className="text-sm font-medium text-slate-500">Status</p>
          <p className="text-2xl font-light text-emerald-600 mt-1">Online</p>
        </div>
      </div>

      {/* Configuration table */}
      <h3 className="text-base font-medium text-brand mb-4 flex items-center gap-2">
        <Settings size={16} className="text-slate-400" />
        Configuration
      </h3>
      <table className="min-w-full text-left text-sm">
        <thead className="border-b border-slate-200 text-slate-500">
          <tr>
            <th className="font-medium py-3 px-2 w-48">Setting</th>
            <th className="font-medium py-3 px-2">Value</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          <ConfigRow
            icon={FolderOpen}
            label="Knowledge Store"
            value={settings.knowledge_store_dir}
          />
          <ConfigRow
            icon={FolderOpen}
            label="Sources Directory"
            value={settings.knowledge_sources_dir}
          />
          <ConfigRow
            icon={Database}
            label="Registry Database"
            value={settings.registry_db}
          />
          <ConfigRow
            icon={Database}
            label="Collections Database"
            value={settings.collections_db}
          />
          <ConfigRow
            icon={Users}
            label="Users Database"
            value={settings.users_db}
          />
          <ConfigRow
            icon={Server}
            label="Host"
            value={settings.host}
          />
          <ConfigRow
            icon={Server}
            label="Port"
            value={settings.port}
          />
        </tbody>
      </table>
    </div>
  );
}

function ConfigRow({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Server;
  label: string;
  value: string;
}) {
  return (
    <tr className="hover:bg-slate-50 transition-colors">
      <td className="py-3.5 px-2">
        <span className="flex items-center gap-2 text-slate-600">
          <Icon size={14} className="text-slate-400" />
          {label}
        </span>
      </td>
      <td className="py-3.5 px-2 font-mono text-xs text-slate-700">{value}</td>
    </tr>
  );
}
