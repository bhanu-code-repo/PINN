import { useState } from "react";
import { Link } from "react-router-dom";
import { Library, Plus, Trash2, ChevronRight } from "lucide-react";
import {
  useCollections,
  useCreateCollection,
  useDeleteCollection,
} from "@/hooks/use-collections";
import { PageHeader } from "@/components/shared/page-header";
import { Spinner } from "@/components/ui/spinner";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/ui/empty-state";

export function CollectionsPage() {
  const { data: collections, isLoading } = useCollections();
  const createMutation = useCreateCollection();
  const deleteMutation = useDeleteCollection();
  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState({
    name: "",
    description: "",
    access: "public",
    allowed_groups: "",
  });

  const handleCreate = () => {
    if (!form.name.trim()) return;
    createMutation.mutate(form, {
      onSuccess: () => {
        setShowCreate(false);
        setForm({ name: "", description: "", access: "public", allowed_groups: "" });
      },
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="animate-page">
      <PageHeader
        title="Collections"
        description="Organize documents into logical groups."
        actions={
          <button
            onClick={() => setShowCreate(true)}
            className="inline-flex items-center gap-2 rounded-md bg-brand px-4 py-2 text-sm font-medium text-white hover:bg-brand-light transition-colors"
          >
            <Plus size={16} />
            New Collection
          </button>
        }
      />

      {/* Create form */}
      {showCreate && (
        <div className="mb-6 border border-slate-200 rounded-lg p-5 bg-white">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">
            Create Collection
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-slate-600 mb-1">
                Name
              </label>
              <input
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand focus:outline-none focus:ring-1 focus:ring-brand"
                placeholder="Collection name"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-600 mb-1">
                Access
              </label>
              <select
                value={form.access}
                onChange={(e) => setForm({ ...form, access: e.target.value })}
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand focus:outline-none focus:ring-1 focus:ring-brand"
              >
                <option value="public">Public</option>
                <option value="restricted">Restricted</option>
              </select>
            </div>
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium text-slate-600 mb-1">
              Description
            </label>
            <input
              value={form.description}
              onChange={(e) =>
                setForm({ ...form, description: e.target.value })
              }
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand focus:outline-none focus:ring-1 focus:ring-brand"
              placeholder="Optional description"
            />
          </div>
          {form.access === "restricted" && (
            <div className="mb-4">
              <label className="block text-sm font-medium text-slate-600 mb-1">
                Allowed Groups
              </label>
              <input
                value={form.allowed_groups}
                onChange={(e) =>
                  setForm({ ...form, allowed_groups: e.target.value })
                }
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand focus:outline-none focus:ring-1 focus:ring-brand"
                placeholder="Comma-separated group names"
              />
            </div>
          )}
          <div className="flex gap-3">
            <button
              onClick={handleCreate}
              disabled={createMutation.isPending || !form.name.trim()}
              className="inline-flex items-center gap-2 rounded-md bg-brand px-4 py-2 text-sm font-medium text-white hover:bg-brand-light transition-colors disabled:opacity-50"
            >
              {createMutation.isPending && <Spinner size="sm" />}
              Create
            </button>
            <button
              onClick={() => setShowCreate(false)}
              className="rounded-md bg-slate-100 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-200 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Table */}
      {!collections || collections.length === 0 ? (
        <EmptyState
          icon={Library}
          title="No collections"
          description="Create your first collection to organize documents."
        />
      ) : (
        <table className="min-w-full text-left text-sm">
          <thead className="border-b border-slate-200 text-slate-500">
            <tr>
              <th className="font-medium py-3 px-2">Name</th>
              <th className="font-medium py-3 px-2">Description</th>
              <th className="font-medium py-3 px-2">Access</th>
              <th className="font-medium py-3 px-2 text-right">Documents</th>
              <th className="font-medium py-3 px-2 text-right">Updated</th>
              <th className="font-medium py-3 px-2 w-12" />
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {collections.map((c) => (
              <tr
                key={c.id}
                className="hover:bg-slate-50 transition-colors group"
              >
                <td className="py-4 px-2">
                  <Link
                    to={`/collections/${c.id}`}
                    className="font-medium text-slate-900 hover:text-brand transition-colors inline-flex items-center gap-1"
                  >
                    {c.name}
                    <ChevronRight
                      size={14}
                      className="opacity-0 group-hover:opacity-100 transition-opacity text-slate-400"
                    />
                  </Link>
                </td>
                <td className="py-4 px-2 text-slate-500 max-w-xs truncate">
                  {c.description || "—"}
                </td>
                <td className="py-4 px-2">
                  <Badge
                    variant={c.access === "public" ? "success" : "warning"}
                  >
                    {c.access}
                  </Badge>
                </td>
                <td className="py-4 px-2 text-right text-slate-700">
                  {c.doc_count}
                </td>
                <td className="py-4 px-2 text-right text-slate-500 text-xs">
                  {new Date(c.updated_at).toLocaleDateString()}
                </td>
                <td className="py-4 px-2 text-right">
                  <button
                    onClick={() => {
                      if (confirm(`Delete "${c.name}"?`))
                        deleteMutation.mutate(c.id);
                    }}
                    className="p-1.5 rounded-md text-slate-400 hover:text-red-600 hover:bg-red-50 transition-colors opacity-0 group-hover:opacity-100"
                  >
                    <Trash2 size={14} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
