import { useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  FileText,
  Plus,
  X,
  Pencil,
  Trash2,
} from "lucide-react";
import {
  useCollection,
  useAddDocToCollection,
  useRemoveDocFromCollection,
  useUpdateCollection,
  useDeleteCollection,
} from "@/hooks/use-collections";
import { Spinner } from "@/components/ui/spinner";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/ui/empty-state";

export function CollectionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data, isLoading } = useCollection(id ?? "");
  const addDoc = useAddDocToCollection(id ?? "");
  const removeDoc = useRemoveDocFromCollection(id ?? "");
  const updateMutation = useUpdateCollection(id ?? "");
  const deleteMutation = useDeleteCollection();
  const [editing, setEditing] = useState(false);
  const [editForm, setEditForm] = useState({
    name: "",
    description: "",
    access: "public",
    allowed_groups: "",
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Spinner size="lg" />
      </div>
    );
  }

  if (!data) return null;

  const { collection: coll, documents, available_docs } = data;

  const startEdit = () => {
    setEditForm({
      name: coll.name,
      description: coll.description,
      access: coll.access,
      allowed_groups: coll.allowed_groups?.join(", ") ?? "",
    });
    setEditing(true);
  };

  const handleUpdate = () => {
    updateMutation.mutate(editForm, {
      onSuccess: () => setEditing(false),
    });
  };

  const handleDelete = () => {
    if (confirm(`Delete "${coll.name}" permanently?`)) {
      deleteMutation.mutate(coll.id, {
        onSuccess: () => navigate("/collections"),
      });
    }
  };

  return (
    <div className="animate-page">
      {/* Back nav */}
      <Link
        to="/collections"
        className="inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-brand transition-colors mb-4"
      >
        <ArrowLeft size={14} />
        Collections
      </Link>

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between py-4 mb-6 border-b border-slate-200">
        <div>
          <h1 className="text-2xl font-semibold text-brand tracking-tight">
            {coll.name}
          </h1>
          <div className="mt-1.5 flex items-center gap-3 text-sm text-slate-500">
            <Badge variant={coll.access === "public" ? "success" : "warning"}>
              {coll.access}
            </Badge>
            {coll.description && <span>{coll.description}</span>}
          </div>
        </div>
        <div className="mt-3 sm:mt-0 flex items-center gap-2">
          <button
            onClick={startEdit}
            className="inline-flex items-center gap-1.5 rounded-md bg-slate-100 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-200 transition-colors"
          >
            <Pencil size={14} />
            Edit
          </button>
          <button
            onClick={handleDelete}
            className="inline-flex items-center gap-1.5 rounded-md bg-red-50 px-3 py-1.5 text-sm font-medium text-red-600 hover:bg-red-100 transition-colors border border-red-200"
          >
            <Trash2 size={14} />
            Delete
          </button>
        </div>
      </div>

      {/* Edit form */}
      {editing && (
        <div className="mb-6 border border-slate-200 rounded-lg p-5 bg-white">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-slate-600 mb-1">
                Name
              </label>
              <input
                value={editForm.name}
                onChange={(e) =>
                  setEditForm({ ...editForm, name: e.target.value })
                }
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand focus:outline-none focus:ring-1 focus:ring-brand"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-600 mb-1">
                Access
              </label>
              <select
                value={editForm.access}
                onChange={(e) =>
                  setEditForm({ ...editForm, access: e.target.value })
                }
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
              value={editForm.description}
              onChange={(e) =>
                setEditForm({ ...editForm, description: e.target.value })
              }
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand focus:outline-none focus:ring-1 focus:ring-brand"
            />
          </div>
          <div className="flex gap-3">
            <button
              onClick={handleUpdate}
              disabled={updateMutation.isPending}
              className="inline-flex items-center gap-2 rounded-md bg-brand px-4 py-2 text-sm font-medium text-white hover:bg-brand-light transition-colors disabled:opacity-50"
            >
              {updateMutation.isPending && <Spinner size="sm" />}
              Save
            </button>
            <button
              onClick={() => setEditing(false)}
              className="rounded-md bg-slate-100 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-200 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Metadata */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 mb-8">
        <div className="border-l-2 border-slate-200 pl-4">
          <p className="text-xs font-medium text-slate-500">Documents</p>
          <p className="text-2xl font-light text-brand">{documents.length}</p>
        </div>
        <div className="border-l-2 border-slate-200 pl-4">
          <p className="text-xs font-medium text-slate-500">Available</p>
          <p className="text-2xl font-light text-slate-700">
            {available_docs.length}
          </p>
        </div>
        <div className="border-l-2 border-slate-200 pl-4">
          <p className="text-xs font-medium text-slate-500">Created</p>
          <p className="text-sm font-medium text-slate-700 mt-1">
            {new Date(coll.created_at).toLocaleDateString()}
          </p>
        </div>
        <div className="border-l-2 border-slate-200 pl-4">
          <p className="text-xs font-medium text-slate-500">Updated</p>
          <p className="text-sm font-medium text-slate-700 mt-1">
            {new Date(coll.updated_at).toLocaleDateString()}
          </p>
        </div>
      </div>

      {/* Documents in collection */}
      <div className="mb-8">
        <h3 className="text-base font-medium text-brand mb-4">
          Documents in Collection
        </h3>
        {documents.length === 0 ? (
          <EmptyState
            icon={FileText}
            title="No documents"
            description="Add documents from the available list below."
          />
        ) : (
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-slate-200 text-slate-500">
              <tr>
                <th className="font-medium py-3 px-2">Document</th>
                <th className="font-medium py-3 px-2 text-right w-20">
                  Remove
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {documents.map((d) => (
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
                  <td className="py-3.5 px-2 text-right">
                    <button
                      onClick={() => removeDoc.mutate(d.doc_id)}
                      disabled={removeDoc.isPending}
                      className="p-1.5 rounded-md text-slate-400 hover:text-red-600 hover:bg-red-50 transition-colors opacity-0 group-hover:opacity-100"
                    >
                      <X size={14} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Available docs to add */}
      {available_docs.length > 0 && (
        <div>
          <h3 className="text-base font-medium text-slate-600 mb-4">
            Available Documents
          </h3>
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-slate-200 text-slate-500">
              <tr>
                <th className="font-medium py-3 px-2">Document</th>
                <th className="font-medium py-3 px-2 text-right w-20">Add</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {available_docs.map((d) => (
                <tr
                  key={d.doc_id}
                  className="hover:bg-slate-50 transition-colors group"
                >
                  <td className="py-3.5 px-2 text-slate-700">{d.doc_name}</td>
                  <td className="py-3.5 px-2 text-right">
                    <button
                      onClick={() => addDoc.mutate(d.doc_id)}
                      disabled={addDoc.isPending}
                      className="p-1.5 rounded-md text-slate-400 hover:text-emerald-600 hover:bg-emerald-50 transition-colors opacity-0 group-hover:opacity-100"
                    >
                      <Plus size={14} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
