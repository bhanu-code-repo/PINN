import { useParams, Link, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  FileText,
  ChevronDown,
  ChevronRight,
  Trash2,
  Library,
} from "lucide-react";
import { useState } from "react";
import { useDocument, useDeleteDocument } from "@/hooks/use-documents";
import { Spinner } from "@/components/ui/spinner";
import { Badge } from "@/components/ui/badge";
import type { TreeNode } from "@/api/documents";

export function DocumentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data, isLoading } = useDocument(id ?? "");
  const deleteMutation = useDeleteDocument();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Spinner size="lg" />
      </div>
    );
  }

  if (!data) return null;

  const { doc, tree, collections } = data;

  const handleDelete = () => {
    if (confirm(`Delete "${doc.doc_name}" permanently?`)) {
      deleteMutation.mutate(doc.doc_id, {
        onSuccess: () => navigate("/documents"),
      });
    }
  };

  return (
    <div className="animate-page">
      <Link
        to="/documents"
        className="inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-brand transition-colors mb-4"
      >
        <ArrowLeft size={14} />
        Documents
      </Link>

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between py-4 mb-6 border-b border-slate-200">
        <div>
          <h1 className="text-2xl font-semibold text-brand tracking-tight">
            {doc.doc_name}
          </h1>
          <div className="mt-1.5 flex items-center gap-2 flex-wrap">
            {doc.pde_type && <Badge variant="brand">{doc.pde_type}</Badge>}
            {doc.techniques.map((t) => (
              <Badge key={t}>{t}</Badge>
            ))}
          </div>
        </div>
        <button
          onClick={handleDelete}
          className="mt-3 sm:mt-0 inline-flex items-center gap-1.5 rounded-md bg-red-50 px-3 py-1.5 text-sm font-medium text-red-600 hover:bg-red-100 transition-colors border border-red-200"
        >
          <Trash2 size={14} />
          Delete
        </button>
      </div>

      {/* Metadata grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 mb-8">
        <div className="border-l-2 border-slate-200 pl-4">
          <p className="text-xs font-medium text-slate-500">Nodes</p>
          <p className="text-2xl font-light text-brand">{doc.node_count}</p>
        </div>
        <div className="border-l-2 border-slate-200 pl-4">
          <p className="text-xs font-medium text-slate-500">Tokens</p>
          <p className="text-2xl font-light text-brand">
            {doc.total_tokens.toLocaleString()}
          </p>
        </div>
        <div className="border-l-2 border-slate-200 pl-4">
          <p className="text-xs font-medium text-slate-500">Indexed</p>
          <p className="text-sm font-medium text-slate-700 mt-1">
            {doc.indexed_at
              ? new Date(doc.indexed_at).toLocaleDateString()
              : "—"}
          </p>
        </div>
        <div className="border-l-2 border-slate-200 pl-4">
          <p className="text-xs font-medium text-slate-500">Collections</p>
          <p className="text-2xl font-light text-slate-700">
            {collections.length}
          </p>
        </div>
      </div>

      {/* Keywords & known issues */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-8">
        {doc.keywords.length > 0 && (
          <div>
            <h3 className="text-sm font-medium text-slate-500 mb-2">
              Keywords
            </h3>
            <div className="flex gap-1.5 flex-wrap">
              {doc.keywords.map((k) => (
                <Badge key={k}>{k}</Badge>
              ))}
            </div>
          </div>
        )}
        {doc.known_issues.length > 0 && (
          <div>
            <h3 className="text-sm font-medium text-slate-500 mb-2">
              Known Issues
            </h3>
            <div className="flex gap-1.5 flex-wrap">
              {doc.known_issues.map((i) => (
                <Badge key={i} variant="warning">
                  {i}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Collections membership */}
      {collections.length > 0 && (
        <div className="mb-8">
          <h3 className="text-base font-medium text-brand mb-3 flex items-center gap-2">
            <Library size={16} className="text-slate-400" />
            Collections
          </h3>
          <div className="flex gap-2 flex-wrap">
            {collections.map((c) => (
              <Link
                key={c}
                to={`/collections/${c}`}
                className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:border-brand-200 hover:text-brand transition-colors"
              >
                <Library size={14} />
                {c}
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Document tree */}
      <div>
        <h3 className="text-base font-medium text-brand mb-4 flex items-center gap-2">
          <FileText size={16} className="text-slate-400" />
          Content Structure
        </h3>
        {tree.root_nodes.length === 0 ? (
          <p className="text-sm text-slate-400 py-6 text-center">
            No tree data available
          </p>
        ) : (
          <div className="border border-slate-200 rounded-lg divide-y divide-slate-100">
            {tree.root_nodes.map((node) => (
              <TreeNodeItem key={node.node_id} node={node} depth={0} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function TreeNodeItem({ node, depth }: { node: TreeNode; depth: number }) {
  const [expanded, setExpanded] = useState(depth < 1);
  const hasChildren = node.children && node.children.length > 0;

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left flex items-start gap-2 px-4 py-3 hover:bg-slate-50 transition-colors"
        style={{ paddingLeft: `${16 + depth * 20}px` }}
      >
        {hasChildren ? (
          expanded ? (
            <ChevronDown size={14} className="mt-0.5 text-slate-400 shrink-0" />
          ) : (
            <ChevronRight
              size={14}
              className="mt-0.5 text-slate-400 shrink-0"
            />
          )
        ) : (
          <span className="w-3.5 shrink-0" />
        )}
        <div className="min-w-0">
          <p className="text-sm font-medium text-slate-800">{node.title}</p>
          {node.summary && (
            <p className="text-xs text-slate-500 mt-0.5 line-clamp-2">
              {node.summary}
            </p>
          )}
        </div>
      </button>
      {expanded && hasChildren && (
        <div>
          {node.children.map((child) => (
            <TreeNodeItem
              key={child.node_id}
              node={child}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}
