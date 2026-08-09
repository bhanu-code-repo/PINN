import { useState, useEffect } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { FileText, Search, ChevronLeft, ChevronRight, Trash2 } from "lucide-react";
import { useDocuments, useSearchDocuments, useDeleteDocument } from "@/hooks/use-documents";
import { PageHeader } from "@/components/shared/page-header";
import { Spinner } from "@/components/ui/spinner";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/ui/empty-state";

export function DocumentsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialQuery = searchParams.get("q") ?? "";
  const [searchInput, setSearchInput] = useState(initialQuery);
  const [searchQuery, setSearchQuery] = useState(initialQuery);
  const [page, setPage] = useState(1);
  const deleteMutation = useDeleteDocument();

  // Sync URL param
  useEffect(() => {
    const q = searchParams.get("q") ?? "";
    if (q !== searchQuery) {
      setSearchInput(q);
      setSearchQuery(q);
    }
  }, [searchParams]);

  const isSearching = searchQuery.length > 0;
  const { data: docsPage, isLoading: docsLoading } = useDocuments(page);
  const { data: searchResults, isLoading: searchLoading } =
    useSearchDocuments(searchQuery);

  const handleSearch = () => {
    const q = searchInput.trim();
    setSearchQuery(q);
    if (q) {
      setSearchParams({ q });
    } else {
      setSearchParams({});
    }
  };

  const clearSearch = () => {
    setSearchInput("");
    setSearchQuery("");
    setSearchParams({});
  };

  const isLoading = isSearching ? searchLoading : docsLoading;

  return (
    <div className="animate-page">
      <PageHeader
        title="Documents"
        description="Browse and search indexed documents."
        actions={
          <Link
            to="/upload"
            className="inline-flex items-center gap-2 rounded-md bg-brand px-4 py-2 text-sm font-medium text-white hover:bg-brand-light transition-colors"
          >
            Upload
          </Link>
        }
      />

      {/* Search bar */}
      <div className="mb-6 flex gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            placeholder="Search by keyword, name, or content..."
            className="w-full pl-9 pr-4 py-2 border border-slate-200 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-brand/30 focus:border-brand-200 transition-all"
          />
        </div>
        <button
          onClick={handleSearch}
          className="rounded-md bg-slate-100 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-200 transition-colors"
        >
          Search
        </button>
        {isSearching && (
          <button
            onClick={clearSearch}
            className="rounded-md px-3 py-2 text-sm text-slate-500 hover:bg-slate-100 transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-16">
          <Spinner size="lg" />
        </div>
      ) : isSearching ? (
        // Search results
        <div>
          <p className="text-sm text-slate-500 mb-4">
            {searchResults?.length ?? 0} result
            {searchResults?.length !== 1 ? "s" : ""} for "{searchQuery}"
          </p>
          {!searchResults || searchResults.length === 0 ? (
            <EmptyState
              icon={Search}
              title="No results"
              description="Try a different search term."
            />
          ) : (
            <table className="min-w-full text-left text-sm">
              <thead className="border-b border-slate-200 text-slate-500">
                <tr>
                  <th className="font-medium py-3 px-2">Document</th>
                  <th className="font-medium py-3 px-2">Keywords</th>
                  <th className="font-medium py-3 px-2">Context</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {searchResults.map((r) => (
                  <tr
                    key={r.doc_id}
                    className="hover:bg-slate-50 transition-colors"
                  >
                    <td className="py-3.5 px-2">
                      <Link
                        to={`/documents/${r.doc_id}`}
                        className="font-medium text-slate-900 hover:text-brand transition-colors"
                      >
                        {r.doc_name}
                      </Link>
                    </td>
                    <td className="py-3.5 px-2">
                      <div className="flex gap-1 flex-wrap">
                        {r.keywords.slice(0, 3).map((k) => (
                          <Badge key={k}>{k}</Badge>
                        ))}
                      </div>
                    </td>
                    <td className="py-3.5 px-2 text-slate-500 text-xs max-w-xs truncate">
                      {r.context}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      ) : (
        // Paginated list
        <div>
          {!docsPage || docsPage.documents.length === 0 ? (
            <EmptyState
              icon={FileText}
              title="No documents"
              description="Upload documents to get started."
            />
          ) : (
            <>
              <div className="mb-3 text-sm text-slate-500">
                {docsPage.total} document{docsPage.total !== 1 ? "s" : ""} total
              </div>
              <table className="min-w-full text-left text-sm">
                <thead className="border-b border-slate-200 text-slate-500">
                  <tr>
                    <th className="font-medium py-3 px-2">Name</th>
                    <th className="font-medium py-3 px-2">PDE Type</th>
                    <th className="font-medium py-3 px-2">Keywords</th>
                    <th className="font-medium py-3 px-2 text-right">Nodes</th>
                    <th className="font-medium py-3 px-2 text-right">
                      Tokens
                    </th>
                    <th className="font-medium py-3 px-2 w-12" />
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {docsPage.documents.map((d) => (
                    <tr
                      key={d.doc_id}
                      className="hover:bg-slate-50 transition-colors group"
                    >
                      <td className="py-4 px-2">
                        <Link
                          to={`/documents/${d.doc_id}`}
                          className="font-medium text-slate-900 hover:text-brand transition-colors"
                        >
                          {d.doc_name}
                        </Link>
                      </td>
                      <td className="py-4 px-2">
                        <Badge variant="brand">{d.pde_type || "—"}</Badge>
                      </td>
                      <td className="py-4 px-2">
                        <div className="flex gap-1 flex-wrap">
                          {d.keywords.slice(0, 3).map((k) => (
                            <Badge key={k}>{k}</Badge>
                          ))}
                        </div>
                      </td>
                      <td className="py-4 px-2 text-right text-slate-500">
                        {d.node_count}
                      </td>
                      <td className="py-4 px-2 text-right text-slate-500">
                        {d.total_tokens.toLocaleString()}
                      </td>
                      <td className="py-4 px-2 text-right">
                        <button
                          onClick={() => {
                            if (confirm(`Delete "${d.doc_name}"?`))
                              deleteMutation.mutate(d.doc_id);
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

              {/* Pagination */}
              {docsPage.total_pages > 1 && (
                <div className="flex items-center justify-between mt-6 pt-4 border-t border-slate-100">
                  <p className="text-sm text-slate-500">
                    Page {docsPage.page} of {docsPage.total_pages}
                  </p>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                      disabled={page <= 1}
                      className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    >
                      <ChevronLeft size={14} />
                      Previous
                    </button>
                    <button
                      onClick={() =>
                        setPage((p) => Math.min(docsPage.total_pages, p + 1))
                      }
                      disabled={page >= docsPage.total_pages}
                      className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    >
                      Next
                      <ChevronRight size={14} />
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
