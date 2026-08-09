import { useState } from "react";
import { Link } from "react-router-dom";
import {
  FlaskConical,
  Search,
  BookOpen,
  MessageSquare,
  ChevronDown,
  ChevronRight,
  FileText,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { useRagInfo, useRagSearch, useRagRetrieve } from "@/hooks/use-rag";
import { PageHeader } from "@/components/shared/page-header";
import { Spinner } from "@/components/ui/spinner";
import { Badge } from "@/components/ui/badge";
import type { RagSearchResult, RagContextPart } from "@/api/rag";

type Mode = "search" | "retrieve";

export function RagTesterPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [mode, setMode] = useState<Mode>("search");
  const { data: ragInfo } = useRagInfo();
  const searchMutation = useRagSearch();
  const retrieveMutation = useRagRetrieve();

  const handleSubmit = () => {
    if (!query.trim()) return;
    if (mode === "search") {
      searchMutation.mutate({ query: query.trim(), topK });
    } else {
      retrieveMutation.mutate({ query: query.trim(), topK });
    }
  };

  const isPending = searchMutation.isPending || retrieveMutation.isPending;

  return (
    <div className="animate-page">
      <PageHeader
        title="RAG Tester"
        description="Test retrieval and context generation."
        actions={
          <Link
            to="/rag-tester/chat"
            className="inline-flex items-center gap-2 rounded-md bg-brand px-4 py-2 text-sm font-medium text-white hover:bg-brand-light transition-colors"
          >
            <MessageSquare size={16} />
            Open Chat
          </Link>
        }
      />

      {/* Info */}
      {ragInfo && (
        <div className="mb-6 border-l-2 border-slate-200 pl-5">
          <p className="text-sm font-medium text-slate-500 flex items-center gap-1.5">
            <FlaskConical size={14} className="text-slate-400" />
            Knowledge Store
          </p>
          <p className="text-3xl font-light text-brand">
            {ragInfo.doc_count}{" "}
            <span className="text-sm font-medium text-slate-500">
              documents loaded
            </span>
          </p>
        </div>
      )}

      {/* Query controls */}
      <div className="mb-6">
        {/* Mode toggle */}
        <div className="flex border-b border-slate-200 mb-4">
          <button
            onClick={() => setMode("search")}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors -mb-px ${
              mode === "search"
                ? "border-brand text-brand"
                : "border-transparent text-slate-500 hover:text-slate-700"
            }`}
          >
            <span className="flex items-center gap-1.5">
              <Search size={14} />
              BM25 Search
            </span>
          </button>
          <button
            onClick={() => setMode("retrieve")}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors -mb-px ${
              mode === "retrieve"
                ? "border-brand text-brand"
                : "border-transparent text-slate-500 hover:text-slate-700"
            }`}
          >
            <span className="flex items-center gap-1.5">
              <BookOpen size={14} />
              Context Retrieval
            </span>
          </button>
        </div>

        {/* Query input */}
        <div className="flex gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              placeholder={
                mode === "search"
                  ? "Search documents by keyword..."
                  : "Enter query for context retrieval..."
              }
              className="w-full pl-9 pr-4 py-2.5 border border-slate-200 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-brand/30 focus:border-brand-200 transition-all"
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs font-medium text-slate-500 whitespace-nowrap">
              Top K
            </label>
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-16 rounded-md border border-slate-200 px-2 py-2.5 text-sm text-center focus:outline-none focus:ring-1 focus:ring-brand/30"
            />
          </div>
          <button
            onClick={handleSubmit}
            disabled={isPending || !query.trim()}
            className="inline-flex items-center gap-2 rounded-md bg-brand px-5 py-2.5 text-sm font-medium text-white hover:bg-brand-light transition-colors disabled:opacity-50"
          >
            {isPending ? <Spinner size="sm" /> : <Search size={14} />}
            Run
          </button>
        </div>

        {/* Suggestion chips */}
        <div className="flex gap-2 mt-3 flex-wrap">
          {[
            "physics-informed neural network",
            "boundary conditions",
            "loss function",
            "Navier-Stokes",
          ].map((s) => (
            <button
              key={s}
              onClick={() => setQuery(s)}
              className="px-3 py-1 rounded-full border border-slate-200 text-xs text-slate-600 hover:bg-slate-50 hover:border-slate-300 transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Results */}
      {mode === "search" && searchMutation.data && (
        <SearchResults results={searchMutation.data} />
      )}
      {mode === "retrieve" && retrieveMutation.data && (
        <ContextResults parts={retrieveMutation.data} />
      )}
    </div>
  );
}

function SearchResults({ results }: { results: RagSearchResult[] }) {
  if (results.length === 0) {
    return (
      <p className="text-sm text-slate-400 py-8 text-center">
        No results found.
      </p>
    );
  }

  return (
    <div>
      <p className="text-sm text-slate-500 mb-4">
        {results.length} document{results.length !== 1 ? "s" : ""} matched
      </p>
      <table className="min-w-full text-left text-sm">
        <thead className="border-b border-slate-200 text-slate-500">
          <tr>
            <th className="font-medium py-3 px-2">Document</th>
            <th className="font-medium py-3 px-2">PDE Type</th>
            <th className="font-medium py-3 px-2">Sections</th>
            <th className="font-medium py-3 px-2 text-right">Nodes</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {results.map((r) => (
            <SearchResultRow key={r.doc_id} result={r} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SearchResultRow({ result }: { result: RagSearchResult }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <tr
        className="hover:bg-slate-50 transition-colors cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <td className="py-3.5 px-2">
          <div className="flex items-center gap-2">
            {expanded ? (
              <ChevronDown size={14} className="text-slate-400" />
            ) : (
              <ChevronRight size={14} className="text-slate-400" />
            )}
            <Link
              to={`/documents/${result.doc_id}`}
              onClick={(e) => e.stopPropagation()}
              className="font-medium text-slate-900 hover:text-brand transition-colors"
            >
              {result.doc_name}
            </Link>
          </div>
        </td>
        <td className="py-3.5 px-2">
          <Badge variant="brand">{result.pde_type || "—"}</Badge>
        </td>
        <td className="py-3.5 px-2 text-slate-500">
          {result.node_titles.length}
        </td>
        <td className="py-3.5 px-2 text-right text-slate-500">
          {result.node_count}
        </td>
      </tr>
      {expanded && result.node_titles.length > 0 && (
        <tr>
          <td colSpan={4} className="px-2 pb-3">
            <div className="ml-6 border-l-2 border-slate-100 pl-4 py-2">
              {result.node_titles.map((nt) => (
                <p
                  key={nt.node_id}
                  className="text-xs text-slate-600 py-0.5"
                  style={{ paddingLeft: `${nt.depth * 12}px` }}
                >
                  <FileText
                    size={10}
                    className="inline mr-1.5 text-slate-400"
                  />
                  {nt.title}
                </p>
              ))}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function ContextResults({ parts }: { parts: RagContextPart[] }) {
  if (parts.length === 0) {
    return (
      <p className="text-sm text-slate-400 py-8 text-center">
        No context retrieved.
      </p>
    );
  }

  return (
    <div>
      <p className="text-sm text-slate-500 mb-4">
        {parts.length} context part{parts.length !== 1 ? "s" : ""} retrieved
      </p>
      <div className="space-y-4">
        {parts.map((part, i) => (
          <ContextPartCard key={i} part={part} index={i} />
        ))}
      </div>
    </div>
  );
}

function ContextPartCard({
  part,
  index,
}: {
  part: RagContextPart;
  index: number;
}) {
  const [expanded, setExpanded] = useState(index === 0);

  return (
    <div className="border border-slate-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-slate-400">#{index + 1}</span>
          <span className="text-sm font-medium text-slate-800">
            {part.doc_name}
          </span>
          {part.pde_type && <Badge variant="brand">{part.pde_type}</Badge>}
        </div>
        {expanded ? (
          <ChevronDown size={14} className="text-slate-400" />
        ) : (
          <ChevronRight size={14} className="text-slate-400" />
        )}
      </button>
      {expanded && (
        <div className="px-4 pb-4 border-t border-slate-100">
          <div className="mt-3 prose-chat text-xs bg-slate-50 rounded-md p-4 max-h-80 overflow-y-auto">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex]}
            >
              {part.text}
            </ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}
