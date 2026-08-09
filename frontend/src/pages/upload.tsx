import { useState, useRef, type DragEvent } from "react";
import { Upload, FileUp, CheckCircle, AlertCircle, X } from "lucide-react";
import { useUploadDocument } from "@/hooks/use-documents";
import { useCollections } from "@/hooks/use-collections";
import { PageHeader } from "@/components/shared/page-header";
import { Spinner } from "@/components/ui/spinner";

export function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [collectionId, setCollectionId] = useState("");
  const [hybridPdf, setHybridPdf] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const uploadMutation = useUploadDocument();
  const { data: collections } = useCollections();

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  };

  const handleUpload = () => {
    if (!file) return;
    uploadMutation.mutate(
      { file, collectionId: collectionId || undefined, hybridPdf },
      {
        onSuccess: () => {
          setFile(null);
          setCollectionId("");
          setHybridPdf(false);
        },
      },
    );
  };

  return (
    <div className="animate-page">
      <PageHeader
        title="Upload Document"
        description="Add new documents to the knowledge base."
      />

      <div className="max-w-xl">
        {/* Drop zone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors ${
            dragging
              ? "border-brand bg-brand-50"
              : "border-slate-200 hover:border-slate-300 hover:bg-slate-50"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept=".pdf,.txt,.md"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) setFile(f);
            }}
          />
          <FileUp
            size={32}
            className={`mx-auto mb-3 ${dragging ? "text-brand" : "text-slate-400"}`}
          />
          <p className="text-sm font-medium text-slate-600">
            Drop a file here or click to browse
          </p>
          <p className="text-xs text-slate-400 mt-1">
            PDF, TXT, or Markdown files
          </p>
        </div>

        {/* Selected file */}
        {file && (
          <div className="mt-4 flex items-center justify-between border border-slate-200 rounded-md px-4 py-3">
            <div className="flex items-center gap-3 min-w-0">
              <Upload size={16} className="text-brand shrink-0" />
              <div className="min-w-0">
                <p className="text-sm font-medium text-slate-800 truncate">
                  {file.name}
                </p>
                <p className="text-xs text-slate-500">
                  {(file.size / 1024).toFixed(1)} KB
                </p>
              </div>
            </div>
            <button
              onClick={() => setFile(null)}
              className="p-1 text-slate-400 hover:text-slate-600 transition-colors"
            >
              <X size={14} />
            </button>
          </div>
        )}

        {/* Options */}
        <div className="mt-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-600 mb-1">
              Add to Collection (optional)
            </label>
            <select
              value={collectionId}
              onChange={(e) => setCollectionId(e.target.value)}
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand focus:outline-none focus:ring-1 focus:ring-brand"
            >
              <option value="">None</option>
              {collections?.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>

          {file?.name.endsWith(".pdf") && (
            <label className="flex items-center gap-2 text-sm text-slate-600 cursor-pointer">
              <input
                type="checkbox"
                checked={hybridPdf}
                onChange={(e) => setHybridPdf(e.target.checked)}
                className="rounded border-slate-300 text-brand focus:ring-brand"
              />
              Hybrid PDF extraction
            </label>
          )}
        </div>

        {/* Upload button */}
        <button
          onClick={handleUpload}
          disabled={!file || uploadMutation.isPending}
          className="mt-6 w-full inline-flex items-center justify-center gap-2 rounded-md bg-brand px-4 py-2.5 text-sm font-medium text-white hover:bg-brand-light transition-colors disabled:opacity-50"
        >
          {uploadMutation.isPending ? (
            <>
              <Spinner size="sm" />
              Uploading...
            </>
          ) : (
            <>
              <Upload size={16} />
              Upload Document
            </>
          )}
        </button>

        {/* Result feedback */}
        {uploadMutation.isSuccess && (
          <div className="mt-4 flex items-center gap-2 text-sm text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-md px-4 py-3">
            <CheckCircle size={16} />
            {uploadMutation.data.message || "Document uploaded successfully."}
          </div>
        )}
        {uploadMutation.isError && (
          <div className="mt-4 flex items-center gap-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded-md px-4 py-3">
            <AlertCircle size={16} />
            Upload failed. The server may not support this yet.
          </div>
        )}
      </div>
    </div>
  );
}
