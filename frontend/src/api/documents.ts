import { apiFetch, apiUpload } from "./client";

export interface DocumentMeta {
  doc_id: string;
  doc_name: string;
  pde_type: string;
  techniques: string[];
  keywords: string[];
  known_issues: string[];
  node_count: number;
  total_tokens: number;
  indexed_at: string;
}

export interface TreeNode {
  node_id: string;
  title: string;
  text: string;
  level: number;
  summary: string;
  children: TreeNode[];
}

export interface DocumentDetail {
  doc: DocumentMeta;
  tree: {
    doc_name: string;
    root_nodes: TreeNode[];
  };
  collections: string[];
}

export interface DocumentsPage {
  documents: DocumentMeta[];
  page: number;
  total_pages: number;
  total: number;
}

export interface SearchResult {
  doc_id: string;
  doc_name: string;
  keywords: string[];
  context: string;
}

export interface UploadResult {
  doc_id: string;
  status: string;
  message: string;
}

export async function listDocuments(page = 1): Promise<DocumentsPage> {
  return apiFetch<DocumentsPage>(`/documents?page=${page}`);
}

export async function getDocument(id: string): Promise<DocumentDetail> {
  return apiFetch<DocumentDetail>(`/documents/${id}`);
}

export async function searchDocuments(q: string): Promise<SearchResult[]> {
  return apiFetch<SearchResult[]>(`/documents/search?q=${encodeURIComponent(q)}`);
}

export async function uploadDocument(
  file: File,
  collectionId?: string,
  hybridPdf?: boolean,
): Promise<UploadResult> {
  const fd = new FormData();
  fd.append("file", file);
  if (collectionId) fd.append("collection_id", collectionId);
  if (hybridPdf) fd.append("hybrid_pdf", "1");
  return apiUpload<UploadResult>("/documents/upload", fd);
}

export async function deleteDocument(id: string): Promise<void> {
  await apiFetch<{ ok: boolean }>(`/documents/${id}`, { method: "DELETE" });
}
