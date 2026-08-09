import { apiFetch } from "./client";

export interface RagInfo {
  doc_count: number;
}

export interface RagSearchResult {
  doc_id: string;
  doc_name: string;
  pde_type: string;
  techniques: string[];
  keywords: string[];
  node_count: number;
  total_tokens: number;
  node_titles: Array<{ title: string; depth: number; node_id: string }>;
}

export interface RagContextPart {
  doc_id: string;
  doc_name: string;
  pde_type: string;
  text: string;
}

export interface RagChatResponse {
  query: string;
  context_parts: RagContextPart[];
  llm_response: string;
  llm_error: string;
}

export async function getRagInfo(): Promise<RagInfo> {
  return apiFetch<RagInfo>("/rag/info");
}

export async function ragSearch(
  query: string,
  topK = 5,
): Promise<RagSearchResult[]> {
  return apiFetch<RagSearchResult[]>("/rag/search", {
    method: "POST",
    body: JSON.stringify({ query, top_k: topK }),
  });
}

export async function ragRetrieve(
  query: string,
  topK = 5,
): Promise<RagContextPart[]> {
  return apiFetch<RagContextPart[]>("/rag/retrieve", {
    method: "POST",
    body: JSON.stringify({ query, top_k: topK }),
  });
}

export async function ragChat(
  query: string,
  topK = 5,
): Promise<RagChatResponse> {
  return apiFetch<RagChatResponse>("/rag/chat", {
    method: "POST",
    body: JSON.stringify({ query, top_k: topK }),
  });
}
