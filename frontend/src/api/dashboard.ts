import { apiFetch } from "./client";

export interface DashboardStats {
  collections: number;
  documents: number;
  files: number;
  restricted: number;
  recent_collections: Array<{
    id: string;
    name: string;
    access: string;
    doc_count: number;
  }>;
  recent_documents: Array<{
    doc_id: string;
    doc_name: string;
    node_count: number;
    total_tokens: number;
  }>;
}

export async function getStats(): Promise<DashboardStats> {
  return apiFetch<DashboardStats>("/dashboard/stats");
}
