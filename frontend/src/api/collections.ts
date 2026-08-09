import { apiFetch } from "./client";

export interface Collection {
  id: string;
  name: string;
  description: string;
  access: string;
  allowed_groups: string[];
  doc_count: number;
  created_at: string;
  updated_at: string;
}

export interface CollectionDetail {
  collection: Collection;
  documents: Array<{ doc_id: string; doc_name: string }>;
  available_docs: Array<{ doc_id: string; doc_name: string }>;
}

export interface CreateCollectionRequest {
  name: string;
  description: string;
  access: string;
  allowed_groups: string;
}

export async function listCollections(): Promise<Collection[]> {
  return apiFetch<Collection[]>("/collections");
}

export async function getCollection(id: string): Promise<CollectionDetail> {
  return apiFetch<CollectionDetail>(`/collections/${id}`);
}

export async function createCollection(
  data: CreateCollectionRequest,
): Promise<Collection> {
  return apiFetch<Collection>("/collections", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updateCollection(
  id: string,
  data: CreateCollectionRequest,
): Promise<Collection> {
  return apiFetch<Collection>(`/collections/${id}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function deleteCollection(id: string): Promise<void> {
  await apiFetch<{ ok: boolean }>(`/collections/${id}`, { method: "DELETE" });
}

export async function addDocToCollection(
  collId: string,
  docId: string,
): Promise<void> {
  await apiFetch<{ ok: boolean }>(`/collections/${collId}/docs`, {
    method: "POST",
    body: JSON.stringify({ doc_id: docId }),
  });
}

export async function removeDocFromCollection(
  collId: string,
  docId: string,
): Promise<void> {
  await apiFetch<{ ok: boolean }>(`/collections/${collId}/docs/${docId}`, {
    method: "DELETE",
  });
}
