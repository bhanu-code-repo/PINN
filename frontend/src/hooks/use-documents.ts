import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  deleteDocument,
  getDocument,
  listDocuments,
  searchDocuments,
  uploadDocument,
} from "@/api/documents";

export function useDocuments(page = 1) {
  return useQuery({
    queryKey: ["documents", page],
    queryFn: () => listDocuments(page),
  });
}

export function useDocument(id: string) {
  return useQuery({
    queryKey: ["documents", "detail", id],
    queryFn: () => getDocument(id),
    enabled: !!id,
  });
}

export function useSearchDocuments(q: string) {
  return useQuery({
    queryKey: ["documents", "search", q],
    queryFn: () => searchDocuments(q),
    enabled: q.length > 0,
  });
}

export function useUploadDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      file,
      collectionId,
      hybridPdf,
    }: {
      file: File;
      collectionId?: string;
      hybridPdf?: boolean;
    }) => uploadDocument(file, collectionId, hybridPdf),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["documents"] }),
  });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["documents"] }),
  });
}
