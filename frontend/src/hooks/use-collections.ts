import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  addDocToCollection,
  createCollection,
  deleteCollection,
  getCollection,
  listCollections,
  removeDocFromCollection,
  updateCollection,
  type CreateCollectionRequest,
} from "@/api/collections";

export function useCollections() {
  return useQuery({
    queryKey: ["collections"],
    queryFn: listCollections,
  });
}

export function useCollection(id: string) {
  return useQuery({
    queryKey: ["collections", id],
    queryFn: () => getCollection(id),
    enabled: !!id,
  });
}

export function useCreateCollection() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: CreateCollectionRequest) => createCollection(data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["collections"] }),
  });
}

export function useUpdateCollection(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: CreateCollectionRequest) => updateCollection(id, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["collections"] });
      qc.invalidateQueries({ queryKey: ["collections", id] });
    },
  });
}

export function useDeleteCollection() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: deleteCollection,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["collections"] }),
  });
}

export function useAddDocToCollection(collId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (docId: string) => addDocToCollection(collId, docId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["collections", collId] }),
  });
}

export function useRemoveDocFromCollection(collId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (docId: string) => removeDocFromCollection(collId, docId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["collections", collId] }),
  });
}
