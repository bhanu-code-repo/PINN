import { useMutation, useQuery } from "@tanstack/react-query";
import { getRagInfo, ragChat, ragRetrieve, ragSearch } from "@/api/rag";

export function useRagInfo() {
  return useQuery({
    queryKey: ["rag", "info"],
    queryFn: getRagInfo,
  });
}

export function useRagSearch() {
  return useMutation({
    mutationFn: ({ query, topK }: { query: string; topK: number }) =>
      ragSearch(query, topK),
  });
}

export function useRagRetrieve() {
  return useMutation({
    mutationFn: ({ query, topK }: { query: string; topK: number }) =>
      ragRetrieve(query, topK),
  });
}

export function useRagChat() {
  return useMutation({
    mutationFn: ({ query, topK }: { query: string; topK: number }) =>
      ragChat(query, topK),
  });
}
