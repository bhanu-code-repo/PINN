import { useQuery } from "@tanstack/react-query";
import { getStats } from "@/api/dashboard";

export function useDashboardStats() {
  return useQuery({
    queryKey: ["dashboard", "stats"],
    queryFn: getStats,
    staleTime: 30 * 1000,
  });
}
