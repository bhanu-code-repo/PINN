import { Navigate } from "react-router-dom";
import { useMe } from "@/hooks/use-auth";
import { Spinner } from "@/components/ui/spinner";
import { AppShell } from "./app-shell";

export function AuthGuard() {
  const { isLoading, isError } = useMe();

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Spinner size="lg" />
      </div>
    );
  }

  if (isError) {
    return <Navigate to="/login" replace />;
  }

  return <AppShell />;
}
