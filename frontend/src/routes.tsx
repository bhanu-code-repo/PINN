import { createBrowserRouter } from "react-router-dom";
import { AppShell } from "@/components/layout/app-shell";
import { LoginPage } from "@/pages/login";
import { DashboardPage } from "@/pages/dashboard";
import { NotFoundPage } from "@/pages/not-found";

// Placeholder pages — will be built out in later phases
function Placeholder({ name }: { name: string }) {
  return (
    <div className="py-10 text-center">
      <h2 className="text-lg font-semibold text-slate-600">{name}</h2>
      <p className="mt-1 text-sm text-slate-400">Coming soon</p>
    </div>
  );
}

export const router = createBrowserRouter([
  {
    path: "/login",
    element: <LoginPage />,
  },
  {
    element: <AppShell />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: "collections", element: <Placeholder name="Collections" /> },
      { path: "collections/:id", element: <Placeholder name="Collection Detail" /> },
      { path: "documents", element: <Placeholder name="Documents" /> },
      { path: "documents/:id", element: <Placeholder name="Document Detail" /> },
      { path: "upload", element: <Placeholder name="Upload" /> },
      { path: "rag-tester", element: <Placeholder name="RAG Tester" /> },
      { path: "rag-tester/chat", element: <Placeholder name="Chat" /> },
      { path: "settings", element: <Placeholder name="Settings" /> },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
]);
