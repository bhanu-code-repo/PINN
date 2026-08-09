import { createBrowserRouter } from "react-router-dom";
import { AuthGuard } from "@/components/layout/auth-guard";
import { LoginPage } from "@/pages/login";
import { DashboardPage } from "@/pages/dashboard";
import { CollectionsPage } from "@/pages/collections";
import { CollectionDetailPage } from "@/pages/collection-detail";
import { DocumentsPage } from "@/pages/documents";
import { DocumentDetailPage } from "@/pages/document-detail";
import { UploadPage } from "@/pages/upload";
import { RagTesterPage } from "@/pages/rag-tester";
import { RagChatPage } from "@/pages/rag-chat";
import { SettingsPage } from "@/pages/settings";
import { NotFoundPage } from "@/pages/not-found";

export const router = createBrowserRouter([
  {
    path: "/login",
    element: <LoginPage />,
  },
  {
    element: <AuthGuard />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: "collections", element: <CollectionsPage /> },
      { path: "collections/:id", element: <CollectionDetailPage /> },
      { path: "documents", element: <DocumentsPage /> },
      { path: "documents/:id", element: <DocumentDetailPage /> },
      { path: "upload", element: <UploadPage /> },
      { path: "rag-tester", element: <RagTesterPage /> },
      { path: "rag-tester/chat", element: <RagChatPage /> },
      { path: "settings", element: <SettingsPage /> },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
]);
