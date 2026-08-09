import { Outlet } from "react-router-dom";
import { clsx } from "clsx";
import { Sidebar } from "./sidebar";
import { Topbar } from "./topbar";
import { useUIStore } from "@/stores/ui-store";

export function AppShell() {
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);

  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar />
      <div
        className={clsx(
          "transition-all duration-200",
          sidebarOpen ? "ml-60" : "ml-16",
        )}
      >
        <Topbar />
        <main className="px-6 py-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
