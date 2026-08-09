import { Outlet } from "react-router-dom";
import { clsx } from "clsx";
import { Sidebar } from "./sidebar";
import { Topbar } from "./topbar";
import { useUIStore } from "@/stores/ui-store";

export function AppShell() {
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);

  return (
    <div className="min-h-screen bg-slate-50 flex">
      <Sidebar />
      <div
        className={clsx(
          "flex-1 flex flex-col min-w-0 transition-all duration-200",
          sidebarOpen ? "ml-60" : "ml-16",
        )}
      >
        <Topbar />
        <main className="flex-1 px-6 py-6 sm:px-10 sm:py-8 lg:px-12 overflow-y-auto">
          <div className="max-w-6xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
