import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Library,
  FileText,
  Upload,
  FlaskConical,
  MessageSquare,
  Settings,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { clsx } from "clsx";
import { useUIStore } from "@/stores/ui-store";

const navItems = [
  { to: "/", icon: LayoutDashboard, label: "Overview" },
  { to: "/collections", icon: Library, label: "Collections" },
  { to: "/documents", icon: FileText, label: "Documents" },
  { to: "/upload", icon: Upload, label: "Upload" },
  { to: "/rag-tester", icon: FlaskConical, label: "RAG Tester" },
  { to: "/rag-tester/chat", icon: MessageSquare, label: "Chat" },
  { to: "/settings", icon: Settings, label: "Settings" },
];

export function Sidebar() {
  const open = useUIStore((s) => s.sidebarOpen);
  const toggle = useUIStore((s) => s.toggleSidebar);

  return (
    <aside
      className={clsx(
        "fixed left-0 top-0 z-40 h-screen bg-white border-r border-slate-200 transition-all duration-200 flex flex-col",
        open ? "w-60" : "w-16",
      )}
    >
      {/* Logo */}
      <div className="flex h-14 items-center border-b border-slate-200 px-4 shrink-0">
        <div className="flex items-center gap-2 overflow-hidden">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-brand text-white">
            <div className="w-2.5 h-2.5 bg-white rounded-full" />
          </div>
          {open && (
            <span className="text-lg font-semibold text-brand whitespace-nowrap tracking-tight">
              PINN Core
            </span>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 flex flex-col gap-0.5 px-2 py-4">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-all duration-200",
                isActive
                  ? "bg-brand/5 text-brand"
                  : "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
              )
            }
          >
            <Icon
              size={18}
              className={clsx("shrink-0")}
            />
            {open && <span className="truncate">{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle */}
      <div className="p-3 border-t border-slate-200 shrink-0">
        <button
          onClick={toggle}
          className={clsx(
            "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-700 transition-colors w-full",
          )}
        >
          {open ? (
            <>
              <ChevronLeft size={18} className="shrink-0" />
              <span>Collapse</span>
            </>
          ) : (
            <ChevronRight size={18} className="shrink-0" />
          )}
        </button>
      </div>
    </aside>
  );
}
