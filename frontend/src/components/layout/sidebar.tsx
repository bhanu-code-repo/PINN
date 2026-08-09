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
  { to: "/", icon: LayoutDashboard, label: "Dashboard" },
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
        "fixed left-0 top-0 z-40 h-screen border-r border-slate-200 bg-white transition-all duration-200",
        open ? "w-60" : "w-16",
      )}
    >
      {/* Logo */}
      <div className="flex h-14 items-center border-b border-slate-200 px-4">
        <div className="flex items-center gap-2 overflow-hidden">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-brand text-white text-sm font-bold">
            P
          </div>
          {open && (
            <span className="text-sm font-semibold text-brand whitespace-nowrap">
              PINN Admin
            </span>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-0.5 px-2 py-3">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-brand-muted text-brand"
                  : "text-slate-600 hover:bg-slate-50 hover:text-slate-900",
              )
            }
          >
            <Icon size={18} className="shrink-0" />
            {open && <span className="truncate">{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={toggle}
        className="absolute bottom-4 left-0 flex w-full items-center justify-center px-2"
      >
        <div className="flex h-8 w-8 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-400 hover:text-slate-600 transition-colors">
          {open ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
        </div>
      </button>
    </aside>
  );
}
