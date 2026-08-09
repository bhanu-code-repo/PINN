import { LogOut, Search, ChevronRight, Menu } from "lucide-react";
import { useAuthStore } from "@/stores/auth-store";
import { useLogout } from "@/hooks/use-auth";
import { useNavigate, useLocation } from "react-router-dom";
import { useState, type FormEvent } from "react";
import { useUIStore } from "@/stores/ui-store";

const routeLabels: Record<string, string> = {
  "": "Dashboard",
  collections: "Collections",
  documents: "Documents",
  upload: "Upload",
  "rag-tester": "RAG Tester",
  settings: "Settings",
};

export function Topbar() {
  const user = useAuthStore((s) => s.user);
  const logoutMutation = useLogout();
  const navigate = useNavigate();
  const location = useLocation();
  const toggle = useUIStore((s) => s.toggleSidebar);
  const [searchQuery, setSearchQuery] = useState("");

  const handleLogout = () => {
    logoutMutation.mutate(undefined, {
      onSuccess: () => navigate("/login"),
    });
  };

  const handleSearch = (e: FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/documents?q=${encodeURIComponent(searchQuery.trim())}`);
      setSearchQuery("");
    }
  };

  // Build breadcrumb from path
  const segments = location.pathname.split("/").filter(Boolean);
  const firstSegment = segments[0] ?? "";
  const currentLabel =
    segments.length === 0
      ? "Dashboard"
      : routeLabels[firstSegment] ?? firstSegment;

  return (
    <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-slate-200 bg-white/95 backdrop-blur-sm px-6">
      <div className="flex items-center gap-4">
        <button
          onClick={toggle}
          className="p-1.5 rounded-md text-slate-500 hover:bg-slate-100 transition-colors"
        >
          <Menu size={18} />
        </button>
        <div className="hidden sm:flex items-center gap-2 text-sm text-slate-500">
          <span>PINN</span>
          <ChevronRight size={14} />
          <span className="font-medium text-brand">{currentLabel}</span>
          {segments.length > 1 && (
            <>
              <ChevronRight size={14} />
              <span className="font-medium text-slate-700 max-w-48 truncate">
                {segments.slice(1).join("/")}
              </span>
            </>
          )}
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Search */}
        <form onSubmit={handleSearch} className="relative hidden md:block">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search documents..."
            className="pl-9 pr-4 py-1.5 bg-slate-50 border border-slate-200 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-brand/30 focus:border-brand-200 focus:bg-white transition-all w-56"
          />
        </form>

        <div className="h-5 w-px bg-slate-200" />

        {/* User */}
        <div className="flex items-center gap-2 text-sm text-slate-600">
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-brand text-white text-xs font-medium">
            {user?.username?.charAt(0).toUpperCase() ?? "?"}
          </div>
          <span className="hidden sm:block font-medium">
            {user?.username ?? "Guest"}
          </span>
          {user?.is_admin && (
            <span className="rounded-full bg-brand-muted px-2 py-0.5 text-xs font-medium text-brand border border-brand-200">
              Admin
            </span>
          )}
        </div>

        <button
          onClick={handleLogout}
          className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm text-slate-500 hover:bg-slate-50 hover:text-slate-700 transition-colors"
        >
          <LogOut size={15} />
        </button>
      </div>
    </header>
  );
}
