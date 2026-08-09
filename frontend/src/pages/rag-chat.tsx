import { useState, useRef, useEffect } from "react";
import {
  Send,
  ArrowLeft,
  ChevronDown,
  ChevronRight,
  Bot,
  User,
  AlertCircle,
} from "lucide-react";
import { useRagChat } from "@/hooks/use-rag";
import { Spinner } from "@/components/ui/spinner";
import { Badge } from "@/components/ui/badge";
import { Link } from "react-router-dom";
import type { RagChatResponse } from "@/api/rag";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  context?: RagChatResponse;
  error?: string;
}

export function RagChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [topK, setTopK] = useState(5);
  const chatMutation = useRagChat();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, chatMutation.isPending]);

  const handleSend = () => {
    const q = input.trim();
    if (!q || chatMutation.isPending) return;

    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setInput("");

    chatMutation.mutate(
      { query: q, topK },
      {
        onSuccess: (data) => {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: data.llm_response || "",
              context: data,
              error: data.llm_error || undefined,
            },
          ]);
        },
        onError: () => {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: "",
              error: "Failed to get response. Please try again.",
            },
          ]);
        },
      },
    );
  };

  return (
    <div className="animate-page flex flex-col h-[calc(100vh-8rem)]">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-200 shrink-0">
        <div className="flex items-center gap-3">
          <Link
            to="/rag-tester"
            className="p-1.5 rounded-md text-slate-500 hover:bg-slate-100 transition-colors"
          >
            <ArrowLeft size={16} />
          </Link>
          <div>
            <h1 className="text-lg font-semibold text-brand">RAG Chat</h1>
            <p className="text-xs text-slate-500">
              Ask questions about the knowledge base
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-slate-500">Top K</label>
          <input
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-14 rounded-md border border-slate-200 px-2 py-1 text-sm text-center focus:outline-none focus:ring-1 focus:ring-brand/30"
          />
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto py-6 space-y-6">
        {messages.length === 0 && (
          <div className="text-center py-16">
            <Bot size={40} className="mx-auto mb-4 text-slate-300" />
            <p className="text-sm font-medium text-slate-500">
              Ask a question to get started
            </p>
            <p className="text-xs text-slate-400 mt-1">
              The assistant will retrieve relevant context and generate an
              answer
            </p>
            <div className="flex gap-2 justify-center mt-6 flex-wrap">
              {[
                "What are physics-informed neural networks?",
                "Explain boundary condition enforcement",
                "How does the loss function work in PINNs?",
              ].map((s) => (
                <button
                  key={s}
                  onClick={() => setInput(s)}
                  className="px-3 py-1.5 rounded-full border border-slate-200 text-xs text-slate-600 hover:bg-slate-50 hover:border-slate-300 transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}

        {chatMutation.isPending && (
          <div className="flex items-start gap-3">
            <div className="flex h-7 w-7 items-center justify-center rounded-full bg-brand-muted text-brand shrink-0">
              <Bot size={14} />
            </div>
            <div className="flex items-center gap-2 text-sm text-slate-500 pt-1">
              <Spinner size="sm" />
              Thinking...
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="shrink-0 border-t border-slate-200 pt-4">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Type your question..."
            className="flex-1 rounded-md border border-slate-200 px-4 py-2.5 text-sm focus:outline-none focus:ring-1 focus:ring-brand/30 focus:border-brand-200 transition-all"
            disabled={chatMutation.isPending}
          />
          <button
            onClick={handleSend}
            disabled={chatMutation.isPending || !input.trim()}
            className="inline-flex items-center gap-2 rounded-md bg-brand px-5 py-2.5 text-sm font-medium text-white hover:bg-brand-light transition-colors disabled:opacity-50"
          >
            <Send size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const [showContext, setShowContext] = useState(false);

  return (
    <div className={`flex items-start gap-3 ${isUser ? "" : ""}`}>
      <div
        className={`flex h-7 w-7 items-center justify-center rounded-full shrink-0 ${
          isUser
            ? "bg-slate-200 text-slate-600"
            : "bg-brand-muted text-brand"
        }`}
      >
        {isUser ? <User size={14} /> : <Bot size={14} />}
      </div>
      <div className="min-w-0 flex-1">
        {isUser ? (
          <p className="text-sm text-slate-800">{message.content}</p>
        ) : message.error && !message.content ? (
          <div className="flex items-center gap-2 text-sm text-red-600">
            <AlertCircle size={14} />
            {message.error}
          </div>
        ) : (
          <div>
            {/* Markdown response */}
            <div className="prose-chat text-sm max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex]}
              >
                {message.content}
              </ReactMarkdown>
            </div>

            {/* LLM error note */}
            {message.error && (
              <p className="mt-2 text-xs text-amber-600 flex items-center gap-1">
                <AlertCircle size={12} />
                {message.error}
              </p>
            )}

            {/* Context toggle */}
            {message.context &&
              message.context.context_parts.length > 0 && (
                <div className="mt-3">
                  <button
                    onClick={() => setShowContext(!showContext)}
                    className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-700 transition-colors"
                  >
                    {showContext ? (
                      <ChevronDown size={12} />
                    ) : (
                      <ChevronRight size={12} />
                    )}
                    {message.context.context_parts.length} source
                    {message.context.context_parts.length !== 1 ? "s" : ""} used
                  </button>
                  {showContext && (
                    <div className="mt-2 space-y-2">
                      {message.context.context_parts.map((cp, j) => (
                        <div
                          key={j}
                          className="border border-slate-200 rounded-md px-3 py-2"
                        >
                          <p className="text-xs font-medium text-slate-700 flex items-center gap-2">
                            {cp.doc_name}
                            {cp.pde_type && (
                              <Badge variant="brand">{cp.pde_type}</Badge>
                            )}
                          </p>
                          <pre className="mt-1.5 text-xs text-slate-600 whitespace-pre-wrap font-mono leading-relaxed max-h-40 overflow-y-auto">
                            {cp.text.substring(0, 500)}
                            {cp.text.length > 500 ? "..." : ""}
                          </pre>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
          </div>
        )}
      </div>
    </div>
  );
}
