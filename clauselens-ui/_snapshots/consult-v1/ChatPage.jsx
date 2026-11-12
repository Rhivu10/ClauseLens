import { useState } from "react";
import { sampleConversation } from "@/data/mockData";
import CitationChip from "@/components/CitationChip";

const SUGGESTIONS = [
  "What changed in the confidentiality clause?",
  "Which obligations survive termination in v4?",
  "Summarize every clause that mentions the Agent.",
];

export default function ChatPage() {
  const [messages, setMessages] = useState(sampleConversation);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);

  function send(text) {
    const q = text ?? input;
    if (!q.trim()) return;
    setMessages((m) => [...m, { role: "user", text: q }]);
    setInput("");
    setThinking(true);
    // Stand-in for hybrid_search() + generation against the real Node 2 index.
    setTimeout(() => {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: "This is a mock response — wire this screen up to the real Node 2 hybrid_search() output and a generation step to answer from the actual indexed clauses.",
          citations: [],
        },
      ]);
      setThinking(false);
    }, 700);
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 flex flex-col h-[calc(100vh-4rem)]">
      <div className="mb-6">
        <h1 className="font-serif text-3xl">Ask the index</h1>
        <p className="text-(--color-ink-soft) text-sm mt-1">
          Every answer is grounded in retrieved clauses — semantic, keyword and graph
          matches fused with Reciprocal Rank Fusion.
        </p>
      </div>

      <div className="flex-1 overflow-y-auto space-y-6 pr-1">
        {messages.map((m, i) => (
          <Message key={i} message={m} />
        ))}
        {thinking && (
          <div className="font-mono text-xs text-(--color-ink-faint) flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-(--color-brass) animate-pulse" />
            retrieving · fusing · drafting
          </div>
        )}
      </div>

      <div className="pt-4">
        {messages.length <= 1 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => send(s)}
                className="text-xs px-3 py-1.5 rounded-full border border-(--color-line) text-(--color-ink-soft) hover:border-(--color-brass) hover:text-(--color-brass-dim) transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
        )}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
          className="flex items-center gap-3 rounded-full border border-(--color-line) bg-(--color-surface) px-2 py-2"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about a clause, obligation, or change…"
            className="flex-1 bg-transparent outline-none px-3 font-serif-text text-[15px] placeholder:text-(--color-ink-faint)"
          />
          <button
            type="submit"
            className="w-9 h-9 rounded-full bg-(--color-brass) text-(--color-brass-ink) flex items-center justify-center hover:opacity-90"
            aria-label="Send"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M2 8h11m0 0L8 3m5 5l-5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
}

function Message({ message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[85%] ${isUser ? "items-end" : "items-start"} flex flex-col gap-2`}>
        <div
          className={`px-4 py-3 rounded-2xl font-serif-text text-[15px] leading-relaxed ${
            isUser
              ? "bg-(--color-surface-2) text-(--color-ink) rounded-br-sm"
              : "bg-(--color-surface) border border-(--color-line) rounded-bl-sm"
          }`}
        >
          {message.text}
        </div>
        {message.citations?.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {message.citations.map((c) => (
              <CitationChip key={c.chunk_id} citation={c} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
