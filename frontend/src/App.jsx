import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Sparkles, ShieldCheck, Cpu, Send, Loader2, Activity } from "lucide-react";
import ChatMessage from "./components/ChatMessage";
import { StatusPill } from "./components/StatusPill";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

const App = () => {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hi! Drop PDFs into /data and ask me anything. I will cite doc names and pages.",
      sources: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [k, setK] = useState(6);
  const [health, setHealth] = useState({ status: "checking", watching: "/data", count: null });
  const [latencyMs, setLatencyMs] = useState(null);
  const [lastError, setLastError] = useState("");

  const ask = async () => {
    const question = input.trim();
    if (!question) return;
    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setInput("");
    setLoading(true);
    try {
      const started = performance.now();
      const resp = await fetch(`${BACKEND_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, k }),
      });
      if (!resp.ok) {
        throw new Error(`Backend responded with ${resp.status}`);
      }
      const data = await resp.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.answer || "", sources: data.sources || [] },
      ]);
      setLatencyMs(Math.round(performance.now() - started));
      setLastError("");
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Backend unreachable. Check server & CORS.", sources: [] },
      ]);
      setLastError(err?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const ping = async () => {
      try {
        const resp = await fetch(`${BACKEND_URL}/health`);
        const data = await resp.json();
        setHealth({
          status: "online",
          watching: data?.watching || "/data",
          count: data?.collection_count ?? null,
        });
      } catch (e) {
        setHealth((prev) => ({ ...prev, status: "offline" }));
      }
    };
    ping();
    const id = setInterval(ping, 20000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="app-shell">
      <div className="sidebar">
        <div className="pill-grid">
          <StatusPill
            label={`System: ${health.status === "online" ? "Online" : "Offline"}`}
            sub={health.status === "online" ? "Pathway + Ollama" : "Check containers"}
            type={health.status === "online" ? "online" : "offline"}
          />
          <StatusPill
            label={`Watching: ${health.watching || "/data"}`}
            sub="Streaming ingest"
            type="watch"
          />
          <StatusPill
            label={`Chunks: ${health.count ?? "--"}`}
            sub="Chroma collection"
            type="metric"
          />
        </div>
        <div className="chip">
          <Cpu size={14} /> Pathway ingestion + Chroma + Ollama (llama3.2)
        </div>
        <div className="chip">
          <ShieldCheck size={14} /> Citations every answer
        </div>
        <div className="chip">
          <Sparkles size={14} /> SIH-grade motion & polish
        </div>
      </div>

      <div className="main">
        <div className="hero">
          <div>
            <h1>Live RAG Console</h1>
            <p>Real-time ingestion with Pathway, grounded answers with sources.</p>
          </div>
          <motion.div
            animate={{ rotate: [0, 6, -6, 0], scale: [1, 1.05, 1.05, 1] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
            className="badge"
          >
            <Sparkles size={14} /> Live
          </motion.div>
        </div>

        <div className="chat-window">
          {messages.map((m, idx) => (
            <ChatMessage key={idx} role={m.role} text={m.text} sources={m.sources} />
          ))}
          {loading && (
            <div className="message">
              <div className="avatar">
                <Loader2 className="spin" size={18} />
              </div>
              <div className="body">
                <div className="meta">Live RAG</div>
                <div className="text">Thinking with context…</div>
              </div>
            </div>
          )}
        </div>

        <div className="form">
          <textarea
            placeholder="Ask anything about the docs..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask();
              }
            }}
          />
          <div className="controls">
            <div className="control-row">
              <label htmlFor="k-slider">Depth (k): {k}</label>
              <input
                id="k-slider"
                type="range"
                min="2"
                max="12"
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
              />
            </div>
            <button onClick={ask} disabled={loading}>
              {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              &nbsp;Ask
            </button>
            <div className="form-meta">
              <span className="inline-metric">
                <Activity size={14} /> {latencyMs ? `${latencyMs} ms` : "--"}
              </span>
              {lastError && <span className="inline-error">{lastError}</span>}
            </div>
          </div>
        </div>

        <div className="hint-row">
          <span>Tips: ask specific intents ("summarize section X"), mention doc names, or increase depth.</span>
        </div>
      </div>
    </div>
  );
};

export default App;
