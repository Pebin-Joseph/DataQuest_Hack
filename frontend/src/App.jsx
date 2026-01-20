import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Sparkles,
  ShieldCheck,
  Cpu,
  Send,
  Loader2,
  Activity,
  Waves,
  Database,
  FolderOpen,
} from "lucide-react";
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
      <div className="bg" aria-hidden="true">
        <div className="bg-orb orb-1" />
        <div className="bg-orb orb-2" />
        <div className="bg-grid" />
      </div>

      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Waves size={18} />
          </div>
          <div className="brand-text">
            <div className="brand-title">Live RAG Console</div>
            <div className="brand-subtitle">Streaming ingestion • grounded answers • citations</div>
          </div>
        </div>

        <div className="topbar-right">
          <div className={`top-status ${health.status === "online" ? "ok" : "bad"}`}>
            <span className="dot" />
            {health.status === "online" ? "Online" : "Offline"}
          </div>
          <motion.div
            animate={{ rotate: [0, 4, -4, 0] }}
            transition={{ duration: 3.2, repeat: Infinity, ease: "easeInOut" }}
            className="top-badge"
            title="Live updates enabled"
          >
            <Sparkles size={14} /> Live
          </motion.div>
        </div>
      </header>

      <aside className="panel panel-left">
        <div className="panel-title">System</div>
        <div className="pill-grid">
          <StatusPill
            label={health.status === "online" ? "Backend online" : "Backend offline"}
            sub={health.status === "online" ? "Pathway + Chroma + Ollama" : "Start Docker/Ollama"}
            type={health.status === "online" ? "online" : "offline"}
          />
          <StatusPill label="Watching folder" sub={health.watching || "/data"} type="watch" />
          <StatusPill
            label="Indexed chunks"
            sub={health.count === null ? "--" : String(health.count)}
            type="metric"
          />
        </div>

        <div className="panel-section">
          <div className="mini-row">
            <FolderOpen size={16} />
            <div>
              <div className="mini-title">Live ingestion</div>
              <div className="mini-sub">Files added/updated/deleted are reflected automatically.</div>
            </div>
          </div>
          <div className="mini-row">
            <Database size={16} />
            <div>
              <div className="mini-title">Vector store</div>
              <div className="mini-sub">Chroma persistent collection with doc/page metadata.</div>
            </div>
          </div>
          <div className="mini-row">
            <ShieldCheck size={16} />
            <div>
              <div className="mini-title">Grounded answers</div>
              <div className="mini-sub">Responses are restricted to retrieved context.</div>
            </div>
          </div>
        </div>
      </aside>

      <main className="panel panel-main">
        <div className="chat-window" role="log" aria-live="polite">
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

        <div className="composer">
          <textarea
            placeholder="Ask about the latest docs… (Enter to send, Shift+Enter for newline)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask();
              }
            }}
          />
          <div className="composer-bar">
            <div className="control-row">
              <label htmlFor="k-slider">Retrieval depth</label>
              <div className="range-wrap">
                <input
                  id="k-slider"
                  type="range"
                  min="2"
                  max="12"
                  value={k}
                  onChange={(e) => setK(Number(e.target.value))}
                />
                <div className="range-value">k={k}</div>
              </div>
            </div>

            <button className="primary" onClick={ask} disabled={loading}>
              {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              Ask
            </button>

            <div className="composer-meta">
              <span className="inline-metric" title="Last request latency">
                <Activity size={14} /> {latencyMs ? `${latencyMs} ms` : "--"}
              </span>
              {lastError && <span className="inline-error">{lastError}</span>}
            </div>
          </div>
        </div>

        <div className="hint-row">
          <Cpu size={14} />
          <span>
            Tip: mention a doc name, or ask for a specific summary. Increase <b>k</b> for broader recall.
          </span>
        </div>
      </main>

      <aside className="panel panel-right">
        <div className="panel-title">Demo</div>
        <div className="chip">
          <Sparkles size={14} /> Live add/update/delete simulator supported
        </div>
        <div className="chip">
          <Database size={14} /> Persistent vectors (Docker volume)
        </div>
        <div className="chip">
          <ShieldCheck size={14} /> Source cards show doc + page
        </div>
        <div className="chip subtle">
          <Activity size={14} /> Try: “When is Product Alpha launching, and was there any correction?”
        </div>
      </aside>
    </div>
  );
};

export default App;
