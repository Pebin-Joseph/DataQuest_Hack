import React from "react";
import { motion } from "framer-motion";
import { BookOpen, User, Bot, Hash } from "lucide-react";

const ChatMessage = ({ role, text, sources }) => {
  const isUser = role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, translateY: 12 }}
      animate={{ opacity: 1, translateY: 0 }}
      transition={{ duration: 0.18 }}
      className={`message ${isUser ? "user" : ""}`}
    >
      <div className="avatar">{isUser ? <User size={18} /> : <Bot size={18} />}</div>
      <div className="body">
        <div className="meta">{isUser ? "You" : "Live RAG"}</div>
        <div className="text">{text}</div>
        {!isUser && sources && sources.length > 0 && (
          <div className="sources">
            {sources.map((s, idx) => (
              <div key={idx} className="source-card">
                <div className="source-title">
                  <BookOpen size={14} /> {s.doc || "Unknown"}
                </div>
                <div className="source-meta">
                  <span>p.{s.page || "?"}</span>
                  {s.chunk_id && (
                    <span className="chunk-id">
                      <Hash size={12} /> {s.chunk_id.slice(0, 8)}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default ChatMessage;
