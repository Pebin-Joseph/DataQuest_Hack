import React from "react";
import { motion } from "framer-motion";
import { Activity, FolderOpen, AlertTriangle, BarChart3 } from "lucide-react";

export const StatusPill = ({ label, sub, type = "online" }) => {
  const className =
    type === "online"
      ? "status"
      : type === "offline"
      ? "status offline"
      : type === "metric"
      ? "status metric"
      : "watch";

  const Icon =
    type === "watch"
      ? FolderOpen
      : type === "metric"
      ? BarChart3
      : type === "offline"
      ? AlertTriangle
      : Activity;
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      className={className}
    >
      <Icon size={16} />
      <div>
        <div>{label}</div>
        {sub && <div style={{ color: "#e2e8f0", fontSize: 12 }}>{sub}</div>}
      </div>
    </motion.div>
  );
};
