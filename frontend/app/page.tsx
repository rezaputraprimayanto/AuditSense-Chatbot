"use client";

import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import dynamic from "next/dynamic";
import { Plus_Jakarta_Sans } from "next/font/google";
import styles from "./page.module.css";

const DocumentViewer = dynamic(() => import("../components/DocumentViewer"), { ssr: false });

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-jakarta",
});

const API_BASE = "http://127.0.0.1:8000";
const MAX_UPLOAD_MB = 200;

type Citation = {
  doc_name: string;
  doc_path: string;
  page: number;
  kind: string;
  chunk_id?: string | null;
  snippet?: string | null;

  bbox?: [number, number, number, number] | null;
  paragraph_bboxes?: [number, number, number, number][] | null;
  page_w?: number | null;
  page_h?: number | null;

  anchor?: string | null;
  para_start?: number | null;
  para_end?: number | null;
  table_index?: number | null;

  score?: number | null;
};

type SessionInfo = {
  session_id: string;
  title?: string;
  created_at?: number;
  updated_at?: number;
  num_messages?: number;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  ts?: number;
  citations?: Citation[];
};

type ToastKind = "info" | "success" | "warning" | "error";
type ToastItem = {
  id: string;
  text: string;
  kind: ToastKind;
};

type ModalKey = "upload" | "index" | "scope" | "history" | null;

type DocMeta = {
  name: string;
  firstSeenAt: number;
};

type IndexStatus = {
  docs: string[];
  indexed: string[];
  missing: string[];
};

type ProcessKind = "upload" | "index" | "reindex" | "success" | "error";

type ProcessCardState = {
  open: boolean;
  kind: ProcessKind;
  title: string;
  status: "idle" | "running" | "success" | "error";
  stage: string;
  message: string;
  progress: number;
  currentDocument?: string | null;
  currentPage?: number | null;
  totalPages?: number | null;
  processedDocs?: number | null;
  totalDocs?: number | null;
  processedChunks?: number | null;
  totalChunks?: number | null;
  processedBatches?: number | null;
  totalBatches?: number | null;
  extractionMode?: string | null;
  startedAt?: number;
};

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function extOf(name: string) {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function citeLabel(c: Citation) {
  const ext = extOf(c.doc_name);
  if (ext === ".pdf") return `Page ${c.page}`;
  if (ext === ".docx") {
    if (c.kind === "docx_table" && c.table_index) return `Table ${c.table_index}`;
    if (c.para_start && c.para_end && c.para_end !== c.para_start) {
      return `Paragraph ${c.para_start}-${c.para_end}`;
    }
    if (c.para_start) return `Paragraph ${c.para_start}`;
    return "DOCX";
  }
  return "Source";
}

function fmtElapsed(ms?: number) {
  if (!ms || ms <= 0) return "0:00";
  const sec = Math.floor(ms / 1000);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function humanStage(stage?: string) {
  switch ((stage || "").toLowerCase()) {
    case "queued":
      return "Queued";
    case "preparing":
      return "Preparing";
    case "skipping":
      return "Skipping";
    case "chunking":
      return "Extracting";
    case "embedding":
      return "Embedding";
    case "writing_chunks":
      return "Writing chunks";
    case "storing":
      return "Writing vectors";
    case "finalizing":
      return "Finalizing";
    case "removing_old_index":
      return "Removing old index";
    case "done":
      return "Completed";
    case "error":
      return "Failed";
    default:
      return "Processing";
  }
}

function ToastHost({ items, onRemove }: { items: ToastItem[]; onRemove: (id: string) => void }) {
  return (
    <div className={styles.toastStack} aria-live="polite" aria-relevant="additions">
      {items.map((t) => (
        <div key={t.id} className={styles.toast} role="status">
          <div className={styles.toastText}>
            <span className={styles.toastLabel}>
              {t.kind === "success" ? "Success:" : t.kind === "error" ? "Error:" : "Info:"}
            </span>
            <span className={styles.toastStrong}> {t.text}</span>
          </div>

          <button className={styles.toastClose} onClick={() => onRemove(t.id)} aria-label="Close" type="button">
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}

function Modal({
  open,
  title,
  onClose,
  children,
}: {
  open: boolean;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return createPortal(
    <>
      <div className={styles.modalOverlay} onMouseDown={onClose} />
      <div className={styles.modalCard} role="dialog" aria-modal="true">
        <div className={styles.modalHeader}>
          <div className={styles.modalTitle}>{title}</div>
          <button className={styles.modalCloseBtn} onClick={onClose} type="button" aria-label="Close">
            ✕
          </button>
        </div>
        <div className={styles.modalBody}>{children}</div>
      </div>
    </>,
    document.body
  );
}

function ProcessCardOverlay({
  state,
  onClose,
}: {
  state: ProcessCardState;
  onClose: () => void;
}) {
  const [mounted, setMounted] = useState(false);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!state.open || !state.startedAt) return;
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [state.open, state.startedAt]);

  const palette =
    state.status === "error"
      ? {
          accent: "#D92D20",
          soft: "rgba(217,45,32,0.10)",
          border: "rgba(217,45,32,0.18)",
          text: "#D92D20",
        }
      : state.status === "success"
      ? {
          accent: "#079455",
          soft: "rgba(7,148,85,0.10)",
          border: "rgba(7,148,85,0.18)",
          text: "#079455",
        }
      : {
          accent: "#1165FF",
          soft: "rgba(17,101,255,0.10)",
          border: "rgba(17,101,255,0.18)",
          text: "#1165FF",
        };

  const showClose = state.status === "success" || state.status === "error";
  const elapsed = state.startedAt ? fmtElapsed(now - state.startedAt) : "0:00";

  if (!state.open || !mounted) return null;

  return createPortal(
    <div
      aria-live="polite"
      aria-busy={state.status === "running"}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 20,
        background: "rgba(15,23,42,0.18)",
        backdropFilter: "blur(8px)",
        WebkitBackdropFilter: "blur(8px)",
      }}
    >
      <div
        style={{
          width: "min(460px, calc(100vw - 32px))",
          background: "rgba(255,255,255,0.98)",
          border: "1px solid rgba(20,20,20,0.08)",
          borderRadius: 14,
          boxShadow: "0 18px 44px rgba(0,0,0,0.12)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            padding: "16px 16px 14px 16px",
            borderBottom: "1px solid rgba(20,20,20,0.06)",
            display: "flex",
            alignItems: "flex-start",
            gap: 12,
          }}
        >
          <div
            style={{
              width: 40,
              height: 40,
              minWidth: 40,
              borderRadius: 10,
              display: "grid",
              placeItems: "center",
              background: palette.soft,
              border: `1px solid ${palette.border}`,
              color: palette.accent,
              fontWeight: 900,
              fontSize: 18,
            }}
          >
            {state.status === "error" ? "!" : state.status === "success" ? "✓" : "•"}
          </div>

          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ fontSize: 17, fontWeight: 900, color: "rgba(20,20,20,0.96)", lineHeight: 1.2 }}>{state.title}</div>
            <div style={{ marginTop: 4, fontSize: 12, fontWeight: 800, color: palette.text }}>{humanStage(state.stage)}</div>
          </div>

          {showClose ? (
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              style={{
                border: "1px solid rgba(20,20,20,0.08)",
                background: "#fff",
                color: "rgba(20,20,20,0.72)",
                borderRadius: 10,
                width: 32,
                height: 32,
                cursor: "pointer",
                fontWeight: 900,
              }}
            >
              ✕
            </button>
          ) : null}
        </div>

        <div style={{ padding: 16, display: "grid", gap: 14 }}>
          <div
            style={{
              fontSize: 13,
              lineHeight: 1.6,
              fontWeight: 600,
              color: "rgba(20,20,20,0.70)",
            }}
          >
            {state.message || "Processing..."}
          </div>

          <div
            style={{
              height: 10,
              borderRadius: 999,
              background: "rgba(20,20,20,0.08)",
              overflow: "hidden",
              border: "1px solid rgba(20,20,20,0.04)",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${clamp(state.progress || 0, 0, 100)}%`,
                background:
                  state.status === "error"
                    ? "linear-gradient(90deg, #F97066 0%, #D92D20 100%)"
                    : state.status === "success"
                    ? "linear-gradient(90deg, #12B76A 0%, #079455 100%)"
                    : "linear-gradient(90deg, #27B3E6 0%, #1165FF 100%)",
                transition: "width 260ms ease",
              }}
            />
          </div>

          <div style={{ display: "grid", gap: 8 }}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 8,
              }}
            >
              <div
                style={{
                  border: "1px solid rgba(20,20,20,0.06)",
                  borderRadius: 12,
                  padding: "10px 12px",
                  background: "#fff",
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 800, color: "rgba(20,20,20,0.48)", marginBottom: 4 }}>Progress</div>
                <div style={{ fontSize: 15, fontWeight: 900, color: "rgba(20,20,20,0.96)" }}>{clamp(state.progress || 0, 0, 100)}%</div>
              </div>

              <div
                style={{
                  border: "1px solid rgba(20,20,20,0.06)",
                  borderRadius: 12,
                  padding: "10px 12px",
                  background: "#fff",
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 800, color: "rgba(20,20,20,0.48)", marginBottom: 4 }}>Elapsed</div>
                <div style={{ fontSize: 15, fontWeight: 900, color: "rgba(20,20,20,0.96)" }}>{elapsed}</div>
              </div>
            </div>

            {state.currentDocument ? (
              <div
                style={{
                  border: "1px solid rgba(20,20,20,0.06)",
                  borderRadius: 12,
                  padding: "10px 12px",
                  background: "#fff",
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 800, color: "rgba(20,20,20,0.48)", marginBottom: 4 }}>Current document</div>
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 800,
                    color: "rgba(20,20,20,0.92)",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                  title={state.currentDocument}
                >
                  {state.currentDocument}
                </div>
              </div>
            ) : null}

            {(state.currentPage || state.totalPages) ? (
              <div
                style={{
                  border: "1px solid rgba(20,20,20,0.06)",
                  borderRadius: 12,
                  padding: "10px 12px",
                  background: "#fff",
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 800, color: "rgba(20,20,20,0.48)", marginBottom: 4 }}>Pages</div>
                <div style={{ fontSize: 13, fontWeight: 800, color: "rgba(20,20,20,0.92)" }}>
                  {state.currentPage ?? 0} / {state.totalPages ?? 0}
                  {state.extractionMode ? ` • ${String(state.extractionMode).toUpperCase()}` : ""}
                </div>
              </div>
            ) : null}

            {(state.processedDocs != null || state.totalDocs != null) ? (
              <div
                style={{
                  border: "1px solid rgba(20,20,20,0.06)",
                  borderRadius: 12,
                  padding: "10px 12px",
                  background: "#fff",
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 800, color: "rgba(20,20,20,0.48)", marginBottom: 4 }}>Documents</div>
                <div style={{ fontSize: 13, fontWeight: 800, color: "rgba(20,20,20,0.92)" }}>
                  {state.processedDocs ?? 0} / {state.totalDocs ?? 0}
                </div>
              </div>
            ) : null}

            {(state.processedChunks != null || state.totalChunks != null) ? (
              <div
                style={{
                  border: "1px solid rgba(20,20,20,0.06)",
                  borderRadius: 12,
                  padding: "10px 12px",
                  background: "#fff",
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 800, color: "rgba(20,20,20,0.48)", marginBottom: 4 }}>Chunks</div>
                <div style={{ fontSize: 13, fontWeight: 800, color: "rgba(20,20,20,0.92)" }}>
                  {state.processedChunks ?? 0} / {state.totalChunks ?? 0}
                </div>
              </div>
            ) : null}

            {(state.processedBatches != null || state.totalBatches != null) ? (
              <div
                style={{
                  border: "1px solid rgba(20,20,20,0.06)",
                  borderRadius: 12,
                  padding: "10px 12px",
                  background: "#fff",
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 800, color: "rgba(20,20,20,0.48)", marginBottom: 4 }}>Batches</div>
                <div style={{ fontSize: 13, fontWeight: 800, color: "rgba(20,20,20,0.92)" }}>
                  {state.processedBatches ?? 0} / {state.totalBatches ?? 0}
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}

function SideBtn({ title, iconSrc, onClick }: { title: string; iconSrc: string; onClick: () => void }) {
  return (
    <button className={styles.sidebarIconBtn} title={title} onClick={onClick} type="button">
      <img src={iconSrc} alt={title} style={{ width: 22, height: 22, display: "block", objectFit: "contain" }} />
    </button>
  );
}

export default function PageTsx() {
  useEffect(() => {
    document.documentElement.classList.add(jakarta.variable);
  }, []);

  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const toastTimers = useRef<Map<string, number>>(new Map());

  const pushToast = useCallback((text: string, kind: ToastKind = "info") => {
    const id = uid();
    setToasts((prev) => [{ id, text, kind }, ...prev].slice(0, 4));

    const timer = window.setTimeout(() => {
      setToasts((prev) => prev.filter((x) => x.id !== id));
      toastTimers.current.delete(id);
    }, 6000);

    toastTimers.current.set(id, timer);
  }, []);

  const removeToast = useCallback((id: string) => {
    const t = toastTimers.current.get(id);
    if (t) window.clearTimeout(t);
    toastTimers.current.delete(id);
    setToasts((prev) => prev.filter((x) => x.id !== id));
  }, []);

  const [docs, setDocs] = useState<string[]>([]);
  const [selectedPdf, setSelectedPdf] = useState<string>("Semua dokumen");
  const [viewerDoc, setViewerDoc] = useState<string | null>(null);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [citationJumpKey, setCitationJumpKey] = useState<string>("");

  const [docMeta, setDocMeta] = useState<Record<string, DocMeta>>({});
  const DOC_META_KEY = "pdh_doc_meta_v1";

  const [indexStatus, setIndexStatus] = useState<IndexStatus>({
    docs: [],
    indexed: [],
    missing: [],
  });
  const [indexStatusLoading, setIndexStatusLoading] = useState(false);

  const [input, setInput] = useState<string>("");
  const promptRef = useRef<HTMLTextAreaElement | null>(null);
  const [chat, setChat] = useState<ChatMessage[]>([]);

  const [sessionId, setSessionId] = useState<string>("");
  const [sessions, setSessions] = useState<SessionInfo[]>([]);

  const [queueInfo, setQueueInfo] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  const [modal, setModal] = useState<ModalKey>(null);

  const fileRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const esRef = useRef<EventSource | null>(null);
  const indexEsRef = useRef<EventSource | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const [pdfW, setPdfW] = useState(860);
  const mainDragRef = useRef<{ dragging: boolean; startX: number; startW: number } | null>(null);
  const mainRowRef = useRef<HTMLDivElement | null>(null);

  const [scopeMode, setScopeMode] = useState<"all" | "one">(selectedPdf === "Semua dokumen" ? "all" : "one");
  const [scopePick, setScopePick] = useState<string>("");
  const [scopeSearch, setScopeSearch] = useState("");
  const [scopeSort, setScopeSort] = useState<"name" | "date">("date");
  const [scopeSortOpen, setScopeSortOpen] = useState(false);
  const scopeSortWrapRef = useRef<HTMLDivElement | null>(null);

  const [hoveredSessionId, setHoveredSessionId] = useState<string | null>(null);
  const [pressedSessionId, setPressedSessionId] = useState<string | null>(null);

  const [examplePrompts, setExamplePrompts] = useState<string[]>([]);
  const [examplePromptsLoading, setExamplePromptsLoading] = useState(false);

  const [processCard, setProcessCard] = useState<ProcessCardState>({
    open: false,
    kind: "index",
    title: "Processing",
    status: "idle",
    stage: "queued",
    message: "",
    progress: 0,
    startedAt: 0,
  });

  const autoCloseTimerRef = useRef<number | null>(null);

  const openProcessCard = useCallback((next: Partial<ProcessCardState>) => {
    setProcessCard((prev) => ({
      open: true,
      kind: next.kind ?? prev.kind ?? "index",
      title: next.title ?? prev.title ?? "Processing",
      status: next.status ?? prev.status ?? "running",
      stage: next.stage ?? prev.stage ?? "queued",
      message: next.message ?? prev.message ?? "",
      progress: next.progress ?? prev.progress ?? 0,
      currentDocument: next.currentDocument ?? prev.currentDocument ?? null,
      currentPage: next.currentPage ?? prev.currentPage ?? null,
      totalPages: next.totalPages ?? prev.totalPages ?? null,
      processedDocs: next.processedDocs ?? prev.processedDocs ?? null,
      totalDocs: next.totalDocs ?? prev.totalDocs ?? null,
      processedChunks: next.processedChunks ?? prev.processedChunks ?? null,
      totalChunks: next.totalChunks ?? prev.totalChunks ?? null,
      processedBatches: next.processedBatches ?? prev.processedBatches ?? null,
      totalBatches: next.totalBatches ?? prev.totalBatches ?? null,
      extractionMode: next.extractionMode ?? prev.extractionMode ?? null,
      startedAt: next.startedAt ?? prev.startedAt ?? Date.now(),
    }));
  }, []);

  const clearAutoCloseTimer = useCallback(() => {
    if (autoCloseTimerRef.current) {
      window.clearTimeout(autoCloseTimerRef.current);
      autoCloseTimerRef.current = null;
    }
  }, []);

  const closeProcessCard = useCallback(() => {
    clearAutoCloseTimer();
    setProcessCard((prev) => ({ ...prev, open: false }));
  }, [clearAutoCloseTimer]);

  useEffect(() => {
    clearAutoCloseTimer();

    if (!processCard.open) return;
    if (!(processCard.status === "success" || processCard.status === "error")) return;

    const delay = processCard.status === "success" ? 1200 : 1800;
    autoCloseTimerRef.current = window.setTimeout(() => {
      setProcessCard((prev) => ({ ...prev, open: false }));
      autoCloseTimerRef.current = null;
    }, delay);

    return clearAutoCloseTimer;
  }, [processCard.open, processCard.status, clearAutoCloseTimer]);

  useLayoutEffect(() => {
    function fitInitialPdfW() {
      const row = mainRowRef.current;
      if (!row) return;

      const rowW = row.getBoundingClientRect().width;
      const minPdf = 520;
      const minChat = 420;
      const splitter = 10;

      const target = Math.round(rowW * 0.58);
      const maxPdf = Math.max(minPdf, rowW - minChat - splitter);
      setPdfW(clamp(target, minPdf, maxPdf));
    }

    fitInitialPdfW();
    window.addEventListener("resize", fitInitialPdfW);
    return () => window.removeEventListener("resize", fitInitialPdfW);
  }, []);

  const canChat = useMemo(() => docs.length > 0, [docs]);

  const autosizePrompt = useCallback(() => {
    const el = promptRef.current;
    if (!el) return;
    el.style.height = "0px";
    const max = 120;
    el.style.height = Math.min(el.scrollHeight, max) + "px";
  }, []);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    });
  }, []);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(DOC_META_KEY);
      if (raw) setDocMeta(JSON.parse(raw));
    } catch {}
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(DOC_META_KEY, JSON.stringify(docMeta));
    } catch {}
  }, [docMeta]);

  async function fetchDocs() {
    try {
      const res = await fetch(`${API_BASE}/api/pdfs`);
      const data = await res.json();
      const list = (data.pdfs || []) as string[];
      setDocs(list);

      const now = Date.now();
      setDocMeta((prev) => {
        const next = { ...prev };
        for (const d of list) {
          if (!next[d]) next[d] = { name: d, firstSeenAt: now };
        }
        for (const k of Object.keys(next)) {
          if (!list.includes(k)) delete next[k];
        }
        return next;
      });
    } catch {}
  }

  async function fetchIndexStatus() {
    setIndexStatusLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/index/status`);
      const data = await res.json();

      const next: IndexStatus = {
        docs: Array.isArray(data?.docs) ? data.docs : [],
        indexed: Array.isArray(data?.indexed) ? data.indexed : [],
        missing: Array.isArray(data?.missing) ? data.missing : [],
      };

      setIndexStatus(next);
    } catch {
      setIndexStatus({ docs: [], indexed: [], missing: [] });
    } finally {
      setIndexStatusLoading(false);
    }
  }

  async function fetchSessions(): Promise<SessionInfo[]> {
    try {
      const res = await fetch(`${API_BASE}/api/sessions`);
      const data = await res.json();
      const list = (data.sessions || []) as SessionInfo[];
      setSessions(list);
      return list;
    } catch {
      return [];
    }
  }

  async function fetchExamplePrompts(scope: string) {
    if (!docs.length) {
      setExamplePrompts([]);
      return;
    }

    setExamplePromptsLoading(true);
    try {
      const url = new URL(`${API_BASE}/api/example-prompts`);
      url.searchParams.set("selected_pdf", scope || "Semua dokumen");

      const res = await fetch(url.toString());
      const data = await res.json();
      const prompts = Array.isArray(data?.examples)
        ? data.examples.filter((x: unknown) => typeof x === "string" && x.trim())
        : [];

      setExamplePrompts(prompts.slice(0, 4));
    } catch {
      setExamplePrompts([]);
    } finally {
      setExamplePromptsLoading(false);
    }
  }

  async function createNewSession() {
    const res = await fetch(`${API_BASE}/api/session/new`, { method: "POST" });
    const data = await res.json();

    setSessionId(data.session_id);
    setChat([]);
    setActiveCitation(null);
    setCitationJumpKey("");
    setQueueInfo("");
    setInput("");

    pushToast("New chat created.", "success");
    await fetchSessions();
  }

  async function loadSession(sid: string) {
    try {
      const res = await fetch(`${API_BASE}/api/session/${sid}`);
      const data = await res.json();

      if (!data || !data.session_id || data.error === "not_found") {
        setChat([]);
        return;
      }

      const msgs = (data.messages || []) as any[];
      setChat(
        msgs.map((m) => ({
          id: uid(),
          role: m.role,
          content: m.content,
          ts: m.ts,
          citations: m.citations || undefined,
        }))
      );

      setTimeout(scrollToBottom, 50);
    } catch {}
  }

  useEffect(() => {
    autosizePrompt();
  }, [input, autosizePrompt]);

  useEffect(() => {
    (async () => {
      await fetchDocs();
      await fetchIndexStatus();
      const list = await fetchSessions();

      if (!sessionId) {
        if (list && list.length > 0) {
          setSessionId(list[0].session_id);
          await loadSession(list[0].session_id);
        } else {
          await createNewSession();
        }
      }
    })();
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    loadSession(sessionId);
  }, [sessionId]);

  useEffect(() => {
    setActiveCitation(null);
    setCitationJumpKey("");

    if (selectedPdf === "Semua dokumen") {
      setViewerDoc(null);
      return;
    }

    setViewerDoc(selectedPdf);
  }, [selectedPdf]);

  useEffect(() => {
    if (modal !== "scope") return;
    const mode = selectedPdf === "Semua dokumen" ? "all" : "one";
    setScopeMode(mode);
    setScopePick(mode === "one" ? selectedPdf : "");
    setScopeSortOpen(false);
  }, [modal, selectedPdf]);

  useEffect(() => {
    if (!scopeSortOpen) return;

    const onDown = (e: MouseEvent) => {
      const wrap = scopeSortWrapRef.current;
      if (!wrap) return;
      if (!wrap.contains(e.target as Node)) setScopeSortOpen(false);
    };

    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [scopeSortOpen]);

  useEffect(() => {
    if (chat.length > 0) return;
    fetchExamplePrompts(selectedPdf);
  }, [chat.length, selectedPdf, docs.length]);

  const stopStream = useCallback(
    (showToast = true) => {
      esRef.current?.close();
      esRef.current = null;
      setIsStreaming(false);
      if (showToast) pushToast("Streaming stopped.", "warning");
    },
    [pushToast]
  );

  const stopIndexStream = useCallback(() => {
    indexEsRef.current?.close();
    indexEsRef.current = null;
  }, []);

  function startStream() {
    if (!input.trim()) return;
    if (!sessionId) return;
    if (isStreaming) return;

    stopStream(false);
    setIsStreaming(true);
    setQueueInfo("");

    const userText = input.trim();
    setInput("");

    const assistantId = uid();
    setChat((prev) => [
      ...prev,
      { id: uid(), role: "user", content: userText, ts: Date.now() },
      { id: assistantId, role: "assistant", content: "", ts: Date.now() },
    ]);
    scrollToBottom();

    const url = new URL(`${API_BASE}/api/chat/stream`);
    url.searchParams.set("message", userText);
    url.searchParams.set("selected_pdf", selectedPdf);
    url.searchParams.set("session_id", sessionId);

    const es = new EventSource(url.toString());
    esRef.current = es;

    es.addEventListener("toast", (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        const msg = payload?.message;
        const kind = (payload?.kind || "info") as ToastKind;
        if (msg) pushToast(msg, kind);
      } catch {}
    });

    es.addEventListener("session_title", async () => {
      await fetchSessions();
    });

    es.addEventListener("queued", (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        if (payload.position && payload.position <= 5) {
          pushToast(`Queued: position #${payload.position}`, "info");
        }
      } catch {}
    });

    es.addEventListener("token", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      const token = payload.token ?? "";
      setChat((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + token } : m)));
      scrollToBottom();
    });

    es.addEventListener("done", async (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      const cites = (payload.citations || []) as Citation[];

      setChat((prev) => prev.map((m) => (m.id === assistantId ? { ...m, citations: cites } : m)));

      setIsStreaming(false);
      es.close();
      esRef.current = null;

      pushToast("Done.", "success");

      await fetchSessions();
      await loadSession(sessionId);
    });

    const onErr = () => {
      setIsStreaming(false);
      es.close();
      esRef.current = null;
      pushToast("Error / connection lost.", "error");
    };

    es.addEventListener("error", onErr);
    es.onerror = onErr;
  }

  function clickCitation(c: Citation) {
    setViewerDoc(c.doc_name);
    setActiveCitation({ ...c });
    setCitationJumpKey(`${Date.now()}-${Math.random().toString(16).slice(2)}`);
  }

  function uniqueCitationPills(cites: Citation[]) {
    const seen = new Set<string>();
    const out: Citation[] = [];
    for (const c of cites) {
      const ext = extOf(c.doc_name);
      const key =
        ext === ".pdf"
          ? `${c.doc_name}::${c.page}::${JSON.stringify(c.paragraph_bboxes || c.bbox || null)}`
          : ext === ".docx"
          ? `${c.doc_name}::${c.kind}::${c.para_start ?? ""}::${c.para_end ?? ""}::${c.table_index ?? ""}`
          : `${c.doc_name}::${c.chunk_id ?? ""}`;

      if (seen.has(key)) continue;
      seen.add(key);
      out.push(c);
    }
    return out;
  }

  async function uploadFile(file: File) {
    stopStream(false);
    stopIndexStream();
    setActiveCitation(null);
    setCitationJumpKey("");
    setQueueInfo("");

    try {
      openProcessCard({
        open: true,
        kind: "upload",
        title: "Uploading document",
        status: "running",
        stage: "preparing",
        message: `Uploading ${file.name}`,
        progress: 20,
        currentDocument: file.name,
        startedAt: Date.now(),
      });

      pushToast("Uploading document…", "info");

      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${API_BASE}/api/pdf/upload`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || data?.error || "Upload failed");

      openProcessCard({
        kind: "success",
        title: "Upload completed",
        status: "success",
        stage: "done",
        message: "Document uploaded successfully.",
        progress: 100,
        currentDocument: data?.doc_name || file.name,
      });

      setModal(null);

      await fetchDocs();
      await fetchIndexStatus();
      if (chat.length === 0) {
        await fetchExamplePrompts(selectedPdf);
      }
      pushToast("Upload successful.", "success");
    } catch (err: any) {
      openProcessCard({
        kind: "error",
        title: "Upload failed",
        status: "error",
        stage: "error",
        message: err?.message || "Upload error",
        progress: 100,
        currentDocument: file.name,
      });

      pushToast(err?.message || "Upload error", "error");
      alert(err?.message || "Upload error");
    }
  }

  async function onPickUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    await uploadFile(f);
    e.target.value = "";
  }

  async function doIndex(force: boolean) {
    stopIndexStream();

    setQueueInfo("");
    openProcessCard({
      open: true,
      kind: force ? "reindex" : "index",
      title: force ? "Re-indexing documents" : "Indexing documents",
      status: "running",
      stage: "queued",
      message: "Preparing indexing request",
      progress: 0,
      startedAt: Date.now(),
      currentDocument: null,
      currentPage: null,
      totalPages: null,
      processedDocs: 0,
      totalDocs: null,
      processedChunks: 0,
      totalChunks: 0,
      processedBatches: 0,
      totalBatches: 0,
      extractionMode: null,
    });

    pushToast(force ? "Re-indexing…" : "Indexing…", "info");

    const url = new URL(`${API_BASE}/api/index/stream`);
    url.searchParams.set("force", force ? "true" : "false");
    url.searchParams.set("scope", selectedPdf);

    const es = new EventSource(url.toString());
    indexEsRef.current = es;

    es.addEventListener("start", (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        openProcessCard({
          kind: force ? "reindex" : "index",
          title: payload?.title || (force ? "Re-indexing documents" : "Indexing documents"),
          status: "running",
          stage: payload?.stage || "preparing",
          message: payload?.message || "Starting indexing pipeline",
          progress: payload?.progress ?? 1,
          processedDocs: payload?.processed_docs ?? 0,
          totalDocs: payload?.total_docs ?? null,
        });
      } catch {}
    });

    es.addEventListener("progress", (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);

        openProcessCard({
          kind: force ? "reindex" : "index",
          title: force ? "Re-indexing documents" : "Indexing documents",
          status: payload?.status === "error" ? "error" : payload?.status === "done" ? "success" : "running",
          stage: payload?.stage || "processing",
          message: payload?.message || "Processing",
          progress: payload?.progress ?? 0,
          currentDocument: payload?.current_document ?? null,
          currentPage: payload?.current_page ?? null,
          totalPages: payload?.total_pages ?? null,
          processedDocs: payload?.processed_docs ?? null,
          totalDocs: payload?.total_docs ?? null,
          processedChunks: payload?.processed_chunks ?? null,
          totalChunks: payload?.total_chunks ?? null,
          processedBatches: payload?.processed_batches ?? null,
          totalBatches: payload?.total_batches ?? null,
          extractionMode: payload?.extraction_mode ?? null,
        });
      } catch {}
    });

    es.addEventListener("done", async (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        openProcessCard({
          kind: "success",
          title: force ? "Re-index completed" : "Index completed",
          status: "success",
          stage: "done",
          message: payload?.message || (force ? "Re-index completed successfully." : "Index completed successfully."),
          progress: 100,
          currentDocument: payload?.current_document ?? null,
          processedDocs: payload?.added_docs ?? payload?.processed_docs ?? null,
          totalDocs: payload?.total_docs ?? null,
          processedChunks: payload?.added_chunks ?? payload?.processed_chunks ?? null,
          totalChunks: payload?.total_chunks ?? null,
        });
      } catch {
        openProcessCard({
          kind: "success",
          title: force ? "Re-index completed" : "Index completed",
          status: "success",
          stage: "done",
          message: force ? "Re-index completed successfully." : "Index completed successfully.",
          progress: 100,
        });
      }

      setQueueInfo(force ? "Re-index completed." : "Index completed.");
      pushToast(force ? "Re-index completed." : "Index completed.", "success");
      setModal(null);
      stopIndexStream();
      await fetchDocs();
      await fetchIndexStatus();
      if (chat.length === 0) {
        await fetchExamplePrompts(selectedPdf);
      }
    });

    es.addEventListener("error", async (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        openProcessCard({
          kind: "error",
          title: force ? "Re-index failed" : "Index failed",
          status: "error",
          stage: "error",
          message: payload?.message || "Indexing failed",
          progress: 100,
          currentDocument: payload?.current_document ?? null,
        });
        pushToast(payload?.message || "Indexing failed", "error");
      } catch {
        openProcessCard({
          kind: "error",
          title: force ? "Re-index failed" : "Index failed",
          status: "error",
          stage: "error",
          message: "Error / connection lost.",
          progress: 100,
        });
        pushToast("Error / connection lost.", "error");
      } finally {
        stopIndexStream();
        await fetchIndexStatus();
      }
    });

    es.addEventListener("toast", (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        const msg = payload?.message;
        const kind = (payload?.kind || "info") as ToastKind;
        if (msg) pushToast(msg, kind);
      } catch {}
    });

    es.onerror = () => {};
  }

  const indexedSet = useMemo(() => new Set(indexStatus.indexed), [indexStatus.indexed]);
  const newDocs = useMemo(() => docs.filter((d) => !indexedSet.has(d)), [docs, indexedSet]);

  const scopeList = useMemo(() => {
    const q = scopeSearch.trim().toLowerCase();
    let list = docs.slice();
    if (q) list = list.filter((d) => d.toLowerCase().includes(q));

    if (scopeSort === "name") {
      list.sort((a, b) => a.localeCompare(b));
    } else {
      list.sort((a, b) => {
        const ta = docMeta[a]?.firstSeenAt ?? 0;
        const tb = docMeta[b]?.firstSeenAt ?? 0;
        return tb - ta;
      });
    }
    return list;
  }, [docs, scopeSearch, scopeSort, docMeta]);

  function fmtDate(ts?: number) {
    if (!ts) return "";
    try {
      const d = new Date(ts);
      return d.toLocaleString("en-US", { day: "2-digit", month: "short", year: "numeric" });
    } catch {
      return "";
    }
  }

  async function applyScopeSelection() {
    if (scopeMode === "all") {
      setSelectedPdf("Semua dokumen");
      setModal(null);
      if (chat.length === 0) {
        await fetchExamplePrompts("Semua dokumen");
      }
      return;
    }

    if (!scopePick) {
      pushToast("Please select one document.", "warning");
      return;
    }

    setSelectedPdf(scopePick);
    setModal(null);
    if (chat.length === 0) {
      await fetchExamplePrompts(scopePick);
    }
  }

  useEffect(() => {
    function onMove(ev: MouseEvent) {
      if (!mainDragRef.current?.dragging) return;

      const dx = ev.clientX - mainDragRef.current.startX;
      const row = mainRowRef.current;
      const rowW = row ? row.getBoundingClientRect().width : 1200;

      const minPdf = 520;
      const minChat = 420;
      const splitter = 10;

      const maxPdf = Math.max(minPdf, rowW - minChat - splitter);
      setPdfW(clamp(mainDragRef.current.startW + dx, minPdf, maxPdf));
    }

    function onUp() {
      if (mainDragRef.current?.dragging) mainDragRef.current.dragging = false;
      document.body.style.userSelect = "";
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  const sortedSessions = useMemo(() => {
    const list = sessions.slice();
    list.sort((a, b) => (b.updated_at ?? b.created_at ?? 0) - (a.updated_at ?? a.created_at ?? 0));
    return list;
  }, [sessions]);

  const currentSession = useMemo(() => sessions.find((s) => s.session_id === sessionId) || null, [sessions, sessionId]);

  const scopeSortLabel = scopeSort === "date" ? "Sort: Upload date" : "Sort: Name";

  const fillExamplePrompt = (text: string) => {
    setInput(text);
    requestAnimationFrame(() => {
      autosizePrompt();
      promptRef.current?.focus();
    });
  };

  const renderPromptBar = (variant: "embedded" | "bottom") => {
    const isEmbedded = variant === "embedded";
    return (
      <div style={{ padding: isEmbedded ? "12px 0 0 0" : "12px 2px 2px 2px" }}>
        <div className={styles.promptWrap}>
          <textarea
            ref={promptRef}
            className={styles.promptBox}
            placeholder={canChat ? "Ask something…" : "No documents yet…"}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              requestAnimationFrame(autosizePrompt);
            }}
            disabled={!canChat || !sessionId || isStreaming}
            rows={1}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (!isStreaming) startStream();
              }
            }}
          />

          <button
            type="button"
            className={`${styles.iconBtn} ${isStreaming ? styles.iconBtnDanger : styles.iconBtnPrimary}`}
            onClick={() => (isStreaming ? stopStream() : startStream())}
            disabled={!sessionId || (!isStreaming && (!canChat || !input.trim()))}
            title={isStreaming ? "Stop" : "Send"}
          >
            {isStreaming ? (
              <svg className={styles.iconSvg} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <rect x="7" y="7" width="10" height="10" rx="2" />
              </svg>
            ) : (
              <svg className={styles.iconSvg} viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M3.5 11.2L20 4.5l-6.7 16.5-2.4-6.2-7.4-3.6z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                <path d="M10.9 14.8L20 4.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            )}
          </button>
        </div>
      </div>
    );
  };

  return (
    <main
      className={jakarta.variable}
      style={{
        position: "relative",
        height: "100vh",
        display: "flex",
        overflow: "hidden",
        color: "rgba(20,20,20,0.92)",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage: "url('/makunin-galicia-116549.jpg')",
          backgroundSize: "cover",
          backgroundPosition: "center",
          filter: "blur(10px)",
          transform: "scale(1.05)",
          zIndex: 0,
        }}
      />

      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "rgba(255,255,255,0.65)",
          zIndex: 1,
        }}
      />

      <style jsx global>{`
        @keyframes pdhSpin {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
        @keyframes pdhDots {
          0%,
          20% {
            opacity: 0.25;
          }
          50% {
            opacity: 1;
          }
          100% {
            opacity: 0.25;
          }
        }
        .pdh-spinner {
          width: 14px;
          height: 14px;
          border: 2px solid rgba(20, 20, 20, 0.18);
          border-top-color: rgba(17, 101, 255, 0.9);
          border-radius: 999px;
          animation: pdhSpin 0.8s linear infinite;
        }
        .pdh-dots span {
          display: inline-block;
          margin-right: 2px;
          animation: pdhDots 1.1s infinite;
          font-weight: 900;
        }
        .pdh-dots span:nth-child(2) {
          animation-delay: 0.12s;
        }
        .pdh-dots span:nth-child(3) {
          animation-delay: 0.24s;
        }
      `}</style>

      <ToastHost items={toasts} onRemove={removeToast} />
      <ProcessCardOverlay state={processCard} onClose={closeProcessCard} />

      <aside
        style={{
          position: "relative",
          zIndex: 2,
          width: 72,
          minWidth: 72,
          maxWidth: 72,
          display: "flex",
          flexDirection: "column",
          gap: 10,
          padding: 12,
          background: "#ffffff",
          borderRight: "1px solid rgba(20,20,20,0.10)",
        }}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
          onChange={onPickUpload}
          style={{ display: "none" }}
        />

        <SideBtn title="Upload document" iconSrc="/ikonuploadV22.png" onClick={() => setModal("upload")} />
        <SideBtn title="Index documents" iconSrc="/ikonindexV22.png" onClick={() => setModal("index")} />
        <SideBtn title="Document scope" iconSrc="/ikonscopeV22.png" onClick={() => setModal("scope")} />

        <div className={styles.sidebarDivider} />

        <SideBtn
          title="New chat"
          iconSrc="/ikonchatV22.png"
          onClick={async () => {
            setModal(null);
            await createNewSession();
            if (docs.length > 0) {
              await fetchExamplePrompts(selectedPdf);
            }
          }}
        />
        <SideBtn title="Chat history" iconSrc="/ikonhistoryV22.png" onClick={() => setModal("history")} />
      </aside>

      <Modal open={modal === "upload"} title="Upload document" onClose={() => setModal(null)}>
        <div
          className={`${styles.dropzone} ${dragOver ? styles.dropzoneActive : ""}`}
          onDragEnter={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setDragOver(true);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setDragOver(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setDragOver(false);
          }}
          onDrop={async (e) => {
            e.preventDefault();
            e.stopPropagation();
            setDragOver(false);
            const f = e.dataTransfer.files?.[0];
            if (!f) return;
            await uploadFile(f);
          }}
        >
          <div style={{ fontWeight: 900, fontSize: 14, color: "rgba(20,20,20,0.92)" }}>Drag & drop your document here</div>
          <div className={styles.helpText}>
            Supported: <b>PDF / DOCX</b>
          </div>

          <button className={styles.uiBtn} type="button" style={{ width: "auto" }} onClick={() => fileRef.current?.click()}>
            Choose file
          </button>
        </div>

        <div style={{ marginTop: 14 }} className={styles.helpText}>
          Maximum upload size: <b>{MAX_UPLOAD_MB} MB</b>.
        </div>
      </Modal>

      <Modal open={modal === "index"} title="Index documents" onClose={() => setModal(null)}>
        <div style={{ display: "grid", gap: 10 }}>
          <div className={styles.kv}>
            <div className={styles.kvLabel}>Uploaded documents</div>
            <div className={styles.kvValue}>{docs.length}</div>
          </div>

          <div className={styles.kv}>
            <div className={styles.kvLabel}>Indexed documents</div>
            <div className={styles.kvValue}>{indexStatus.indexed.length}</div>
          </div>

          <div className={styles.kv}>
            <div className={styles.kvLabel}>Pending index</div>
            <div className={styles.kvValue}>{newDocs.length}</div>
          </div>

          <div className={styles.helpText}>
            {indexStatusLoading
              ? "Reading latest index status from backend…"
              : newDocs.length > 0
              ? "There are documents that have not been indexed yet."
              : "All current documents are already indexed."}
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 6 }}>
            <button className={styles.uiBtn} type="button" style={{ width: "auto" }} onClick={() => doIndex(false)}>
              Index
            </button>
            <button className={`${styles.uiBtn} ${styles.uiBtnDanger}`} type="button" style={{ width: "auto" }} onClick={() => doIndex(true)}>
              Re-index
            </button>
            <button className={`${styles.uiBtn} ${styles.uiBtnGhost}`} type="button" style={{ width: "auto" }} onClick={fetchIndexStatus}>
              Refresh status
            </button>
          </div>

          {queueInfo ? <div className={styles.helpText}>• {queueInfo}</div> : null}
        </div>
      </Modal>

      <Modal open={modal === "scope"} title="Document scope" onClose={() => setModal(null)}>
        <div className={styles.radioRow}>
          <div
            className={`${styles.radioPill} ${scopeMode === "all" ? styles.radioPillActive : ""}`}
            onClick={() => setScopeMode("all")}
            role="button"
            tabIndex={0}
          >
            <span>✓</span> All documents
          </div>
          <div
            className={`${styles.radioPill} ${scopeMode === "one" ? styles.radioPillActive : ""}`}
            onClick={() => setScopeMode("one")}
            role="button"
            tabIndex={0}
          >
            <span>✓</span> Single document
          </div>
        </div>

        <div className={styles.scopeToolbar}>
          <input
            className={styles.scopeInput}
            value={scopeSearch}
            onChange={(e) => setScopeSearch(e.target.value)}
            placeholder="Search documents…"
            style={{ flex: 1, minWidth: 220 }}
          />

          <div ref={scopeSortWrapRef} style={{ position: "relative", width: 210 }}>
            <button
              type="button"
              className={`${styles.uiBtn} ${styles.uiBtnGhost}`}
              onClick={() => setScopeSortOpen((v) => !v)}
              style={{
                width: "100%",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 10,
                padding: "10px 12px",
                borderRadius: 12,
                fontWeight: 900,
              }}
              aria-haspopup="menu"
              aria-expanded={scopeSortOpen}
              title="Sort"
            >
              <span style={{ fontSize: 12 }}>{scopeSortLabel}</span>
              <span aria-hidden="true" style={{ fontSize: 12, opacity: 0.75 }}>
                ▾
              </span>
            </button>

            {scopeSortOpen ? (
              <div
                role="menu"
                className={styles.panel}
                style={{
                  position: "absolute",
                  top: "calc(100% + 8px)",
                  left: 0,
                  right: 0,
                  zIndex: 50,
                  padding: 8,
                  borderRadius: 12,
                  boxShadow: "0 18px 44px rgba(0,0,0,0.16)",
                }}
              >
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setScopeSort("date");
                    setScopeSortOpen(false);
                  }}
                  className={`${styles.uiBtn} ${styles.uiBtnGhost}`}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    padding: "10px 10px",
                    borderRadius: 10,
                    fontWeight: 900,
                    marginBottom: 6,
                    background: scopeSort === "date" ? "rgba(17,101,255,0.08)" : undefined,
                    borderColor: scopeSort === "date" ? "rgba(17,101,255,0.28)" : undefined,
                  }}
                >
                  Sort: Upload date
                </button>

                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setScopeSort("name");
                    setScopeSortOpen(false);
                  }}
                  className={`${styles.uiBtn} ${styles.uiBtnGhost}`}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    padding: "10px 10px",
                    borderRadius: 10,
                    fontWeight: 900,
                    background: scopeSort === "name" ? "rgba(17,101,255,0.08)" : undefined,
                    borderColor: scopeSort === "name" ? "rgba(17,101,255,0.28)" : undefined,
                  }}
                >
                  Sort: Name
                </button>
              </div>
            ) : null}
          </div>

          <button className={styles.uiBtn} type="button" style={{ width: "auto" }} onClick={applyScopeSelection}>
            Apply
          </button>
        </div>

        {scopeMode === "one" ? (
          <div className={styles.scopeList}>
            {scopeList.map((d) => (
              <div key={d} className={styles.scopeItem}>
                <input type="radio" name="scopePick" checked={scopePick === d} onChange={() => setScopePick(d)} />
                <div className={styles.scopeItemName} title={d}>
                  {d}
                </div>
                <div className={styles.scopeItemMeta}>{fmtDate(docMeta[d]?.firstSeenAt)}</div>
              </div>
            ))}
            {!scopeList.length ? <div className={styles.helpText}>No matching documents.</div> : null}
          </div>
        ) : (
          <div className={styles.helpText}>
            In <b>All documents</b> mode, the viewer opens only when you click a citation in the answer.
          </div>
        )}
      </Modal>

      <Modal open={modal === "history"} title="Chat history" onClose={() => setModal(null)}>
        <div style={{ display: "grid", gap: 12 }}>
          {currentSession ? (
            <div>
              <div className={styles.helpText} style={{ marginBottom: 8 }}>
                Current chat
              </div>
              <div className={styles.scopeItem}>
                <div className={styles.scopeItemName}>
                  {(currentSession.title && currentSession.title.trim()) || `Session ${currentSession.session_id.slice(0, 6)}`}
                </div>
              </div>
            </div>
          ) : null}

          <div>
            <div className={styles.helpText} style={{ marginBottom: 8 }}>
              Recent chats
            </div>

            <div className={styles.historyList}>
              {sortedSessions.map((s) => {
                const label = (s.title && s.title.trim()) || `Session ${s.session_id.slice(0, 6)}`;
                const isActive = s.session_id === sessionId;
                const isHover = hoveredSessionId === s.session_id;
                const isPress = pressedSessionId === s.session_id;

                return (
                  <button
                    key={s.session_id}
                    type="button"
                    className={styles.scopeItem}
                    style={{
                      cursor: "pointer",
                      textAlign: "left",
                      outline: "none",
                      transition: "transform 120ms ease, background 120ms ease, border-color 120ms ease, box-shadow 120ms ease",
                      transform: isPress ? "scale(0.985)" : "scale(1)",
                      boxShadow: isHover ? "0 14px 32px rgba(0,0,0,0.10)" : undefined,
                      borderColor: isActive ? "rgba(17,101,255,0.28)" : isHover ? "rgba(17,101,255,0.16)" : undefined,
                      background: isActive ? "rgba(17,101,255,0.08)" : isHover ? "rgba(20,20,20,0.04)" : undefined,
                    }}
                    onMouseEnter={() => setHoveredSessionId(s.session_id)}
                    onMouseLeave={() => {
                      setHoveredSessionId((cur) => (cur === s.session_id ? null : cur));
                      setPressedSessionId((cur) => (cur === s.session_id ? null : cur));
                    }}
                    onMouseDown={() => setPressedSessionId(s.session_id)}
                    onMouseUp={() => setPressedSessionId((cur) => (cur === s.session_id ? null : cur))}
                    onClick={async () => {
                      setModal(null);
                      setSessionId(s.session_id);
                      await loadSession(s.session_id);
                    }}
                  >
                    <div className={styles.scopeItemName} title={label}>
                      {label}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button
              className={styles.uiBtn}
              type="button"
              style={{ width: "auto" }}
              onClick={async () => {
                setModal(null);
                await createNewSession();
                if (docs.length > 0) {
                  await fetchExamplePrompts(selectedPdf);
                }
              }}
            >
              New chat
            </button>
            <button
              className={`${styles.uiBtn} ${styles.uiBtnGhost}`}
              type="button"
              style={{ width: "auto" }}
              onClick={async () => {
                await fetchSessions();
                pushToast("History refreshed.", "success");
              }}
            >
              Refresh
            </button>
          </div>
        </div>
      </Modal>

      <div
        style={{
          position: "relative",
          zIndex: 2,
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
        }}
      >
        <div
          style={{
            padding: "12px 14px",
            display: "flex",
            alignItems: "center",
            gap: 10,
            background: "transparent",
            minHeight: 52,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <img src="/LogoV22.png" alt="Logo" style={{ height: 36, width: "auto", display: "block", objectFit: "contain" }} />
          </div>

          <div style={{ flex: 1 }} />

          {isStreaming ? (
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div className="pdh-spinner" aria-hidden="true" />
              <div style={{ fontSize: 12, color: "rgba(20,20,20,0.55)", fontWeight: 900 }}>Generating…</div>
            </div>
          ) : null}
        </div>

        <div
          ref={mainRowRef}
          style={{
            flex: 1,
            display: "flex",
            minHeight: 0,
            minWidth: 0,
            padding: 16,
            gap: 14,
          }}
        >
          <section className={styles.surface} style={{ width: pdfW, minWidth: 520, display: "flex", flexDirection: "column", minHeight: 0 }}>
            <div className={styles.surfaceHeader}>
              <div className={styles.chipSm} title={viewerDoc || (selectedPdf !== "Semua dokumen" ? selectedPdf : "No document opened")}>
                {viewerDoc || (selectedPdf !== "Semua dokumen" ? selectedPdf : "Click a citation to open a document")}
              </div>

              {selectedPdf === "Semua dokumen" && viewerDoc ? (
                <button
                  className={styles.pillBtnSm}
                  onClick={() => {
                    setViewerDoc(null);
                    setActiveCitation(null);
                    setCitationJumpKey("");
                  }}
                  title="Close viewer"
                  type="button"
                >
                  Close
                </button>
              ) : null}

              <div style={{ flex: 1 }} />
              <div style={{ fontSize: 12, color: "rgba(20,20,20,0.55)", fontWeight: 900 }}>
                {viewerDoc ? `Type: ${extOf(viewerDoc).replace(".", "").toUpperCase()}` : ""}
              </div>
            </div>

            <div className={styles.surfaceBody} style={{ flex: 1, minHeight: 0 }}>
              {viewerDoc ? (
                <div className={styles.panel} style={{ height: "100%", overflow: "hidden" }}>
                  <DocumentViewer key={`${viewerDoc}::${citationJumpKey || "steady"}`} selectedDoc={viewerDoc} citation={activeCitation} />
                </div>
              ) : (
                <div
                  className={styles.panel}
                  style={{
                    height: "100%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: 18,
                    textAlign: "center",
                    color: "rgba(20,20,20,0.60)",
                  }}
                >
                  <div>
                    <div style={{ marginBottom: 8, color: "rgba(20,20,20,0.92)", fontWeight: 900 }}>Document viewer</div>
                    <div style={{ fontSize: 13, lineHeight: 1.6 }}>
                      {selectedPdf === "Semua dokumen"
                        ? "All-documents scope is active. Click a citation in the answer to open the document here."
                        : "Single-document scope is active. The document opens automatically here."}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>

          <div
            onMouseDown={(e) => {
              mainDragRef.current = { dragging: true, startX: e.clientX, startW: pdfW };
              document.body.style.userSelect = "none";
            }}
            title="Drag to resize viewer/chat"
            className={styles.splitter}
          >
            <div className={styles.splitterBar} />
          </div>

          <section style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, minHeight: 0 }}>
            <div style={{ padding: "8px 2px 10px 2px", display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ fontWeight: 900, color: "rgba(20,20,20,0.92)" }}>Chat</div>
              <div style={{ flex: 1 }} />
              <div style={{ fontSize: 12, color: "rgba(20,20,20,0.55)", fontWeight: 900 }}>{canChat ? "" : "No documents"}</div>
            </div>

            <div className={styles.chatScroll} style={{ flex: 1, minHeight: 0, overflow: "auto", padding: 2, paddingRight: 4 }}>
              {chat.length === 0 ? (
                <div className={styles.panel} style={{ padding: 16 }}>
                  <div style={{ marginBottom: 10, color: "rgba(20,20,20,0.92)", fontWeight: 900 }}>Example prompts</div>

                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {examplePromptsLoading ? (
                      <div className={styles.helpText}>Generating example prompts from indexed documents…</div>
                    ) : examplePrompts.length > 0 ? (
                      examplePrompts.map((p, i) => (
                        <button
                          key={i}
                          className={styles.uiBtn}
                          onClick={() => fillExamplePrompt(p)}
                          style={{ textAlign: "left" }}
                          title="Click to fill input"
                          type="button"
                        >
                          {p}
                        </button>
                      ))
                    ) : (
                      <div className={styles.helpText}>
                        No example prompts available yet. Upload and index documents first.
                      </div>
                    )}
                  </div>

                  <div className={styles.helpText} style={{ marginTop: 14 }}>
                    Tip: after an answer is generated, click <b>Source</b> to open the referenced location.
                  </div>

                  {renderPromptBar("embedded")}
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 2 }}>
                  {chat.map((m, idx) => {
                    const isUser = m.role === "user";
                    const isAssistant = m.role === "assistant";
                    const isLast = idx === chat.length - 1;

                    return (
                      <div key={m.id} style={{ display: "flex", flexDirection: "column", alignItems: isUser ? "flex-end" : "flex-start" }}>
                        <div
                          className={styles.panel}
                          style={{
                            maxWidth: "92%",
                            padding: "10px 12px",
                            background: isUser ? "rgba(17,101,255,0.95)" : "#ffffff",
                            color: isUser ? "#fff" : "rgba(20,20,20,0.92)",
                            borderColor: isUser ? "rgba(17,101,255,0.95)" : "rgba(20,20,20,0.10)",
                          }}
                        >
                          <div style={{ marginBottom: 6, fontSize: 12, opacity: 0.78, fontWeight: 900, display: "flex", alignItems: "center", gap: 8 }}>
                            <span>{isUser ? "You" : isAssistant ? "Assistant" : "System"}</span>
                            {isAssistant && isStreaming && isLast ? <div className="pdh-spinner" aria-hidden="true" /> : null}
                          </div>

                          <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6, fontSize: 14, fontWeight: 600 }}>
                            {m.content || (isAssistant && isStreaming ? "" : "")}
                            {isAssistant && isStreaming && isLast && !m.content ? (
                              <span className="pdh-dots" aria-label="Loading">
                                <span>.</span>
                                <span>.</span>
                                <span>.</span>
                              </span>
                            ) : null}
                          </div>

                          {isAssistant && m.citations && m.citations.length > 0 ? (
                            <div style={{ marginTop: 10 }}>
                              <div style={{ marginBottom: 8, fontSize: 12, color: "rgba(20,20,20,0.55)", fontWeight: 900 }}>Source</div>
                              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                                {uniqueCitationPills(m.citations).map((c, cidx) => (
                                  <button
                                    key={`${m.id}-pill-${cidx}-${c.doc_name}-${c.chunk_id ?? ""}`}
                                    onClick={() => clickCitation(c)}
                                    className={`${styles.uiBtn} ${styles.uiBtnGhost}`}
                                    style={{
                                      width: "auto",
                                      padding: "8px 10px",
                                      borderRadius: 999,
                                      fontSize: 11,
                                      fontWeight: 900,
                                    }}
                                    title={`${c.doc_name} — ${citeLabel(c)}`}
                                    type="button"
                                  >
                                    {c.doc_name} — {citeLabel(c)}
                                  </button>
                                ))}
                              </div>
                            </div>
                          ) : null}
                        </div>
                      </div>
                    );
                  })}
                  <div ref={bottomRef} />
                </div>
              )}
            </div>

            {chat.length > 0 ? renderPromptBar("bottom") : null}
          </section>
        </div>
      </div>
    </main>
  );
}

