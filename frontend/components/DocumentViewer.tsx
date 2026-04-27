"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import PdfViewer, { PdfSelected } from "./PdfViewer";
import DocxViewer from "./DocxViewer";

const API_BASE = "http://127.0.0.1:8000";

type Citation = {
  doc_name: string;
  doc_path?: string;
  page?: number;
  kind?: string;

  // PDF
  bbox?: [number, number, number, number] | null;
  paragraph_bboxes?: [number, number, number, number][] | null;
  page_w?: number | null;
  page_h?: number | null;

  // DOCX
  para_start?: number | null;
  para_end?: number | null;
  table_index?: number | null;
};

type DocxJump =
  | { kind: "para"; paraStart: number; paraEnd?: number | null }
  | { kind: "table"; tableIndex: number }
  | undefined;

function extOf(name: string) {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function clampInt(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

export default function DocumentViewer({
  selectedDoc,
  citation,
}: {
  selectedDoc: string | null;
  citation?: Citation | null;
}) {
  const [pageNumber, setPageNumber] = useState(1);
  const [pageWidth, setPageWidth] = useState<number>(860);
  const [docxPageApprox, setDocxPageApprox] = useState<number>(1);
  const [pdfTotalPages, setPdfTotalPages] = useState<number>(0);

  const viewerWrapRef = useRef<HTMLDivElement | null>(null);

  const citationLockRef = useRef(false);
  const lastCitationSigRef = useRef<string>("");

  const ext = useMemo(() => {
    if (!selectedDoc) return "";
    return extOf(selectedDoc);
  }, [selectedDoc]);

  const fileUrl = useMemo(() => {
    if (!selectedDoc) return "";
    return `${API_BASE}/api/pdf/${encodeURIComponent(selectedDoc)}`;
  }, [selectedDoc]);

  useEffect(() => {
    const el = viewerWrapRef.current;
    if (!el) return;

    const compute = () => {
      const w = el.getBoundingClientRect().width;
      const target = Math.floor(w - 24);
      setPageWidth(clampInt(target, 520, 1400));
    };

    compute();
    const ro = new ResizeObserver(compute);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    setPageNumber(1);
    setDocxPageApprox(1);
    setPdfTotalPages(0);
    citationLockRef.current = false;
    lastCitationSigRef.current = "";
  }, [selectedDoc]);

  useEffect(() => {
    if (!citation || !selectedDoc) return;

    const sig = JSON.stringify({
      selectedDoc,
      ext,
      doc_name: citation.doc_name ?? null,
      page: citation.page ?? null,
      kind: citation.kind ?? null,
      bbox: citation.bbox ?? null,
      paragraph_bboxes: citation.paragraph_bboxes ?? null,
      page_w: citation.page_w ?? null,
      page_h: citation.page_h ?? null,
      para_start: citation.para_start ?? null,
      para_end: citation.para_end ?? null,
      table_index: citation.table_index ?? null,
    });

    if (sig === lastCitationSigRef.current) return;
    lastCitationSigRef.current = sig;

    citationLockRef.current = true;

    if (ext === ".pdf") {
      if (citation.page && Number.isFinite(citation.page)) {
        setPageNumber(Math.max(1, Math.floor(citation.page)));
      }
      return;
    }

    if (ext === ".docx") {
      if (citation.kind === "docx_text" && citation.para_start) {
        setDocxPageApprox(1);
        return;
      }
      if (citation.kind === "docx_table" && citation.table_index) {
        setDocxPageApprox(1);
      }
    }
  }, [citation, ext, selectedDoc]);

  const [pageInput, setPageInput] = useState<string>("1");

  useEffect(() => {
    if (ext === ".pdf") {
      setPageInput(String(pageNumber));
    }
  }, [pageNumber, ext]);

  useEffect(() => {
    if (ext === ".docx") {
      setPageInput(String(docxPageApprox));
    }
  }, [docxPageApprox, ext]);

  const releaseCitationLock = () => {
    citationLockRef.current = false;
  };

  const goPdfPage = (nextPage: number, byUser = false) => {
    const bounded = pdfTotalPages > 0 ? Math.min(Math.max(1, nextPage), pdfTotalPages) : Math.max(1, nextPage);
    if (byUser) releaseCitationLock();
    setPageNumber(bounded);
    setPageInput(String(bounded));
  };

  const goDocxPage = (nextPage: number, byUser = false) => {
    const bounded = Math.max(1, Math.floor(nextPage));
    if (byUser) releaseCitationLock();
    setDocxPageApprox(bounded);
    setPageInput(String(bounded));
  };

  const commitPageInput = () => {
    const raw = pageInput.trim();
    const n = Number(raw);

    if (!Number.isFinite(n)) {
      setPageInput(ext === ".pdf" ? String(pageNumber) : String(docxPageApprox));
      return;
    }

    const page = Math.max(1, Math.floor(n));

    if (ext === ".pdf") {
      goPdfPage(page, true);
      return;
    }

    goDocxPage(page, true);
  };

  const ctrlWrapStyle: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "10px 10px",
    borderBottom: "1px solid rgba(20,20,20,0.08)",
    background: "rgba(255,255,255,0.9)",
    backdropFilter: "blur(8px)",
    position: "sticky",
    top: 0,
    zIndex: 5,
  };

  const btnStyle: React.CSSProperties = {
    border: "1px solid rgba(20,20,20,0.12)",
    background: "#fff",
    borderRadius: 10,
    padding: "8px 10px",
    fontWeight: 900,
    fontSize: 12,
    color: "rgba(20,20,20,0.86)",
    cursor: "pointer",
  };

  const btnDisabledStyle: React.CSSProperties = {
    ...btnStyle,
    opacity: 0.45,
    cursor: "not-allowed",
  };

  const inputStyle: React.CSSProperties = {
    width: 78,
    border: "1px solid rgba(20,20,20,0.14)",
    borderRadius: 10,
    padding: "8px 10px",
    fontWeight: 900,
    fontSize: 12,
    outline: "none",
    background: "#fff",
  };

  const subtleText: React.CSSProperties = {
    fontSize: 12,
    fontWeight: 900,
    color: "rgba(20,20,20,0.55)",
  };

  if (!selectedDoc) {
    return <div style={{ padding: 20 }}>Belum ada dokumen dipilih.</div>;
  }

  if (ext === ".pdf") {
    const selected: PdfSelected | null = citation
      ? {
          page: citation.page || 1,
          bbox: citation.bbox || null,
          paragraph_bboxes: citation.paragraph_bboxes || null,
          page_w: citation.page_w || null,
          page_h: citation.page_h || null,
        }
      : null;

    const safeCurrentPage =
      pdfTotalPages > 0 ? Math.min(Math.max(pageNumber, 1), pdfTotalPages) : Math.max(pageNumber, 1);

    const isPrevDisabled = safeCurrentPage <= 1;
    const isNextDisabled = pdfTotalPages > 0 ? safeCurrentPage >= pdfTotalPages : false;

    return (
      <div
        ref={viewerWrapRef}
        style={{ height: "100%", display: "flex", flexDirection: "column", minHeight: 0 }}
      >
        <div style={ctrlWrapStyle}>
          <button
            type="button"
            style={isPrevDisabled ? btnDisabledStyle : btnStyle}
            disabled={isPrevDisabled}
            onClick={() => goPdfPage(pageNumber - 1, true)}
            title="Previous page"
          >
            ◀ Prev
          </button>

          <button
            type="button"
            style={isNextDisabled ? btnDisabledStyle : btnStyle}
            disabled={isNextDisabled}
            onClick={() => goPdfPage(pageNumber + 1, true)}
            title="Next page"
          >
            Next ▶
          </button>

          <div style={{ width: 10 }} />

          <div style={subtleText}>Page</div>
          <input
            style={inputStyle}
            value={pageInput}
            inputMode="numeric"
            onChange={(e) => setPageInput(e.target.value)}
            onBlur={commitPageInput}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                commitPageInput();
              }
            }}
            aria-label="Page number"
          />

          <div style={subtleText}>of {pdfTotalPages > 0 ? pdfTotalPages : "…"}</div>

          <div style={{ flex: 1 }} />

          <div style={subtleText}>
            {pdfTotalPages > 0 ? `Page ${safeCurrentPage} of ${pdfTotalPages}` : "Loading document…"}
          </div>
        </div>

        <div style={{ flex: 1, minHeight: 0, overflow: "hidden" }}>
          <PdfViewer
            pdfUrl={fileUrl}
            pageNumber={pageNumber}
            setPageNumber={(n) => {
              setPageNumber(n);
            }}
            selected={selected}
            pageWidth={pageWidth}
            onTotalPagesChange={setPdfTotalPages}
          />
        </div>
      </div>
    );
  }

  if (ext === ".docx") {
    let jump: DocxJump = undefined;

    if (citation?.kind === "docx_text" && citation?.para_start) {
      jump = {
        kind: "para",
        paraStart: citation.para_start,
        paraEnd: citation.para_end ?? citation.para_start,
      };
    } else if (citation?.kind === "docx_table" && citation?.table_index) {
      jump = { kind: "table", tableIndex: citation.table_index };
    }

    return (
      <div
        ref={viewerWrapRef}
        style={{ height: "100%", display: "flex", flexDirection: "column", minHeight: 0 }}
      >
        <div style={ctrlWrapStyle}>
          <button
            type="button"
            style={docxPageApprox <= 1 ? btnDisabledStyle : btnStyle}
            disabled={docxPageApprox <= 1}
            onClick={() => goDocxPage(docxPageApprox - 1, true)}
            title="Previous (approx)"
          >
            ◀ Prev
          </button>

          <button
            type="button"
            style={btnStyle}
            onClick={() => goDocxPage(docxPageApprox + 1, true)}
            title="Next (approx)"
          >
            Next ▶
          </button>

          <div style={{ width: 10 }} />

          <div style={subtleText}>Page</div>
          <input
            style={inputStyle}
            value={pageInput}
            inputMode="numeric"
            onChange={(e) => setPageInput(e.target.value)}
            onBlur={commitPageInput}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                commitPageInput();
              }
            }}
            aria-label="Page number (approx)"
          />

          <div style={{ flex: 1 }} />

          <div style={subtleText}>DOCX view · approx page {docxPageApprox}</div>
        </div>

        <div style={{ flex: 1, minHeight: 0, overflow: "hidden" }}>
          <div style={{ height: "100%", overflow: "auto" }}>
            <DocxViewer fileUrl={fileUrl} jump={jump} />
          </div>
        </div>
      </div>
    );
  }

  return <div style={{ padding: 20 }}>Tipe file tidak didukung: {ext}</div>;
}

