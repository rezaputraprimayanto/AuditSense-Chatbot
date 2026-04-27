"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type DocxParagraph = { i: number; text: string };
type DocxTable = { i: number; rows: string[][] };

type DocxPreview = {
  type: "docx";
  name: string;
  paragraphs: DocxParagraph[];
  tables: DocxTable[];
};

type Jump =
  | { kind: "para"; paraStart: number; paraEnd?: number | null }
  | { kind: "table"; tableIndex: number };

function toPreviewUrl(fileUrl: string) {
  return fileUrl.replace("/api/pdf/", "/api/doc/") + "/preview";
}

export default function DocxViewer({
  fileUrl,
  jump,
}: {
  fileUrl: string;
  jump?: Jump;
}) {
  const [data, setData] = useState<DocxPreview | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const previewUrl = useMemo(() => toPreviewUrl(fileUrl), [fileUrl]);

  useEffect(() => {
    let cancelled = false;

    async function run() {
      setErr(null);
      setData(null);

      try {
        const res = await fetch(previewUrl);
        if (!res.ok) throw new Error(`Preview error: ${res.status}`);
        const j = (await res.json()) as DocxPreview;
        if (!cancelled) setData(j);
      } catch (e: any) {
        if (!cancelled) setErr(e?.message || "Gagal memuat DOCX preview");
      }
    }

    run();
    return () => {
      cancelled = true;
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!data || !jump) return;

    let targetEl: HTMLElement | null = null;

    if (jump.kind === "para") {
      targetEl = document.getElementById(`docx-para-${jump.paraStart}`);
    } else {
      targetEl = document.getElementById(`docx-table-${jump.tableIndex}`);
    }

    if (targetEl) {
      targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [data, jump]);

  const isParagraphHighlighted = (idx: number) => {
    if (!jump || jump.kind !== "para") return false;
    const end = Number.isFinite(Number(jump.paraEnd)) ? Number(jump.paraEnd) : jump.paraStart;
    return idx >= jump.paraStart && idx <= end;
  };

  const isTableHighlighted = (idx: number) => {
    return !!jump && jump.kind === "table" && idx === jump.tableIndex;
  };

  if (err) {
    return (
      <div style={{ padding: 12, color: "#b00020" }}>
        Gagal memuat DOCX: {err}
      </div>
    );
  }

  if (!data) {
    return <div style={{ padding: 12 }}>Memuat DOCX…</div>;
  }

  return (
    <div
      ref={wrapRef}
      style={{
        height: "100%",
        width: "100%",
        overflow: "auto",
        padding: 16,
        background: "#ffffff",
        color: "#111111",
      }}
    >
      <div style={{ fontWeight: 700, marginBottom: 12 }}>{data.name}</div>

      <div style={{ marginBottom: 18 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Paragraf</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {data.paragraphs?.map((p) => {
            const active = isParagraphHighlighted(p.i);
            return (
              <div
                key={p.i}
                id={`docx-para-${p.i}`}
                style={{
                  padding: 10,
                  border: active ? "2px solid rgba(77,190,20,0.98)" : "1px solid #eee",
                  borderRadius: 10,
                  lineHeight: 1.5,
                  whiteSpace: "pre-wrap",
                  background: active ? "rgba(77,190,20,0.12)" : "#ffffff",
                  color: "#111111",
                  boxShadow: active ? "0 0 0 2px rgba(77,190,20,0.08)" : undefined,
                }}
              >
                <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 6 }}>
                  Paragraf {p.i}
                </div>
                <div>{p.text}</div>
              </div>
            );
          })}
        </div>
      </div>

      <div>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Tabel</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {data.tables?.map((t) => {
            const active = isTableHighlighted(t.i);
            return (
              <div
                key={t.i}
                id={`docx-table-${t.i}`}
                style={{
                  padding: 10,
                  border: active ? "2px solid rgba(77,190,20,0.98)" : "1px solid #eee",
                  borderRadius: 10,
                  background: active ? "rgba(77,190,20,0.12)" : "#ffffff",
                  color: "#111111",
                  boxShadow: active ? "0 0 0 2px rgba(77,190,20,0.08)" : undefined,
                }}
              >
                <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 8 }}>
                  Tabel {t.i}
                </div>

                <div style={{ overflowX: "auto" }}>
                  <table style={{ borderCollapse: "collapse", width: "100%" }}>
                    <tbody>
                      {t.rows?.map((row, ri) => (
                        <tr key={ri}>
                          {row.map((cell, ci) => (
                            <td
                              key={ci}
                              style={{
                                border: "1px solid #e6e6e6",
                                padding: "6px 8px",
                                fontSize: 13,
                                verticalAlign: "top",
                                whiteSpace: "pre-wrap",
                                color: "#111111",
                                background: active ? "rgba(255,255,255,0.84)" : "#ffffff",
                              }}
                            >
                              {cell || ""}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })}

          {!data.tables?.length ? (
            <div style={{ opacity: 0.7 }}>Tidak ada tabel.</div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

