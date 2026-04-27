"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";

pdfjs.GlobalWorkerOptions.workerSrc = "/pdfjs/pdf.worker.min.mjs";

export type PdfSelected = {
  page: number;
  bbox?: [number, number, number, number] | null;
  paragraph_bboxes?: [number, number, number, number][] | null;
  page_w?: number | null;
  page_h?: number | null;
};

export default function PdfViewer({
  pdfUrl,
  pageNumber,
  setPageNumber,
  selected,
  pageWidth = 860,
  onTotalPagesChange,
}: {
  pdfUrl: string;
  pageNumber: number;
  setPageNumber: (n: number) => void;
  selected: PdfSelected | null;
  pageWidth?: number;
  onTotalPagesChange?: (n: number) => void;
}) {
  const [numPages, setNumPages] = useState<number>(0);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const pageWrapRef = useRef<HTMLDivElement | null>(null);
  const [canvasSize, setCanvasSize] = useState<{ w: number; h: number } | null>(null);

  const lastCitationSyncRef = useRef<string | null>(null);
  const lastPdfUrlRef = useRef<string>("");

  const updateCanvasSize = () => {
    const el = pageWrapRef.current;
    if (!el) {
      setCanvasSize(null);
      return;
    }

    const canvas = el.querySelector("canvas") as HTMLCanvasElement | null;
    if (!canvas) {
      setCanvasSize(null);
      return;
    }

    const r = canvas.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) {
      setCanvasSize({ w: r.width, h: r.height });
    } else {
      setCanvasSize(null);
    }
  };

  useEffect(() => {
    if (lastPdfUrlRef.current !== pdfUrl) {
      lastPdfUrlRef.current = pdfUrl;
      lastCitationSyncRef.current = null;
      setCanvasSize(null);
      setNumPages(0);
    }
  }, [pdfUrl]);

  useEffect(() => {
    if (!selected?.page || !pdfUrl) return;

    const sig = JSON.stringify({
      pdfUrl,
      page: selected.page,
      bbox: selected.bbox ?? null,
      paragraph_bboxes: selected.paragraph_bboxes ?? null,
      page_w: selected.page_w ?? null,
      page_h: selected.page_h ?? null,
    });

    if (lastCitationSyncRef.current === sig) return;
    lastCitationSyncRef.current = sig;

    if (selected.page !== pageNumber) {
      setPageNumber(selected.page);
    }
  }, [selected, pdfUrl, pageNumber, setPageNumber]);

  const canHighlight =
    !!selected &&
    !!selected.page_w &&
    !!selected.page_h &&
    !!canvasSize &&
    selected.page === pageNumber;

  const highlightBoxes = useMemo(() => {
    if (!canHighlight || !selected || !selected.page_w || !selected.page_h || !canvasSize) {
      return [];
    }

    const sourceBoxes =
      Array.isArray(selected.paragraph_bboxes) && selected.paragraph_bboxes.length > 0
        ? selected.paragraph_bboxes
        : selected.bbox
        ? [selected.bbox]
        : [];

    if (!sourceBoxes.length) return [];

    const scaleX = canvasSize.w / selected.page_w;
    const scaleY = canvasSize.h / selected.page_h;

    const boxes = sourceBoxes
      .map((bbox) => {
        const [x0Raw, y0Raw, x1Raw, y1Raw] = bbox;

        const x0 = Math.min(x0Raw, x1Raw);
        const x1 = Math.max(x0Raw, x1Raw);
        const y0 = Math.min(y0Raw, y1Raw);
        const y1 = Math.max(y0Raw, y1Raw);

        const left = x0 * scaleX;
        const top = y0 * scaleY;
        const width = (x1 - x0) * scaleX;
        const height = (y1 - y0) * scaleY;

        const finite = [left, top, width, height].every((v) => Number.isFinite(v));
        if (!finite) return null;
        if (width <= 0 || height <= 0) return null;
        if (top < -4 || top > canvasSize.h + 4) return null;

        return { left, top, width, height };
      })
      .filter(Boolean) as { left: number; top: number; width: number; height: number }[];

    return boxes.sort((a, b) => a.top - b.top || a.left - b.left);
  }, [canHighlight, canvasSize, selected]);

  useEffect(() => {
    const el = pageWrapRef.current;
    if (!el) return;

    const update = () => {
      updateCanvasSize();
    };

    update();

    const ro = new ResizeObserver(update);
    ro.observe(el);

    return () => ro.disconnect();
  }, [pdfUrl, pageNumber, pageWidth]);

  useEffect(() => {
    setCanvasSize(null);
  }, [pageNumber, pdfUrl, pageWidth]);

  useEffect(() => {
    if (!highlightBoxes.length) return;

    const scroller = scrollRef.current;
    const wrap = pageWrapRef.current;
    if (!scroller || !wrap) return;

    const first = highlightBoxes[0];
    const last = highlightBoxes[highlightBoxes.length - 1];
    const centerY = (first.top + last.top + last.height) / 2;

    const run = () => {
      const wrapTop = wrap.offsetTop;
      const desired = wrapTop + centerY - scroller.clientHeight / 2;

      scroller.scrollTo({
        top: Math.max(0, desired),
        behavior: "smooth",
      });
    };

    const t = window.setTimeout(run, 80);
    return () => window.clearTimeout(t);
  }, [highlightBoxes]);

  useEffect(() => {
    if (!numPages) return;

    if (pageNumber > numPages) {
      setPageNumber(numPages);
      return;
    }

    if (pageNumber < 1) {
      setPageNumber(1);
    }
  }, [numPages, pageNumber, setPageNumber]);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", minHeight: 0 }}>
      <div ref={scrollRef} style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
        {!pdfUrl ? (
          <div style={{ padding: 20 }}>PDF viewer muncul setelah upload.</div>
        ) : (
          <div style={{ display: "flex", justifyContent: "center", padding: 10 }}>
            <Document
              key={pdfUrl}
              file={pdfUrl}
              onLoadSuccess={(info: any) => {
                const n = Number(info?.numPages || 0);
                const safePages = Number.isFinite(n) ? n : 0;

                setNumPages(safePages);
                onTotalPagesChange?.(safePages);

                if (safePages > 0) {
                  const safePage = Math.min(Math.max(pageNumber, 1), safePages);
                  if (safePage !== pageNumber) {
                    setPageNumber(safePage);
                  }
                }
              }}
              onLoadError={(err) => {
                console.error("PDF load error:", err);
                setNumPages(0);
                onTotalPagesChange?.(0);
              }}
            >
              <div ref={pageWrapRef} style={{ position: "relative" }}>
                <Page
                  key={`${pdfUrl}-${pageNumber}-${pageWidth}`}
                  pageNumber={pageNumber}
                  width={pageWidth}
                  renderTextLayer={false}
                  renderAnnotationLayer={false}
                  onRenderSuccess={() => {
                    window.requestAnimationFrame(() => {
                      updateCanvasSize();
                      window.setTimeout(updateCanvasSize, 50);
                    });
                  }}
                />

                {highlightBoxes.length > 0 ? (
                  <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
                    {highlightBoxes.map((box, idx) => (
                      <div
                        key={`${idx}-${box.left}-${box.top}-${box.width}-${box.height}`}
                        style={{
                          position: "absolute",
                          left: box.left,
                          top: box.top,
                          width: box.width,
                          height: box.height,
                          outline: "2px solid rgba(77,190,20,0.98)",
                          background: "rgba(77,190,20,0.18)",
                          borderRadius: 3,
                          boxShadow: idx === 0 ? "0 0 0 9999px rgba(77,190,20,0.04)" : undefined,
                        }}
                      />
                    ))}
                  </div>
                ) : null}
              </div>
            </Document>
          </div>
        )}
      </div>
    </div>
  );
}

