# -*- coding: utf-8 -*-
"""RAG v0 for the SHM assistant.

No API key required. Builds a TF-IDF index over local PDFs, retrieves
page-anchored chunks, and converts a chunk into a claim–evidence `literature`
or `standard` evidence object.

Usage:
    python llm/rag.py build
    python llm/rag.py query "t85 definition" -k 5
    python llm/rag.py query "threshold rule" --source literature
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from datetime import datetime
from typing import Any

import pdfplumber

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INDEX = os.path.join(HERE, "rag_index.pkl")

DEFAULT_DIRS = [
    os.path.join(ROOT, "docs", "refs"),
    os.path.join(ROOT, "docs", "iwshm2025", "pdfs"),
]
STANDARD_DIRS = [os.path.join(ROOT, "docs", "standards")]
LOCAL_PAPERS = [
    os.path.join(ROOT, "docs", "IWSHM2025_template_latex", "IWSHM2027_paper.pdf"),
    os.path.join(ROOT, "docs", "icre2027", "paper", "icre2027_paper.pdf"),
]


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunks(text: str, size: int = 800, overlap: int = 120) -> list[tuple[int, int, str]]:
    text = _clean(text)
    if len(text) < 80:
        return []
    out = []
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        end = min(len(text), start + size)
        piece = text[start:end].strip()
        if len(piece) >= 80:
            out.append((start, end, piece))
        if end >= len(text):
            break
    return out


def _page_text(page) -> str:
    """Prefer column-wise extraction for two-column PDFs."""
    try:
        w, h = float(page.width), float(page.height)
        if w > 450:
            left = page.crop((0, 0, w / 2, h)).extract_text() or ""
            right = page.crop((w / 2, 0, w, h)).extract_text() or ""
            if left.strip() and right.strip():
                return left + "\n" + right
        return page.extract_text() or ""
    except Exception:
        return ""


def _pdf_chunks(path: str, source_type: str) -> list[dict[str, Any]]:
    title = os.path.splitext(os.path.basename(path))[0]
    out = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            try:
                text = _page_text(page)
            except Exception:
                continue
            for a, b, piece in _chunks(text):
                out.append({
                    "doc": os.path.relpath(path, ROOT),
                    "title": title,
                    "page": pno,
                    "char_start": a,
                    "char_end": b,
                    "text": piece,
                    "source_type": source_type,
                })
    return out


def collect(dirs: list[str], source_type: str) -> list[dict[str, Any]]:
    chunks = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(".pdf"):
                continue
            try:
                chunks.extend(_pdf_chunks(os.path.join(d, name), source_type))
            except Exception as exc:
                print(f"[skip] {name}: {exc}", file=sys.stderr)
    return chunks


def collect_files(files: list[str], source_type: str) -> list[dict[str, Any]]:
    chunks = []
    for path in files:
        if os.path.exists(path):
            try:
                chunks.extend(_pdf_chunks(path, source_type))
            except Exception as exc:
                print(f"[skip] {path}: {exc}", file=sys.stderr)
    return chunks


def build(index_path: str = INDEX) -> dict[str, Any]:
    chunks = collect(DEFAULT_DIRS, "literature")
    chunks += collect_files(LOCAL_PAPERS, "literature")
    # Standards are a separate provenance class; only real PDFs are indexed.
    chunks += collect(STANDARD_DIRS, "standard")
    if not chunks:
        raise SystemExit("no PDF chunks found; check docs/refs and docs/iwshm2025/pdfs")

    texts = [c["text"] for c in chunks]
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=120000,
        sublinear_tf=True,
    )
    mat = vec.fit_transform(texts)
    with open(index_path, "wb") as fh:
        pickle.dump({"chunks": chunks, "vectorizer": vec, "matrix": mat}, fh)
    print(f"indexed {len(chunks)} chunks from {len({c['doc'] for c in chunks})} PDFs")
    print(f"saved {index_path}")
    return {"n_chunks": len(chunks), "n_docs": len({c["doc"] for c in chunks})}


def load(index_path: str = INDEX) -> dict[str, Any]:
    if not os.path.exists(index_path):
        raise SystemExit(f"index not found: {index_path}; run `python llm/rag.py build`")
    with open(index_path, "rb") as fh:
        return pickle.load(fh)


def retrieve(query: str, k: int = 5, source_type: str | None = None,
             index_path: str = INDEX) -> list[dict[str, Any]]:
    db = load(index_path)
    q = db["vectorizer"].transform([query])
    sim = cosine_similarity(q, db["matrix"]).ravel()
    order = sim.argsort()[::-1]
    out = []
    for i in order:
        c = db["chunks"][int(i)]
        if source_type and c["source_type"] != source_type:
            continue
        out.append({**c, "score": float(sim[int(i)])})
        if len(out) >= k:
            break
    return out


def to_evidence(chunk: dict[str, Any], evidence_id: str) -> dict[str, Any]:
    kind = "standard" if chunk["source_type"] == "standard" else "paper"
    return {
        "evidence_id": evidence_id,
        "evidence_type": "standard" if kind == "standard" else "literature",
        "source": {
            "kind": kind,
            "title": chunk["title"],
            "tool_name": None,
            "version": None,
            "authors": [],
            "year": None,
            "venue": None,
            "doi": None,
            "url": None,
            "clause": None,
        },
        "content": {"chunk_id": chunk.get("chunk_id", ""), "score": chunk.get("score")},
        "quote": chunk["text"],
        "locator": {
            "file": chunk["doc"],
            "page": chunk["page"],
            "section": None,
            "table": None,
            "figure": None,
            "row": None,
            "clause": None,
            "char_start": chunk["char_start"],
            "char_end": chunk["char_end"],
        },
        "scope": {
            "gid": "ALL",
            "campaign": None,
            "modality": "MULTI",
            "cycle_min": None,
            "cycle_max": None,
            "lifetime_min": None,
            "lifetime_max": None,
        },
        "observed_lifetime_fraction": None,
        "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "content_hash": None,
        "reliability": "secondary",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--index", default=INDEX)
    q = sub.add_parser("query")
    q.add_argument("query")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--source", choices=["literature", "standard"], default=None)
    q.add_argument("--json", action="store_true")
    q.add_argument("--index", default=INDEX)
    args = ap.parse_args(argv)

    if args.cmd == "build":
        build(args.index)
        return 0

    hits = retrieve(args.query, args.k, args.source, args.index)
    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
    else:
        for i, h in enumerate(hits, 1):
            print(f"[{i}] score={h['score']:.3f} {h['doc']} p.{h['page']}")
            print("    " + h["text"][:240].replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
