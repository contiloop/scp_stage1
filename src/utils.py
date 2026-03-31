"""Data loading, chunk reassembly, and text cleaning utilities."""

import json
import hashlib
import html
import re
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reassemble_news_chunks(records: list[dict]) -> list[dict]:
    """Group hk/mk news chunks by URL → sort by chunk_id → join content."""
    groups = defaultdict(list)
    for r in records:
        groups[r["url"]].append(r)

    docs = []
    for url, chunks in groups.items():
        chunks.sort(key=lambda x: x["chunk_id"])
        meta = chunks[0]
        docs.append({
            "title": meta["title"],
            "content": "\n".join(c["content"] for c in chunks),
            "source": meta["site_name"],
            "category": meta["category"],
            "date": meta.get("date", ""),
            "original_char_count": meta.get("original_char_count", 0),
        })
    return docs


def reassemble_naver_chunks(records: list[dict]) -> list[dict]:
    """Group naver term chunks. Supports both HF Hub (flat) and local (nested) formats."""
    groups = defaultdict(list)
    for r in records:
        if "meta" in r:
            key = (r["meta"]["title"], r["meta"]["url"])
        else:
            key = (r.get("title", ""), r.get("url", ""))
        groups[key].append(r)

    docs = []
    for (title, _url), chunks in groups.items():
        if "meta" in chunks[0]:
            chunks.sort(key=lambda x: x["meta"]["chunk"])
            text = "\n".join(c["text"] for c in chunks)
            category = chunks[0]["meta"].get("category", "")
        else:
            chunks.sort(key=lambda x: x.get("chunk_id", 0))
            text = "\n".join(c["text"] for c in chunks)
            cats = chunks[0].get("categories", [])
            category = cats[0] if cats else ""

        docs.append({
            "title": title, "content": text,
            "source": "naver_terms", "category": category,
        })
    return docs


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def deduplicate(docs: list[dict], key: str = "content") -> list[dict]:
    seen, out = set(), []
    for d in docs:
        h = hashlib.sha256(d[key].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(d)
    return out


# ── Formatters ──

def fmt_news(doc: dict) -> str:
    return f"{doc['title']}\n\n{doc['content']}"

def fmt_glossary(doc: dict) -> str:
    title = doc.get("title", "").strip()
    content = doc.get("content", "").strip()
    return content if not title else f"{title}: {content}"

def fmt_earnings_call(doc: dict) -> str:
    return doc["text"].strip()
