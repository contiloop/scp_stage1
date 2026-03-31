"""
Preprocess pipeline: load → reassemble → filter → format → upsample → tokenize/pack → save.

Data sources are declared in the YAML config under `sources:`.
Adding a new source = adding an entry to the config. No code changes needed.

Usage:
  python -m src.preprocess --config configs/stage1.yaml        # Qwen3.5
  python -m src.preprocess --config configs/stage1_gemma.yaml  # Gemma
"""

import random
import argparse
from pathlib import Path

import yaml
import numpy as np
from datasets import Dataset, load_dataset

from src.config import ROOT, SEED
from src.utils import (
    reassemble_news_chunks, reassemble_naver_chunks,
    clean_text, deduplicate, fmt_news, fmt_glossary,
)

FORMATTERS = {
    "news": fmt_news,
    "glossary": fmt_glossary,
}

REASSEMBLERS = {
    "news_chunks": reassemble_news_chunks,
    "naver_chunks": reassemble_naver_chunks,
}


def load_source(src_cfg):
    """Load a single data source from HF Hub based on its config."""
    name = src_cfg["name"]
    repo = src_cfg["repo"]
    file = src_cfg["file"]

    ds = load_dataset("json", data_files=f"hf://datasets/{repo}/{file}", split="train")
    records = ds.to_list()

    # Reassemble chunked docs if needed
    reassemble = src_cfg.get("reassemble")
    if reassemble:
        docs = REASSEMBLERS[reassemble](records)
        print(f"  {name}: {len(records)} chunks → {len(docs)} docs")
    else:
        # Direct field mapping
        field_map = src_cfg.get("fields", {"title": "title", "content": "content"})
        docs = [{"title": r[field_map["title"]],
                 "content": r[field_map["content"]],
                 "source": name}
                for r in records]
        print(f"  {name}: {len(docs)} docs")

    for d in docs:
        d["source"] = name
    return docs


def filter_docs(docs, src_cfg):
    """Apply source-specific filters (e.g. exclude_categories)."""
    exclude = src_cfg.get("exclude_categories")
    if exclude:
        exclude_set = set(exclude)
        before = len(docs)
        docs = [d for d in docs if d.get("category") not in exclude_set]
        print(f"  {src_cfg['name']} category filter: {before} → {len(docs)}")
    return docs


def clean_and_filter(docs, name, min_chars=100, dedup=True):
    for d in docs:
        d["content"] = clean_text(d["content"])
        if "title" in d:
            d["title"] = clean_text(d["title"])
    docs = [d for d in docs if len(d["content"]) >= min_chars]
    if dedup:
        docs = deduplicate(docs)
    print(f"  {name}: {len(docs)} docs after clean" + ("+dedup" if dedup else ""))
    return docs


def format_docs(docs, fmt_name):
    formatter = FORMATTERS[fmt_name]
    return [{"text": formatter(d), "source": d["source"]} for d in docs]


def upsample(docs, weights):
    out = []
    for d in docs:
        out.extend([d] * weights.get(d["source"], 1))
    print(f"  upsampled: {len(docs)} → {len(out)}")
    return out


def _find_sentence_boundary(tokens, tokenizer):
    """토큰 리스트에서 마지막 완전한 문장 경계 위치를 찾는다.
    "." 기준으로 디코딩하여 마지막 마침표 위치를 반환.
    못 찾으면 0 반환 (아무것도 넣지 않음).
    """
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    # 마지막 문장 종결 부호 찾기
    last_boundary = -1
    for sep in [".", "。", "!", "?", "!}", "?}"]:
        idx = text.rfind(sep)
        if idx > last_boundary:
            last_boundary = idx

    if last_boundary <= 0:
        return 0

    # 문장 경계까지의 텍스트를 다시 토크나이즈해서 토큰 수 계산
    boundary_text = text[:last_boundary + 1]
    boundary_tokens = tokenizer.encode(boundary_text, add_special_tokens=False)
    return len(boundary_tokens)


def _sentence_fit(tokens, tokenizer, limit):
    """Return the largest prefix <= limit that ends at a sentence boundary."""
    if limit <= 0:
        return 0

    fit = _find_sentence_boundary(tokens[:limit], tokenizer)
    if fit > 0:
        return fit
    return min(len(tokens), limit)


def _sort_docs_for_packing(tok_docs, pool_size=4096):
    """Group similarly sized docs together within random pools to reduce padding."""
    if pool_size <= 1:
        return tok_docs

    ordered = []
    for start in range(0, len(tok_docs), pool_size):
        pool = tok_docs[start:start + pool_size]
        pool.sort(key=len, reverse=True)
        ordered.extend(pool)
    return ordered


def tokenize_and_pack(documents, tokenizer, max_len, pool_size=4096):
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    tok_docs = []
    for d in documents:
        ids = tokenizer.encode(d["text"], add_special_tokens=False)
        ids.append(eos)
        tok_docs.append(ids)

    total = sum(len(d) for d in tok_docs)
    print(f"  total tokens: {total:,}")
    tok_docs = _sort_docs_for_packing(tok_docs, pool_size=pool_size)

    packed = []
    cur_ids = []
    cur_labels = []
    total_padded = 0

    def flush_current():
        nonlocal cur_ids, cur_labels, total_padded
        if not cur_ids:
            return

        pad_len = max_len - len(cur_ids)
        total_padded += pad_len
        packed.append({
            "input_ids": cur_ids + [pad] * pad_len,
            "labels": cur_labels + [-100] * pad_len,
        })
        cur_ids = []
        cur_labels = []

    for doc_tokens in tok_docs:
        remaining = list(doc_tokens)

        while remaining:
            rem = max_len - len(cur_ids)

            if len(remaining) <= rem:
                cur_ids.extend(remaining)
                cur_labels.extend(remaining)
                break

            if rem > 0:
                fit = _find_sentence_boundary(remaining[:rem], tokenizer)
                if fit > 0:
                    cur_ids.extend(remaining[:fit])
                    cur_labels.extend(remaining[:fit])
                    remaining = remaining[fit:]
                    if remaining and remaining[0] == eos and len(cur_ids) < max_len:
                        cur_ids.append(remaining[0])
                        cur_labels.append(remaining[0])
                        remaining = remaining[1:]
                    flush_current()
                    continue

            if cur_ids:
                flush_current()
                continue

            fit = _sentence_fit(remaining, tokenizer, max_len)
            cur_ids.extend(remaining[:fit])
            cur_labels.extend(remaining[:fit])
            remaining = remaining[fit:]
            if remaining and remaining[0] == eos and len(cur_ids) < max_len:
                cur_ids.append(remaining[0])
                cur_labels.append(remaining[0])
                remaining = remaining[1:]
            flush_current()

    # 마지막 시퀀스
    flush_current()

    total_tokens = len(packed) * max_len
    print(f"  packed: {len(packed)} seqs × {max_len} tokens")
    print(f"  padding: {total_padded:,} / {total_tokens:,} ({total_padded/total_tokens*100:.1f}%)")
    return packed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "stage1.yaml"))
    parser.add_argument("--val_ratio", type=float, default=0.02)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    max_seq_len = cfg["model"]["max_seq_length"]
    output_dir = str(ROOT / cfg["data"]["train_path"]).rsplit("/train", 1)[0]

    # Load data sources from separate config
    data_config_path = ROOT / cfg["data"]["config"]
    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)
    sources_cfg = data_cfg["sources"]

    random.seed(SEED)
    np.random.seed(SEED)

    # Load → filter → clean → format per source
    print("=" * 60)
    print("1. Loading data (HF Hub)")
    print("=" * 60)

    all_formatted = []
    weights = {}

    for src_cfg in sources_cfg:
        name = src_cfg["name"]

        docs = load_source(src_cfg)
        docs = filter_docs(docs, src_cfg)
        docs = clean_and_filter(docs, name, dedup=src_cfg.get("dedup", True))
        formatted = format_docs(docs, src_cfg["format"])

        chars = sum(len(d["text"]) for d in formatted)
        print(f"  {name}: {len(formatted)} formatted, {chars:,} chars")

        all_formatted.extend(formatted)
        weights[name] = src_cfg.get("upsample", 1)

    # Upsample + shuffle
    upsampled = upsample(all_formatted, weights)
    random.shuffle(upsampled)

    # Tokenize
    print(f"\n  Loading tokenizer: {model_name}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    packed = tokenize_and_pack(upsampled, tokenizer, max_seq_len)

    # Split
    random.shuffle(packed)
    val_n = max(1, int(len(packed) * args.val_ratio))
    train_data, val_data = packed[val_n:], packed[:val_n]
    print(f"\n  train: {len(train_data)} | val: {len(val_data)}")

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_data).save_to_disk(str(out / "train"))
    Dataset.from_list(val_data).save_to_disk(str(out / "val"))
    print(f"  saved to {out}")


if __name__ == "__main__":
    main()
