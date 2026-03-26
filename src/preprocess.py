"""
Preprocess pipeline: load → reassemble → filter → format → upsample → tokenize/pack → save.

Data: alwaysgood/ko-news-split-512 (hk, mk, korea_bank, naver)

Usage:
  python -m src.preprocess --config configs/stage1.yaml        # Qwen3.5
  python -m src.preprocess --config configs/stage1_gemma.yaml  # Gemma
"""

import sys
import random
import argparse
from pathlib import Path

import yaml
import numpy as np
from datasets import Dataset, load_dataset

from src.config import (
    ROOT, HK_DATA, MK_DATA, KOREA_BANK_DATA, NAVER_DATA,
    DATA_PROCESSED, NAVER_EXCLUDE_CATEGORIES, SEED,
    HF_KOREAN_MONO,
)
from src.utils import (
    load_jsonl, reassemble_news_chunks, reassemble_naver_chunks,
    clean_text, deduplicate, fmt_news, fmt_glossary,
)


# ── Loaders ──

def load_from_hub():
    print("=" * 60)
    print("1. Loading data (HF Hub)")
    print("=" * 60)

    repo = HF_KOREAN_MONO

    hk_ds = load_dataset("json", data_files=f"hf://datasets/{repo}/hk.jsonl", split="train")
    hk = reassemble_news_chunks(hk_ds.to_list())
    print(f"  hk: {len(hk_ds)} chunks → {len(hk)} articles")

    mk_ds = load_dataset("json", data_files=f"hf://datasets/{repo}/mk.jsonl", split="train")
    mk = reassemble_news_chunks(mk_ds.to_list())
    print(f"  mk: {len(mk_ds)} chunks → {len(mk)} articles")

    kb_ds = load_dataset("json", data_files=f"hf://datasets/{repo}/korea-bank-700-cleaned.jsonl", split="train")
    kb = [{"title": r["term"], "content": r["text"], "source": "korea_bank",
           "category": ",".join(r.get("categories", []) or [])} for r in kb_ds.to_list()]
    print(f"  korea_bank: {len(kb)} terms")

    nv_ds = load_dataset("json", data_files=f"hf://datasets/{repo}/naver_terms_clean.jsonl", split="train")
    nv = reassemble_naver_chunks(nv_ds.to_list())
    print(f"  naver: {len(nv_ds)} chunks → {len(nv)} terms")

    return hk, mk, kb, nv


def load_from_local():
    print("=" * 60)
    print("1. Loading data (local)")
    print("=" * 60)

    hk = reassemble_news_chunks(load_jsonl(HK_DATA))
    mk = reassemble_news_chunks(load_jsonl(MK_DATA))
    kb = [{"title": r["term"], "content": r["text"], "source": "korea_bank",
           "category": ",".join(r.get("categories", []))} for r in load_jsonl(KOREA_BANK_DATA)]
    nv = reassemble_naver_chunks(load_jsonl(NAVER_DATA))

    print(f"  hk: {len(hk)} | mk: {len(mk)} | kb: {len(kb)} | naver: {len(nv)}")
    return hk, mk, kb, nv


# ── Processing ──

def filter_domain(hk, mk, kb, nv):
    before = len(nv)
    nv = [d for d in nv if d.get("category") not in NAVER_EXCLUDE_CATEGORIES]
    print(f"  naver domain filter: {before} → {len(nv)}")
    return hk, mk, kb, nv


def clean_and_filter(docs, name, min_chars=100):
    for d in docs:
        d["content"] = clean_text(d["content"])
        if "title" in d:
            d["title"] = clean_text(d["title"])
    docs = [d for d in docs if len(d["content"]) >= min_chars]
    docs = deduplicate(docs)
    print(f"  {name}: {len(docs)} docs after clean+dedup")
    return docs


def format_all(hk, mk, kb, nv):
    out = []
    for d in hk: out.append({"text": fmt_news(d), "source": "hk"})
    for d in mk: out.append({"text": fmt_news(d), "source": "mk"})
    for d in kb: out.append({"text": fmt_glossary(d), "source": "korea_bank"})
    for d in nv: out.append({"text": fmt_glossary(d), "source": "naver"})

    for src in ["hk", "mk", "korea_bank", "naver"]:
        n = sum(1 for d in out if d["source"] == src)
        chars = sum(len(d["text"]) for d in out if d["source"] == src)
        print(f"  {src}: {n} docs, {chars:,} chars")
    return out


def upsample(docs, weights):
    out = []
    for d in docs:
        out.extend([d] * weights.get(d["source"], 1))
    print(f"  upsampled: {len(docs)} → {len(out)}")
    return out


def tokenize_and_pack(documents, tokenizer, max_len):
    eos = tokenizer.eos_token_id
    tok_docs = []
    for d in documents:
        ids = tokenizer.encode(d["text"], add_special_tokens=False)
        ids.append(eos)
        tok_docs.append(ids)

    total = sum(len(d) for d in tok_docs)
    print(f"  total tokens: {total:,}")

    packed = []
    cur_ids = []

    for doc_tokens in tok_docs:
        while len(cur_ids) + len(doc_tokens) > max_len:
            rem = max_len - len(cur_ids)
            if rem > 0:
                cur_ids.extend(doc_tokens[:rem])
                doc_tokens = doc_tokens[rem:]
            packed.append({
                "input_ids": cur_ids,
                "labels": cur_ids.copy(),
            })
            cur_ids = []

        cur_ids.extend(doc_tokens)

    print(f"  packed: {len(packed)} seqs × {max_len} tokens")
    return packed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "stage1.yaml"))
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    max_seq_len = cfg["model"]["max_seq_length"]
    output_dir = str(ROOT / cfg["data"]["train_path"]).rsplit("/train", 1)[0]
    up = cfg.get("upsampling", {})

    random.seed(SEED)
    np.random.seed(SEED)

    # Load
    hk, mk, kb, nv = load_from_local() if args.local else load_from_hub()

    # Filter
    hk, mk, kb, nv = filter_domain(hk, mk, kb, nv)

    print("\n  Cleaning...")
    hk = clean_and_filter(hk, "hk")
    mk = clean_and_filter(mk, "mk")
    kb = clean_and_filter(kb, "korea_bank")
    nv = clean_and_filter(nv, "naver")

    # Format + upsample
    formatted = format_all(hk, mk, kb, nv)
    weights = {"hk": up.get("hk", 1), "mk": up.get("mk", 1),
               "korea_bank": up.get("korea_bank", 1), "naver": up.get("naver", 1)}
    upsampled = upsample(formatted, weights)
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
