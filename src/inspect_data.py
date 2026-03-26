"""
Inspect packed training data for anomalies.

Usage:
  # Check samples around step 280-320 (where grad exploded)
  python -m src.inspect_data --start_step 270 --end_step 330

  # Scan entire dataset for anomalies
  python -m src.inspect_data --scan
"""

import argparse
from collections import Counter

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

from src.config import ROOT, BASE_MODEL


def get_sample_range(start_step, end_step, batch_size=8, grad_accum=2):
    samples_per_step = batch_size * grad_accum
    start_idx = start_step * samples_per_step
    end_idx = end_step * samples_per_step
    return start_idx, end_idx


def analyze_sequence(input_ids, tokenizer, idx):
    """Analyze a single packed sequence for anomalies."""
    total = len(input_ids)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    eos_count = input_ids.count(eos_id)
    pad_count = input_ids.count(pad_id) if pad_id is not None else 0

    # token frequency — check for excessive repetition
    counter = Counter(input_ids)
    most_common_id, most_common_count = counter.most_common(1)[0]
    repeat_ratio = most_common_count / total

    # unique token ratio
    unique_ratio = len(counter) / total

    issues = []
    if pad_count / total > 0.3:
        issues.append(f"HIGH_PAD ({pad_count}/{total} = {pad_count/total:.1%})")
    if repeat_ratio > 0.15:
        tok = tokenizer.decode([most_common_id])
        issues.append(f"REPETITIVE (token '{tok}' appears {most_common_count}x = {repeat_ratio:.1%})")
    if unique_ratio < 0.1:
        issues.append(f"LOW_DIVERSITY (unique ratio {unique_ratio:.1%})")
    if eos_count > 20:
        issues.append(f"MANY_EOS ({eos_count} EOS tokens — very short docs packed)")

    return {
        "idx": idx,
        "total": total,
        "eos_count": eos_count,
        "pad_count": pad_count,
        "repeat_ratio": repeat_ratio,
        "unique_ratio": unique_ratio,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_step", type=int, default=270)
    parser.add_argument("--end_step", type=int, default=330)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--scan", action="store_true", help="Scan entire dataset")
    parser.add_argument("--show_text", action="store_true", help="Print decoded text for flagged samples")
    parser.add_argument("--dump", type=int, default=0, help="Dump N samples as decoded text (regardless of flags)")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("Loading train dataset...")
    ds = load_from_disk(str(ROOT / "data" / "processed" / "train"))

    if args.scan:
        start_idx, end_idx = 0, len(ds)
        print(f"Scanning all {len(ds)} samples...")
    else:
        start_idx, end_idx = get_sample_range(
            args.start_step, args.end_step, args.batch_size, args.grad_accum
        )
        print(f"Steps {args.start_step}-{args.end_step} → samples {start_idx}-{end_idx}")

    end_idx = min(end_idx, len(ds))

    # --dump: 해당 구간 샘플을 직접 출력
    if args.dump > 0:
        for i in range(start_idx, min(start_idx + args.dump, end_idx)):
            text = tokenizer.decode(ds[i]["input_ids"])
            print(f"\n{'='*60}")
            print(f"[Sample {i}] (len={len(ds[i]['input_ids'])})")
            print(f"{'='*60}")
            print(text[:500])
            print("..." if len(text) > 500 else "")
        return

    flagged = []
    for i in range(start_idx, end_idx):
        input_ids = ds[i]["input_ids"]
        result = analyze_sequence(input_ids, tokenizer, i)
        if result["issues"]:
            flagged.append(result)

    print(f"\n{'='*60}")
    print(f"Scanned {end_idx - start_idx} samples, found {len(flagged)} flagged")
    print(f"{'='*60}\n")

    for f in flagged:
        print(f"  [{f['idx']}] {', '.join(f['issues'])}")
        print(f"         eos={f['eos_count']}, pad={f['pad_count']}, "
              f"repeat={f['repeat_ratio']:.1%}, unique={f['unique_ratio']:.1%}")
        if args.show_text:
            text = tokenizer.decode(ds[f["idx"]]["input_ids"][:300])
            print(f"         text: {text[:200]}...")
        print()

    if not flagged:
        print("  No anomalies found in this range.")

    # Summary stats
    print(f"\n{'='*60}")
    print("Dataset-wide stats (sampled range):")
    eos_counts = []
    for i in range(start_idx, min(end_idx, len(ds))):
        eos_counts.append(ds[i]["input_ids"].count(tokenizer.eos_token_id))
    eos_arr = np.array(eos_counts)
    print(f"  EOS per seq: mean={eos_arr.mean():.1f}, max={eos_arr.max()}, min={eos_arr.min()}")


if __name__ == "__main__":
    main()
