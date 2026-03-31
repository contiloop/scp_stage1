"""Loss spike가 발생한 step의 데이터를 디코딩해서 패턴을 찾는 스크립트."""
import json
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path


DATA_PATH = Path("data/processed/train")
MODEL_NAME = "unsloth/Qwen3.5-4B-Base"
BATCH_SIZE = 16
GRAD_ACCUM = 2
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM  # 32

SPIKE_STEPS = [
    310, 430, 480, 680, 960, 1010, 1080, 1210,
    1240, 1370, 1430, 1650, 1800, 1900, 2080,
    2110, 2160, 2350
]


def analyze_step(args):
    step, dataset_path, model_name = args
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    ds = load_from_disk(str(dataset_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    start_idx = step * EFFECTIVE_BATCH
    end_idx = start_idx + EFFECTIVE_BATCH

    if end_idx > len(ds):
        return {"step": step, "error": f"out of range (max {len(ds)})"}

    results = {"step": step, "sample_range": f"{start_idx}-{end_idx-1}", "samples": []}

    for i in range(start_idx, end_idx):
        ids = ds[i]["input_ids"]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        eos_count = ids.count(tokenizer.eos_token_id)
        preview = text[:200].replace("\n", " ")
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
        ko_ratio = korean_chars / max(len(text), 1)

        results["samples"].append({
            "idx": i, "length": len(ids), "eos_count": eos_count,
            "ko_ratio": round(ko_ratio, 2), "preview": preview,
            "full_text": text
        })

    avg_eos = sum(s["eos_count"] for s in results["samples"]) / len(results["samples"])
    avg_ko = sum(s["ko_ratio"] for s in results["samples"]) / len(results["samples"])
    results["summary"] = {"avg_docs_per_seq": round(avg_eos, 1), "avg_ko_ratio": round(avg_ko, 2)}
    return results


def main():
    # 데이터 없으면 전처리
    if not DATA_PATH.exists():
        print("=== 데이터 전처리 중 ===")
        subprocess.run(
            [sys.executable, "-m", "src.preprocess", "--config", "configs/stage1.yaml"],
            check=True,
        )

    print(f"CPUs: {cpu_count()}, analyzing {len(SPIKE_STEPS)} spike steps")
    print(f"Each step = {EFFECTIVE_BATCH} samples (batch={BATCH_SIZE} x accum={GRAD_ACCUM})")

    args = [(step, DATA_PATH, MODEL_NAME) for step in SPIKE_STEPS]

    with Pool(min(cpu_count(), len(SPIKE_STEPS))) as pool:
        results = pool.map(analyze_step, args)

    with open("spike_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'Step':>6} | {'Samples':>12} | {'Avg Docs/Seq':>12} | {'KO Ratio':>8}")
    print("-" * 50)
    for r in results:
        if "error" in r:
            print(f"{r['step']:>6} | ERROR: {r['error']}")
        else:
            s = r["summary"]
            print(f"{r['step']:>6} | {r['sample_range']:>12} | {s['avg_docs_per_seq']:>12} | {s['avg_ko_ratio']:>8}")

    ko_ratios = [r["summary"]["avg_ko_ratio"] for r in results if "summary" in r]
    doc_counts = [r["summary"]["avg_docs_per_seq"] for r in results if "summary" in r]
    print(f"\nKO ratio 평균: {sum(ko_ratios)/len(ko_ratios):.2f}")
    print(f"Docs/seq 평균: {sum(doc_counts)/len(doc_counts):.1f}")
    print(f"상세 결과: spike_analysis.json")


if __name__ == "__main__":
    main()
