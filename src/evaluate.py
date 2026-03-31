"""
Evaluate: perplexity + domain completion + lm-eval-harness benchmarks.

Usage:
  # 전체 평가 (ppl + completion + benchmarks)
  make eval

  # lm-eval-harness만 (base vs CPT 비교)
  python -m src.evaluate --model_path checkpoints/stage1_cpt --base_model unsloth/Qwen3.5-4B-Base --benchmarks_only

  # perplexity만
  python -m src.evaluate --model_path checkpoints/stage1_cpt --skip_completion --skip_benchmarks
"""

import math
import json
import subprocess
import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm

from src.config import ROOT, BASE_MODEL, KOREA_BANK_DATA, MAX_SEQ_LEN
from src.utils import load_jsonl

# catastrophic forgetting 측정용 벤치마크
BENCHMARK_TASKS = "mmlu,hellaswag,arc_easy,arc_challenge,winogrande"
KOREAN_BENCHMARK_TASKS = "kmmlu,kobest_boolq,kobest_copa,kobest_hellaswag"


def _resolve_eval_dtype():
    """Match train-time precision policy as closely as possible for eval."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _model_load_kwargs():
    """Safer default loading kwargs for eval on mixed GPU environments."""
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": _resolve_eval_dtype(),
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        # Eval stability > absolute speed. Avoid backend-specific CUDA kernel issues first.
        kwargs["attn_implementation"] = "eager"
    return kwargs


def _load_tokenizer(path_or_repo: str):
    """Load tokenizer while tolerating the Mistral regex compatibility warning."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            path_or_repo,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(path_or_repo, trust_remote_code=True)


def _free_vram():
    """Force-free GPU memory between eval stages."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def compute_ppl(model, dataset, batch_size=4, max_batches=None):
    model.eval()
    total_loss, total_tok = 0.0, 0
    n = min(len(dataset), max_batches * batch_size) if max_batches else len(dataset)
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc="ppl"):
            batch = dataset[i:min(i + batch_size, n)]
            ids = torch.tensor(batch["input_ids"]).to(device)
            labels = torch.tensor(batch["labels"]).to(device)
            loss = model(input_ids=ids, labels=labels).loss
            toks = (labels != -100).sum().item()
            total_loss += loss.item() * toks
            total_tok += toks

    avg = total_loss / total_tok
    return {"loss": avg, "ppl": math.exp(avg), "tokens": total_tok}


def completion_test(model, tokenizer, n=20):
    if not Path(KOREA_BANK_DATA).exists():
        print(f"  [WARN] completion test skipped: {KOREA_BANK_DATA} not found")
        return []
    kb = load_jsonl(KOREA_BANK_DATA)[:n]
    model.eval()
    device = next(model.parameters()).device
    results = []

    for e in kb:
        prompt_len = max(20, len(e["text"]) // 3)
        prompt = f"{e['term']}: {e['text'][:prompt_len]}"
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=100, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        ref = e["text"][prompt_len:prompt_len + 100]

        results.append({"term": e["term"], "ref": ref[:80], "gen": gen[:80]})
        print(f"  [{e['term']}] {gen[:60]}...")

    return results


def _resolve_merged(model_path: str) -> Path:
    """adapter 경로에서 merged 모델 경로를 찾는다."""
    mp = Path(model_path)
    merged = mp.parent / f"{mp.name}_merged"
    if not merged.exists():
        merged = mp.parent / "stage1_merged"
    return merged if merged.exists() else None


def _ckpt_label(model_path: str) -> str:
    """체크포인트 경로에서 라벨 추출."""
    mp = Path(model_path)
    return mp.name if mp.name.startswith("checkpoint-") else "final"


def run_lm_eval(model_path: str, tasks: str = BENCHMARK_TASKS, batch_size: int = 8, limit: int = 400) -> dict:
    """lm-evaluation-harness로 벤치마크 실행."""
    label = _ckpt_label(model_path)
    out_dir = ROOT / "checkpoints" / "lm_eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # adapter면 merged 모델 우선 사용 (unsloth adapter는 lora_B=0)
    mp = Path(model_path)
    merged_path = _resolve_merged(model_path)
    if (mp / "adapter_config.json").exists() and merged_path:
        print(f"  using merged model for benchmark: {merged_path}")
        model_args = f"pretrained={merged_path},trust_remote_code=True"
    elif (mp / "adapter_config.json").exists():
        print("  [WARN] using adapter for benchmark (lora_B may be zero)")
        model_args = f"pretrained={BASE_MODEL},peft={model_path},trust_remote_code=True"
    else:
        model_args = f"pretrained={model_path},trust_remote_code=True"

    cpt_out = out_dir / f"cpt_{label}"
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--batch_size", str(batch_size),
        "--limit", str(limit),
        "--output_path", str(cpt_out),
    ]

    print(f"\n{'='*60}")
    print(f"lm-eval-harness: {tasks}")
    print(f"model: {model_path} ({label})")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  lm_eval error: {result.stderr[-500:]}")
        return {}, ""
    print(result.stdout[-1000:])

    # 결과 파일 읽기 (lm-eval saves as results_YYYY-MM-DD...json)
    results = {}
    for p in sorted(cpt_out.rglob("results_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        with open(p) as f:
            results = json.load(f)
        break
    return results, result.stdout


def run_benchmark_comparison(model_path: str, base_model: str, tasks: str = BENCHMARK_TASKS,
                             korean: bool = True, limit: int = 400):
    """base 모델과 CPT 모델의 벤치마크 점수 비교."""
    all_tasks = tasks
    if korean:
        all_tasks = f"{tasks},{KOREAN_BENCHMARK_TASKS}"

    print("\n" + "=" * 60)
    print(f"Catastrophic Forgetting Test (limit={limit} per task)")
    print(f"Tasks: {all_tasks}")
    print("=" * 60)

    # 1. base 모델 평가 (이미 결과 있으면 스킵)
    base_out = ROOT / "checkpoints" / "lm_eval_results" / "base"
    existing_base = list(base_out.rglob("results_*.json")) if base_out.exists() else []
    if existing_base:
        print("\n[1/2] Base model evaluation... (cached)")
    else:
        print("\n[1/2] Base model evaluation...")
        base_cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={base_model},trust_remote_code=True",
            "--tasks", all_tasks,
            "--batch_size", "8",
            "--limit", str(limit),
            "--output_path", str(base_out),
        ]
        r1 = subprocess.run(base_cmd, capture_output=True, text=True)
        if r1.returncode != 0:
            print(f"  base eval error: {r1.stderr[-500:]}")
            return
        print(r1.stdout[-500:])

    # 2. CPT 모델 평가
    print("\n[2/2] CPT model evaluation...")
    cpt_results, cpt_stdout = run_lm_eval(model_path, all_tasks, limit=limit)

    # 3. 결과 저장 (lm-eval stdout 표 포함)
    label = _ckpt_label(model_path)
    summary_path = ROOT / "checkpoints" / f"eval_summary_{label}.txt"
    with open(summary_path, "w") as f:
        f.write(f"=== Base Model: {base_model} ===\n")
        if existing_base:
            f.write("(cached from previous run)\n")
        else:
            f.write(r1.stdout)
        f.write(f"\n=== CPT Model: {model_path} ===\n")
        f.write(cpt_stdout)
    print(f"  summary: {summary_path}")


def _need_merge(model_path: str) -> bool:
    """merged 모델이 없고 adapter인 경우 merge 필요."""
    mp = Path(model_path)
    return (mp / "adapter_config.json").exists() and _resolve_merged(model_path) is None


def _auto_merge(model_path: str):
    """unsloth로 체크포인트를 merged 모델로 변환."""
    from unsloth import FastVisionModel
    mp = Path(model_path)
    out = mp.parent / f"{mp.name}_merged"
    print(f"  merging {mp.name}...")
    model, tok = FastVisionModel.from_pretrained(str(mp), max_seq_length=2048, dtype=None, load_in_4bit=False)
    model.save_pretrained_merged(str(out), tok, save_method="merged_16bit")
    del model; _free_vram()
    print(f"  merged: {out}")
    return out


def _eval_single(model_path: str, args, val_ds):
    """단일 체크포인트 eval. merge → eval → merged 삭제 (디스크 절약)."""
    from transformers import AutoModelForCausalLM

    mp = Path(model_path)
    label = _ckpt_label(model_path)
    auto_merged = False
    load_kwargs = _model_load_kwargs()

    print(f"\n{'='*60}")
    print(f"Evaluating: {mp.name} ({label})")
    print(f"{'='*60}")
    print(f"  eval dtype: {load_kwargs['torch_dtype']}")
    if torch.cuda.is_available():
        print(f"  eval attn: {load_kwargs.get('attn_implementation', 'auto')}")

    # merge if needed
    if _need_merge(model_path):
        try:
            _auto_merge(model_path)
            auto_merged = True
        except Exception as e:
            print(f"  [WARN] auto-merge failed: {e}")
            _free_vram()

    # load model
    merged_path = _resolve_merged(model_path)
    if merged_path:
        print(f"  loading merged model: {merged_path}")
        model = AutoModelForCausalLM.from_pretrained(str(merged_path), **load_kwargs)
        tokenizer = _load_tokenizer(str(merged_path))
    elif (mp / "adapter_config.json").exists():
        print("  [WARN] using adapter (lora_B may be zero with unsloth)")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)
        model = PeftModel.from_pretrained(base, str(mp))
        tokenizer = _load_tokenizer(str(mp))
    else:
        model = AutoModelForCausalLM.from_pretrained(str(mp), **load_kwargs)
        tokenizer = _load_tokenizer(str(mp))

    ft = compute_ppl(model, val_ds, args.batch_size, args.max_batches)
    print(f"\n  FT ppl: {ft['ppl']:.2f}")

    # completion test reuses the loaded FT model
    comp = []
    if not args.skip_completion:
        try:
            comp = completion_test(model, tokenizer)
        except (torch.cuda.OutOfMemoryError, Exception) as e:
            print(f"  [WARN] completion test skipped: {e}")
    del model; _free_vram()

    # base model ppl (only once, caller caches)
    base_m = None
    if args.base_model and not hasattr(args, '_base_ppl_cache'):
        try:
            bm = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
            base_m = compute_ppl(bm, val_ds, args.batch_size, args.max_batches)
            args._base_ppl_cache = base_m
            del bm; _free_vram()
        except (torch.cuda.OutOfMemoryError, Exception) as e:
            print(f"  [WARN] base model eval skipped: {e}")
            _free_vram()
    elif hasattr(args, '_base_ppl_cache'):
        base_m = args._base_ppl_cache

    if base_m:
        print(f"  Base ppl: {base_m['ppl']:.2f}")
        print(f"  Improvement: {(base_m['ppl'] - ft['ppl']) / base_m['ppl'] * 100:.1f}%")

    # save results
    out_path = ROOT / "checkpoints" / f"eval_results_{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"ft": ft, "base": base_m, "completions": comp}, f, ensure_ascii=False, indent=2)
    print(f"  results: {out_path}")

    # benchmarks
    _free_vram()
    if not args.skip_benchmarks:
        try:
            base_model = args.base_model or BASE_MODEL
            bench_path = str(merged_path) if merged_path else str(mp)
            run_benchmark_comparison(bench_path, base_model, args.benchmark_tasks)
        except Exception as e:
            print(f"  [WARN] benchmarks skipped: {e}")

    # cleanup auto-merged to save disk
    if auto_merged and merged_path and merged_path.exists():
        import shutil
        shutil.rmtree(merged_path)
        print(f"  cleaned up: {merged_path}")

    return {"label": label, "ft_ppl": ft["ppl"], "base_ppl": base_m["ppl"] if base_m else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", nargs="+", default=None, help="체크포인트 경로 (복수 가능)")
    parser.add_argument("--all_checkpoints", action="store_true", help="output_dir 안의 모든 체크포인트 eval")
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--skip_completion", action="store_true")
    parser.add_argument("--skip_benchmarks", action="store_true")
    parser.add_argument("--benchmarks_only", action="store_true")
    parser.add_argument("--benchmark_tasks", default=BENCHMARK_TASKS)
    parser.add_argument("--config", default=str(ROOT / "configs" / "stage1.yaml"))
    args = parser.parse_args()

    # --- benchmarks only mode ---
    if args.benchmarks_only:
        if not args.model_path:
            print("[ERROR] --model_path required for --benchmarks_only")
            return
        if not args.base_model:
            args.base_model = BASE_MODEL
        run_benchmark_comparison(args.model_path[0], args.base_model, args.benchmark_tasks)
        return

    # --- resolve checkpoint list ---
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["training"]["output_dir"])
    if args.all_checkpoints:
        paths = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if (out_dir / "adapter_config.json").exists() or (out_dir.parent / "stage1_merged").exists():
            paths.append(out_dir)  # final model
        if not paths:
            print(f"  [ERROR] no checkpoints found in {out_dir}")
            return
    elif args.model_path:
        paths = [Path(p) for p in args.model_path]
    else:
        print("[ERROR] --model_path or --all_checkpoints required")
        return

    print(f"Checkpoints to eval: {[p.name for p in paths]}")

    val_path = ROOT / cfg["data"]["val_path"]
    if not val_path.exists():
        print(f"  [ERROR] val dataset not found: {val_path}")
        return
    val_ds = load_from_disk(str(val_path))
    print(f"val: {len(val_ds)} seqs")

    # --- eval each checkpoint ---
    results = []
    for p in paths:
        try:
            r = _eval_single(str(p), args, val_ds)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {p.name} failed: {e}")
            _free_vram()

    # --- summary ---
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("PPL Summary")
        print(f"{'='*60}")
        for r in results:
            imp = ""
            if r["base_ppl"]:
                imp = f"  ({(r['base_ppl'] - r['ft_ppl']) / r['base_ppl'] * 100:.1f}% improvement)"
            print(f"  {r['label']:<20} ppl={r['ft_ppl']:.2f}{imp}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
