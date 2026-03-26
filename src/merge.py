"""Merge LoRA adapter into base model and optionally push to HF Hub."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.config import ROOT, BASE_MODEL, CHECKPOINTS, MERGED_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=str(CHECKPOINTS / "stage1_cpt"))
    parser.add_argument("--output", type=str, default=str(MERGED_DIR))
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--push", type=str, default=None, help="HF repo id to push merged model")
    parser.add_argument("--push_adapter", type=str, default=None, help="HF repo id to push adapter only")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    # 1. Push adapter
    if args.push_adapter:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_adapter, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=args.adapter, repo_id=args.push_adapter,
                          commit_message="Upload LoRA adapter")
        print(f"  adapter: https://huggingface.co/{args.push_adapter}")

    # 2. Use pre-merged model from train.py (unsloth adapter lora_B is zero)
    adapter_path = Path(args.adapter)
    pre_merged = adapter_path.parent / "stage1_merged"

    if pre_merged.exists():
        print(f"  using pre-merged model: {pre_merged}")
        model = AutoModelForCausalLM.from_pretrained(
            str(pre_merged), torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(str(pre_merged), trust_remote_code=True)
    else:
        print(f"  [WARN] pre-merged not found, falling back to peft merge (may lose training effect)")
        print(f"  base: {args.base_model}")
        print(f"  adapter: {args.adapter}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    print(f"  merged: {out}")

    # 3. Push merged
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, private=args.private, exist_ok=True)

        model.push_to_hub(args.push, private=args.private,
                          commit_message="Upload merged model (Stage 1 CPT)")
        tokenizer.push_to_hub(args.push, private=args.private)

        # Upload eval results
        eval_dir = ROOT / "checkpoints"
        for pattern in ["eval_results_*.json", "eval_summary_*.txt", "lm_eval_results/comparison_*.json"]:
            for f in eval_dir.glob(pattern):
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f"eval/{f.name}",
                    repo_id=args.push,
                    commit_message=f"Upload eval: {f.name}",
                )
                print(f"  uploaded: eval/{f.name}")

        print(f"  repo: https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
