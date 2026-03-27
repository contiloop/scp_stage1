"""
Stage 1 CPT — Unsloth + LoRA.

Supports both VLM (Qwen3.5) and text-only (Gemma) models via config.

Tricks (CPT-appropriate only):
  - rsLoRA (rank-stabilized scaling)
  - EMA (exponential moving average of weights)
"""

import math
import argparse
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

import unsloth  # must be imported before transformers for kernel patches
import yaml
import torch
from datasets import load_from_disk
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig

from src.config import ROOT


def is_vision_model(model_name: str) -> bool:
    """VLM 구조 모델은 FastVisionModel 사용."""
    vision_keywords = ["qwen3.5", "qwen3_5", "gemma-3", "gemma3"]
    return any(k in model_name.lower() for k in vision_keywords)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Callbacks ──

class EMACallback(TrainerCallback):
    """Exponential Moving Average — stabilizes final weights."""

    def __init__(self, decay=0.999):
        self.decay = decay
        self.ema, self.backup = {}, {}

    def on_step_end(self, args, state, control, model=None, **kw):
        if not self.ema:
            self.ema = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        else:
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.ema:
                    self.ema[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def on_evaluate(self, args, state, control, model=None, **kw):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.ema:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.ema[n])

    def on_log(self, args, state, control, model=None, **kw):
        if self.backup:
            for n, p in model.named_parameters():
                if n in self.backup:
                    p.data.copy_(self.backup[n])
            self.backup = {}

    def on_train_end(self, args, state, control, model=None, **kw):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.ema:
                p.data.copy_(self.ema[n])




class GradNormTrainer(SFTTrainer):
    """SFTTrainer with per-group gradient norm logging to wandb.

    Hooks optimizer.step directly instead of relying on
    on_before_optimizer_step (which Unsloth bypasses).
    """

    DELTANET_KEYS = {"in_proj_a", "in_proj_b", "in_proj_z", "in_proj_qkv", "out_proj"}
    ATTN_KEYS = {"q_proj", "k_proj", "v_proj", "o_proj"}
    MLP_KEYS = {"gate_proj", "up_proj", "down_proj"}

    def create_optimizer(self):
        super().create_optimizer()
        original_step = self.optimizer.step
        trainer_ref = self

        def hooked_step(*args, **kwargs):
            trainer_ref._log_grad_norms()
            return original_step(*args, **kwargs)

        self.optimizer.step = hooked_step

    def _classify(self, name: str):
        """파라미터 이름 → (group, layer_idx)"""
        for key in self.DELTANET_KEYS:
            if key in name:
                return "deltanet", self._layer_idx(name)
        for key in self.ATTN_KEYS:
            if key in name:
                return "attn", self._layer_idx(name)
        for key in self.MLP_KEYS:
            if key in name:
                return "mlp", self._layer_idx(name)
        return "other", -1

    def _layer_idx(self, name: str):
        """layers.XX 에서 숫자 추출"""
        for part in name.split("."):
            if part.isdigit():
                return int(part)
        return -1

    def _log_grad_norms(self):
        if (self.state.global_step + 1) % self.args.logging_steps != 0:
            return

        group_norms = defaultdict(float)
        layer_norms = defaultdict(float)
        grad_count = 0
        none_count = 0

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                none_count += 1
                continue
            grad_count += 1
            norm_sq = p.grad.data.float().norm().item() ** 2
            group, layer_idx = self._classify(n)
            group_norms[group] += norm_sq
            if layer_idx >= 0:
                layer_norms[layer_idx] += norm_sq

        if self.state.global_step <= 30:
            print(f"[GradNorm] step={self.state.global_step} grad={grad_count} none={none_count}")

        log_dict = {}
        for g, ns in group_norms.items():
            log_dict[f"grad_norm/{g}"] = ns ** 0.5
        for layer, ns in sorted(layer_norms.items()):
            log_dict[f"grad_norm/layer_{layer:02d}"] = ns ** 0.5

        if log_dict:
            import wandb
            if wandb.run is not None:
                wandb.log(log_dict, commit=False)


# ── Data collator ──

class PackedCollator:
    def __call__(self, features):
        input_ids = torch.stack([torch.tensor(f["input_ids"]) for f in features])
        labels = torch.stack([torch.tensor(f["labels"]) for f in features])
        return {"input_ids": input_ids, "labels": labels}




# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "stage1.yaml"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_ema", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    m, l, t, d = cfg["model"], cfg["lora"], cfg["training"], cfg["data"]

    # 1. Model
    print("=" * 60)
    vision = is_vision_model(m["name"])
    if vision:
        from unsloth import FastVisionModel as ModelClass
        print(f"Loading {m['name']} (VLM mode) ...")
    else:
        from unsloth import FastLanguageModel as ModelClass
        print(f"Loading {m['name']} (LM mode) ...")

    model, tokenizer = ModelClass.from_pretrained(
        model_name=m["name"],
        max_seq_length=m["max_seq_length"],
        dtype=None,
        load_in_4bit=m.get("load_in_4bit", False),
    )

    # 2. LoRA
    if vision:
        model = ModelClass.get_peft_model(
            model,
            r=l["r"], lora_alpha=l["lora_alpha"], lora_dropout=l["lora_dropout"],
            target_modules=l["target_modules"],
            bias=l["bias"], use_rslora=l.get("use_rslora", False),
            use_gradient_checkpointing="unsloth",
            random_state=t["seed"],
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        )
    else:
        model = ModelClass.get_peft_model(
            model,
            r=l["r"], lora_alpha=l["lora_alpha"], lora_dropout=l["lora_dropout"],
            target_modules=l["target_modules"],
            bias=l["bias"], use_rslora=l.get("use_rslora", False),
            use_gradient_checkpointing="unsloth",
            random_state=t["seed"],
        )
    model.print_trainable_parameters()

    # 3. Data
    train_ds = load_from_disk(str(ROOT / d["train_path"]))
    val_ds = load_from_disk(str(ROOT / d["val_path"]))
    print(f"  train: {len(train_ds)} | val: {len(val_ds)}")

    # 4. Training config
    out = str(ROOT / t["output_dir"])

    # T4는 bf16 미지원 → 자동 감지
    use_bf16 = t.get("bf16", False) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    # warmup_ratio → warmup_steps 변환 (transformers v5.2+ deprecation)
    total_steps = len(train_ds) * t["num_train_epochs"] // (
        t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
    )
    warmup_steps = int(total_steps * t.get("warmup_ratio", 0.03))

    training_args = SFTConfig(
        output_dir=out,
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        bf16=use_bf16,
        fp16=use_fp16,
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t.get("eval_steps", None),
        eval_strategy=t.get("eval_strategy", "no"),
        save_total_limit=t["save_total_limit"],
        seed=t["seed"],
        report_to=t.get("report_to", "none"),
        dataloader_num_workers=t.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
        dataset_text_field="",  # pre-tokenized, no text processing
    )

    # 5. Callbacks
    cbs = []
    if not args.no_ema:
        cbs.append(EMACallback())

    # 6. Fix eos_token for SFTTrainer validation
    # unsloth sets eos_token to <EOS_TOKEN> which isn't in vocab
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    if tok.eos_token_id is not None:
        tok.eos_token = tok.convert_ids_to_tokens(tok.eos_token_id)

    # 7. Trainer
    trainer = GradNormTrainer(
        model=model, args=training_args,
        processing_class=tok,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=PackedCollator(),
        callbacks=cbs,
    )

    # 8. Train
    trainer.train(resume_from_checkpoint=args.resume) if args.resume else trainer.train()

    # 9. Save adapter
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"  adapter saved: {out}")

    # 10. Save merged model (unsloth adapter format is non-standard)
    import gc; gc.collect(); torch.cuda.empty_cache()
    merged_out = str(Path(out).parent / "stage1_merged")
    try:
        model.save_pretrained_merged(merged_out, tokenizer, save_method="merged_16bit")
        print(f"  merged saved: {merged_out}")
    except Exception as e:
        print(f"  [WARN] merged save failed: {e}")

    # 11. Evaluate
    gc.collect(); torch.cuda.empty_cache()
    try:
        metrics = trainer.evaluate()
        if "eval_loss" in metrics:
            print(f"  eval ppl: {math.exp(metrics['eval_loss']):.2f}")
    except torch.cuda.OutOfMemoryError:
        print("  [WARN] evaluate() skipped: CUDA OOM")


if __name__ == "__main__":
    main()
