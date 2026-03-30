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

    Features:
      - LLRD (layer-wise learning rate decay)
      - Module-aware LR scaling
      - Pre/post clip gradient logging
    """

    DELTANET_KEYS = {"in_proj_a", "in_proj_b", "in_proj_z", "in_proj_qkv", "out_proj"}
    ATTN_KEYS = {"q_proj", "k_proj", "v_proj", "o_proj"}
    MLP_KEYS = {"gate_proj", "up_proj", "down_proj"}
    MODULE_ORDER = {"attn": 0, "mlp": 1, "deltanet": 2, "other": 3}
    DEFAULT_MODULE_LR_MULTIPLIERS = {
        "attn": 1.0,
        "mlp": 1.0,
        "deltanet": 1.0,
        "other": 1.0,
    }

    def __init__(self, *args, llrd_decay=1.0, module_lr_multipliers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llrd_decay = llrd_decay
        self.module_lr_multipliers = self._normalize_module_lr_multipliers(module_lr_multipliers)
        self._param_meta = {}
        self._num_layers = 0
        self._pre_clip_step = None
        self._post_clip_step = None
        self._optimizer_hooks_installed = False
        self._accelerator_clip_hook_installed = False

    @classmethod
    def _normalize_module_lr_multipliers(cls, multipliers):
        merged = dict(cls.DEFAULT_MODULE_LR_MULTIPLIERS)
        if multipliers:
            for key, value in multipliers.items():
                if key in merged:
                    merged[key] = float(value)
        return merged

    def _should_log_gradient_stats(self):
        logging_steps = getattr(self.args, "logging_steps", 0) or 0
        return logging_steps > 0 and (self.state.global_step + 1) % logging_steps == 0

    def _wandb_log(self, payload):
        if not payload:
            return

        try:
            import wandb
        except ImportError:
            return

        if wandb.run is not None:
            wandb.log(payload, commit=False)

    def _refresh_param_metadata(self):
        self._param_meta = {}
        layer_indices = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            module_type, layer_idx = self._classify(name)
            self._param_meta[id(param)] = {
                "name": name,
                "module_type": module_type,
                "layer_idx": layer_idx,
            }
            if layer_idx >= 0:
                layer_indices.append(layer_idx)

        self._num_layers = max(layer_indices) + 1 if layer_indices else 0

    def _needs_custom_lr_groups(self):
        if self.llrd_decay < 1.0:
            return True
        return any(
            not math.isclose(multiplier, 1.0)
            for multiplier in self.module_lr_multipliers.values()
        )

    def _depth_scale_for_layer(self, layer_idx):
        if layer_idx < 0 or self._num_layers == 0:
            return 1.0
        return self.llrd_decay ** (self._num_layers - 1 - layer_idx)

    def _rebuild_optimizer_param_groups(self):
        original_groups = list(self.optimizer.param_groups)
        buckets = {}

        for original_idx, group in enumerate(original_groups):
            template = {k: v for k, v in group.items() if k != "params"}
            base_lr = float(template.get("lr", self.args.learning_rate))

            for param in group["params"]:
                meta = self._param_meta.get(id(param))
                if meta is None:
                    continue

                key = (original_idx, meta["layer_idx"], meta["module_type"])
                if key not in buckets:
                    buckets[key] = {
                        "params": [],
                        "template": dict(template),
                        "base_lr": base_lr,
                        "layer_idx": meta["layer_idx"],
                        "module_type": meta["module_type"],
                    }
                buckets[key]["params"].append(param)

        def bucket_sort_key(item):
            key, bucket = item
            layer_idx = bucket["layer_idx"]
            layer_sort = layer_idx if layer_idx >= 0 else self._num_layers
            module_sort = self.MODULE_ORDER.get(bucket["module_type"], len(self.MODULE_ORDER))
            return (key[0], layer_sort, module_sort)

        new_groups = []
        for _, bucket in sorted(buckets.items(), key=bucket_sort_key):
            depth_scale = self._depth_scale_for_layer(bucket["layer_idx"])
            module_scale = self.module_lr_multipliers[bucket["module_type"]]
            effective_lr = bucket["base_lr"] * depth_scale * module_scale

            group = {
                **bucket["template"],
                "params": bucket["params"],
                "lr": effective_lr,
                "base_lr": bucket["base_lr"],
                "depth_scale": depth_scale,
                "module_scale": module_scale,
                "layer_idx": bucket["layer_idx"],
                "module_type": bucket["module_type"],
            }
            new_groups.append(group)

        self.optimizer.param_groups = new_groups

    def _print_optimizer_summary(self):
        if self._num_layers:
            lr_first = self.args.learning_rate * self._depth_scale_for_layer(0)
            lr_last = self.args.learning_rate * self._depth_scale_for_layer(self._num_layers - 1)
            print(f"[LLRD] decay={self.llrd_decay}, layers={self._num_layers}")
            print(f"  layer  0 lr={lr_first:.2e} (min, 일반 지식 보존)")
            print(f"  layer {self._num_layers-1:2d} lr={lr_last:.2e} (max, 도메인 적응)")

        multipliers = ", ".join(
            f"{module}={value:.2f}"
            for module, value in self.module_lr_multipliers.items()
        )
        print(f"[ModuleLR] {multipliers}")

        lr_values = [float(group.get("lr", 0.0)) for group in self.optimizer.param_groups]
        if lr_values:
            print(
                f"[LRGroups] groups={len(lr_values)} "
                f"lr_min={min(lr_values):.2e} lr_max={max(lr_values):.2e}"
            )

    def _install_logging_hooks(self):
        if not self._optimizer_hooks_installed:
            original_step = self.optimizer.step
            trainer_ref = self

            def hooked_step(*args, **kwargs):
                if trainer_ref._should_log_gradient_stats():
                    step = trainer_ref.state.global_step
                    if trainer_ref._pre_clip_step != step:
                        trainer_ref._log_grad_snapshot("grad_pre")
                        trainer_ref._pre_clip_step = step
                    if trainer_ref._post_clip_step != step:
                        trainer_ref._log_grad_snapshot("grad_post")
                        trainer_ref._log_lr_metrics()
                        trainer_ref._post_clip_step = step
                    trainer_ref._log_update_proxy()
                return original_step(*args, **kwargs)

            self.optimizer.step = hooked_step
            self._optimizer_hooks_installed = True

        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and hasattr(accelerator, "clip_grad_norm_") and not self._accelerator_clip_hook_installed:
            original_clip = accelerator.clip_grad_norm_
            trainer_ref = self

            def hooked_clip(*args, **kwargs):
                if trainer_ref._should_log_gradient_stats():
                    step = trainer_ref.state.global_step
                    if trainer_ref._pre_clip_step != step:
                        trainer_ref._log_grad_snapshot("grad_pre")
                        trainer_ref._pre_clip_step = step

                result = original_clip(*args, **kwargs)

                if trainer_ref._should_log_gradient_stats():
                    step = trainer_ref.state.global_step
                    if trainer_ref._post_clip_step != step:
                        trainer_ref._log_grad_snapshot("grad_post")
                        trainer_ref._log_lr_metrics()
                        trainer_ref._post_clip_step = step

                return result

            accelerator.clip_grad_norm_ = hooked_clip
            self._accelerator_clip_hook_installed = True

    def create_optimizer(self):
        super().create_optimizer()
        self._refresh_param_metadata()

        if self._needs_custom_lr_groups():
            self._rebuild_optimizer_param_groups()
        self._print_optimizer_summary()
        self._install_logging_hooks()

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
        """Unsloth의 prediction_step override를 우회. logits 반환 안 함 → fp32 변환 OOM 방지."""
        model.eval()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)
        loss = outputs.loss.detach()
        model.train()
        return (loss, None, None)

    def _classify(self, name: str):
        """파라미터 이름 → (module_type, layer_idx)"""
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

    @torch.no_grad()
    def _collect_grad_norm_squares(self):
        group_norms = defaultdict(float)
        layer_norms = defaultdict(float)
        grad_count = 0
        none_count = 0

        for param in self.model.parameters():
            meta = self._param_meta.get(id(param))
            if meta is None:
                continue
            if param.grad is None:
                none_count += 1
                continue
            grad_count += 1
            norm_sq = param.grad.detach().float().pow(2).sum().item()
            group_norms[meta["module_type"]] += norm_sq
            layer_idx = meta["layer_idx"]
            if layer_idx >= 0:
                layer_norms[layer_idx] += norm_sq

        return group_norms, layer_norms, grad_count, none_count

    def _format_norm_log_dict(self, prefix, group_norms, layer_norms):
        log_dict = {}
        for g, ns in group_norms.items():
            log_dict[f"{prefix}/{g}"] = ns ** 0.5
        for layer, ns in sorted(layer_norms.items()):
            log_dict[f"{prefix}/layer_{layer:02d}"] = ns ** 0.5
        return log_dict

    def _log_grad_snapshot(self, prefix):
        group_norms, layer_norms, grad_count, none_count = self._collect_grad_norm_squares()

        if prefix == "grad_pre" and self.state.global_step <= 30:
            print(f"[GradStats] step={self.state.global_step} grad={grad_count} none={none_count}")

        self._wandb_log(self._format_norm_log_dict(prefix, group_norms, layer_norms))

    @torch.no_grad()
    def _log_update_proxy(self):
        group_updates = defaultdict(float)
        layer_updates = defaultdict(float)

        for group in self.optimizer.param_groups:
            lr = float(group.get("lr", self.args.learning_rate))
            lr_sq = lr ** 2

            for param in group["params"]:
                meta = self._param_meta.get(id(param))
                if meta is None or param.grad is None:
                    continue

                update_sq = param.grad.detach().float().pow(2).sum().item() * lr_sq
                group_updates[meta["module_type"]] += update_sq
                layer_idx = meta["layer_idx"]
                if layer_idx >= 0:
                    layer_updates[layer_idx] += update_sq

        self._wandb_log(self._format_norm_log_dict("update_proxy", group_updates, layer_updates))

    @torch.no_grad()
    def _log_lr_metrics(self):
        all_lrs = []
        group_lrs = defaultdict(set)
        layer_lrs = defaultdict(set)

        for group in self.optimizer.param_groups:
            lr = float(group.get("lr", self.args.learning_rate))
            all_lrs.append(lr)

            for param in group["params"]:
                meta = self._param_meta.get(id(param))
                if meta is None:
                    continue
                group_lrs[meta["module_type"]].add(lr)
                if meta["layer_idx"] >= 0:
                    layer_lrs[meta["layer_idx"]].add(lr)

        if not all_lrs:
            return

        log_dict = {
            "lr/min": min(all_lrs),
            "lr/max": max(all_lrs),
        }

        for module_type, values in sorted(group_lrs.items(), key=lambda item: self.MODULE_ORDER.get(item[0], len(self.MODULE_ORDER))):
            log_dict[f"lr/{module_type}_min"] = min(values)
            log_dict[f"lr/{module_type}_max"] = max(values)

        for layer_idx, values in sorted(layer_lrs.items()):
            log_dict[f"lr/layer_{layer_idx:02d}_min"] = min(values)
            log_dict[f"lr/layer_{layer_idx:02d}_max"] = max(values)

        self._wandb_log(log_dict)


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
    stab = cfg.get("stability", {})

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
        optim=t.get("optim", "adamw_8bit"),
        optim_args=t.get("optim_args"),
        lr_scheduler_type=t["lr_scheduler_type"],
        lr_scheduler_kwargs=t.get("lr_scheduler_kwargs", {}),
        warmup_steps=warmup_steps,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", t["per_device_train_batch_size"]),
        eval_on_start=t.get("eval_on_start", False),
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
        prediction_loss_only=True,  # eval 시 logits fp32 변환 방지 (OOM)
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
        llrd_decay=stab.get("llrd_decay", 1.0),
        module_lr_multipliers=stab.get("module_lr_multipliers"),
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
