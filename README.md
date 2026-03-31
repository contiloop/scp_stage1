# scp_stage1

Stage 1: Economic domain continued pretraining (Qwen3.5-4B-Base + LoRA)

## Environment

Built for **`unsloth/unsloth:latest`** Docker image (Vast.ai, RunPod, etc.)

- `make setup` does NOT upgrade torch — uses the image's pre-installed torch + CUDA kernels
- Only upgrades `transformers` (Qwen3.5 support requires ≥5.2.0)
- `causal-conv1d` / `flash-linear-attention` are pre-installed in the image; skipped if already present

**VRAM**: 24GB+ (L4, A10G, RTX 3090/4090, A100, etc.)

## Quick Start

### With Docker image (Vast.ai / RunPod) — recommended

```bash
# select unsloth/unsloth:latest as Docker image
git clone https://github.com/contiloop/scp_stage1.git
cd scp_stage1
make setup
python -c "from huggingface_hub import login; login(token='hf_YOUR_TOKEN')"  # write token (for push_to_hub)
wandb login             # optional: wandb disabled to skip
make preprocess

# tmux로 실행 (SSH 끊겨도 학습 유지)
tmux
make train
# Ctrl+B, D → detach (학습은 계속 돌아감)
# 재접속 후: tmux attach
```

### Without Docker image

```bash
git clone https://github.com/contiloop/scp_stage1.git
cd scp_stage1
pip install -e .
pip install "transformers>=5.2.0" --no-deps
pip install causal-conv1d flash-linear-attention
python -c "from huggingface_hub import login; login(token='hf_YOUR_TOKEN')"
wandb login
make preprocess
make train
```

## Colab

Open `notebooks/train.ipynb` and run all cells.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contiloop/scp_stage1/blob/main/notebooks/train.ipynb)

## After Training

### Evaluate

```bash
# 전체 평가 (ppl + 벤치마크)
make eval

# 벤치마크만 (영어 + 한국어, base vs CPT 비교)
# 영어: MMLU, HellaSwag, ARC, WinoGrande
# 한국어: KMMLU, KoBEST (BoolQ, COPA, HellaSwag)
make eval-benchmarks

# 특정 체크포인트만
python -m src.evaluate --model_path checkpoints/stage1_cpt/checkpoint-750 --base_model unsloth/Qwen3.5-4B-Base --config configs/stage1.yaml --batch_size 1 --skip_benchmarks
```

결과 파일:
- `checkpoints/eval_results_{label}.json` — ppl, completion 결과
- `checkpoints/eval_summary_{label}.txt` — 벤치마크 요약 (base vs CPT)

### Upload to HF Hub

```bash
# merged 모델 + eval 결과 업로드
make push-to-hub HF_REPO=your-username/your-model-name

# 특정 체크포인트를 별도 repo로
python -m src.merge --adapter checkpoints/stage1_cpt/checkpoint-750 --push your-username/model-step750 --private
```

### Download & Evaluate from HF Hub

이 repo 없이도 `lm-eval`만 설치하면 HF Hub 모델을 바로 평가할 수 있습니다.

```bash
pip install "lm-eval[all]" accelerate
huggingface-cli login  # private repo인 경우 필요

lm_eval --model hf \
  --model_args pretrained=your-username/your-model-name,trust_remote_code=True \
  --tasks mmlu,hellaswag,arc_easy,arc_challenge,winogrande,kmmlu,kobest_boolq,kobest_copa,kobest_hellaswag \
  --limit 400 \
  --batch_size 4
```

이 repo의 평가 스크립트를 사용하려면 `make setup` 후:

```bash
python -m src.evaluate --model_path your-username/your-model-name --base_model unsloth/Qwen3.5-4B-Base --benchmarks_only
```

> **Note**: unsloth adapter는 `lora_B=0`으로 저장되므로, `train.py`에서 `save_pretrained_merged()`로 merged 모델을 별도 저장합니다. eval/upload 모두 merged 모델을 사용합니다.

## Config

All hyperparameters in `configs/stage1.yaml` (Qwen3.5) or `configs/stage1_gemma.yaml` (Gemma-3).

```bash
# Qwen3.5 (default)
make preprocess
make train

# Gemma-3-4B
make preprocess CONFIG=configs/stage1_gemma.yaml
make train CONFIG=configs/stage1_gemma.yaml
```

## Data

- [`alwaysgood/ko-news-split-512`](https://huggingface.co/datasets/alwaysgood/ko-news-split-512) — hk, mk, korea_bank, naver
