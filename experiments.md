# Experiment Log

## Run 1 — Vast.ai L4 (실패: gradient explosion)
- **GPU**: L4 24GB, unsloth/unsloth:latest Docker
- **설정**: r=64, alpha=128, seq=2048, batch=8, grad_accum=2, lr=5e-5, dropout=0
- **결과**: epoch 0.21에서 grad_norm 폭발 (3→145→3012→1.68M→4.65M), loss 2.15→8.4
- **교훈**: lr=5e-5 + r=64에서 불안정

## Run 2 — Vast.ai L4 (실패: gradient explosion)
- **GPU**: L4 24GB, unsloth/unsloth:latest Docker
- **설정**: r=64, alpha=128, seq=2048, batch=8, grad_accum=2, lr=2e-5, dropout=0
- **결과**: epoch 0.28~0.32에서 grad_norm 재폭발 (3→24→220→797)
- **교훈**: lr 낮춰도 r=64에서 여전히 불안정, 같은 구간 반복 → 데이터 or rank 문제

## Run 3 — Vast.ai L4 (실패: gradient explosion)
- **GPU**: L4 24GB, unsloth/unsloth:latest Docker
- **설정**: r=32, alpha=64, seq=2048, batch=8, grad_accum=2, lr=2e-5, max_grad_norm=0.3, warmup=0.1
- **결과**: epoch 0.31에서 grad_norm 재폭발 (5.7→15.4)
- **교훈**: r=32로 줄여도 같은 구간(0.28~0.32)에서 반복. 데이터 이상 없음 확인.

## Run 4 — Vast.ai L4 (실패: grad_norm 단조 증가 + eval OOM)
- **GPU**: L4 24GB, unsloth/unsloth:latest Docker
- **설정**: r=32, alpha=64, seq=2048, batch=8, grad_accum=2, lr=1e-5, max_grad_norm=1.0, warmup=0.1, weight_decay=0.1
- **관찰**: grad_norm 단조 증가 (1.6→4.5 at epoch 0.50), loss 정상 하강 (2.34→2.18)
- **결과**: step 500에서 eval_steps=500 트리거 → accelerate가 eval 시 fp32 변환 → OOM 크래시. checkpoint 500 미저장.
- **교훈**: 학습 중 eval 비활성화 필요 (eval_strategy="no"), save_steps를 eval_steps보다 작게 설정

## Run 5 — Vast.ai L4 (실패: grad_norm 단조 증가 → 폭발)
- **GPU**: L4 24GB, unsloth/unsloth:latest Docker
- **설정**: r=32, alpha=64, seq=2048, batch=8, grad_accum=2, lr=1e-5, max_grad_norm=1.0, warmup=0.1, weight_decay=0.1, eval_strategy="no", save_steps=250, **use_rslora=true**
- **관찰**: grad_norm 단조 증가 (1.6→4.5 at epoch 0.50), epoch 0.55 부근에서 다시 폭발
- **loss**: 정상 하강 (2.34→2.18) 했으나 grad_norm 폭발로 중단
- **근본 원인 발견**: rsLoRA scaling 문제
  - `use_rslora=true` + `alpha=64, r=32` → effective scaling = 64/√32 = **11.3배**
  - Standard LoRA였으면 64/32 = 2.0배
  - 11.3배 증폭이 gradient 누적 발산의 주 원인으로 추정
- **조치**: lora_alpha: 64→32 (scaling 11.3→5.66), Gemma config 추가

---

## rsLoRA Scaling Factor 정리

| rsLoRA | alpha | r | scaling | 비고 |
|--------|-------|---|---------|------|
| false | 64 | 32 | 64/32 = 2.0 | standard LoRA |
| true | 64 | 32 | 64/√32 = 11.3 | ❌ Run 1~5에서 사용, 너무 큼 |
| true | 32 | 32 | 32/√32 = 5.66 | ✅ rsLoRA 권장 (alpha=r) |
| false | 32 | 32 | 32/32 = 1.0 | 가장 보수적 |

---

## Run 6 — Qwen3.5-4B CPT (Vast.ai 4090)
- **GPU**: RTX 4090 24GB, unsloth/unsloth:latest Docker
- **설정**: r=32, alpha=32, use_rslora=true (scaling 5.66), lr=1e-5, weight_decay=0.1, warmup=0.1, batch=8, grad_accum=2
- **관찰**: grad_norm 단조 증가 경향 있었으나 폭발하지는 않음. loss 정상 하강.
- **결과**:
  - Domain PPL: 10.37 → **9.35** (-9.8%)
  - 영어 벤치마크 (lm-eval, limit=400):
    | Task | Base | CPT | Diff |
    |------|------|-----|------|
    | MMLU | 76.7% | 76.6% | -0.1% |
    | HellaSwag | 50.7% | 50.7% | 0.0% |
    | ARC-Easy | 81.2% | 81.0% | -0.2% |
    | ARC-Challenge | 50.5% | 50.0% | -0.5% |
    | Winogrande | 71.8% | 72.8% | +1.0% |
  - 한국어 벤치마크 (lm-eval, full):
    | Task | Base | CPT | Diff |
    |------|------|-----|------|
    | KMMLU | 48.9% | 48.4% | -0.5% |
    | KoBEST BoolQ | 78.5% | 75.8% | -2.7% |
    | KoBEST COPA | 70.3% | 67.8% | -2.5% |
    | KoBEST HellaSwag | 46.3% | 45.8% | -0.5% |
- **HF**: `alwaysgood/Qwen3.5-4B-CPT-stage1`
- **교훈**: 도메인 PPL 개선, 영어 유지, 한국어 BoolQ/COPA 소폭 하락. weight_decay=0.1이 기존 지식 보존에 부정적 영향 가능성.

## Run 7 — Gemma-3-4B CPT (Vast.ai 4090)
- **GPU**: RTX 4090 24GB, unsloth/unsloth:latest Docker
- **설정**: r=32, alpha=32, use_rslora=true (scaling 5.66), lr=1e-5, weight_decay=0.01, warmup=0.1, batch=8, grad_accum=2
- **관찰**: grad_norm 안정적 (1.1~1.7), loss 안정적 하강 (1.85→1.83)
- **결과**:
  - Domain PPL: 10.46 → **6.35** (-39.3%)
  - 영어 벤치마크 (lm-eval, limit=400):
    | Task | Base | CPT | Diff |
    |------|------|-----|------|
    | MMLU | 60.0% | 60.0% | 0.0% |
    | HellaSwag | 65.5% | 64.8% | -0.7% |
    | ARC-Easy | 82.8% | 83.8% | +1.0% |
    | ARC-Challenge | 55.3% | 55.3% | 0.0% |
    | Winogrande | 71.8% | 69.0% | -2.8% |
  - 한국어 벤치마크 (lm-eval, full):
    | Task | Base | CPT | Diff |
    |------|------|-----|------|
    | KMMLU | 35.2% | 34.6% | -0.6% |
    | KoBEST BoolQ | 64.8% | 64.8% | 0.0% |
    | KoBEST COPA | 73.8% | 73.3% | -0.5% |
    | KoBEST HellaSwag | 45.0% | 43.5% | -1.5% |
- **HF**: `alwaysgood/Gemma-3-4B-CPT-stage1`
- **교훈**: PPL 39% 대폭 개선 (Qwen 대비 4배). 벤치마크 거의 유지. weight_decay=0.01 + 표준 Attention 구조가 안정적 학습에 기여.

---

## Qwen vs Gemma 비교

| | Qwen3.5-4B | Gemma-3-4B |
|---|-----------|-----------|
| Base 한국어 (KMMLU) | 48.9% | 35.2% |
| Domain PPL 개선 | -9.8% | **-39.3%** |
| 영어 벤치마크 유지 | ✅ | ✅ |
| 한국어 벤치마크 유지 | △ (BoolQ -2.7%) | ✅ |
| grad_norm 안정성 | △ 단조 증가 | ✅ 안정 |
| weight_decay | 0.1 | 0.01 |
| 아키텍처 | DeltaNet+Attention | Attention only |

---

## Run 8 — Qwen3.5-4B CPT v2 (예정)
- **변경**: weight_decay 0.1 → 0.01 (Gemma와 동일)
- **가설**: weight_decay가 높으면 LoRA 업데이트와 충돌하여 grad_norm 불안정 + 기존 지식(한국어) 보존 저하
- **기대**: grad_norm 안정화, 한국어 벤치마크 하락 완화
