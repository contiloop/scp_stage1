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

## Run 6 — Qwen3.5 (예정)
- **변경**: lora_alpha: 32, use_rslora: true → scaling 5.66
- **기대**: grad_norm 안정화

## Run 7 — Gemma 3 4B (예정)
- **모델**: unsloth/gemma-3-4b-pt
- **목적**: DeltaNet 없는 표준 Transformer로 비교 실험
- **기대**: gradient explosion 없이 안정적 학습
