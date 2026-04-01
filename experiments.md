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

## Run 8 — Qwen3.5-4B CPT + DeltaNet adapter (Vast.ai 4090)
- **GPU**: RTX 4090 24GB, unsloth/unsloth:latest Docker
- **설정**: r=32, alpha=32, use_rslora=true (scaling 5.66), lr=1e-5, max_grad_norm=1.0, weight_decay=0.01, warmup=0.1, batch=8, grad_accum=2
- **변경**: target_modules에 DeltaNet projection 추가 (`in_proj_a`, `in_proj_b`, `in_proj_z`, `in_proj_qkv`, `out_proj`)
- **Trainable params**: 64,929,792 (1.41%) — Run 6 대비 +22M
- **관찰**: 초반 MLP grad_norm 단조 증가, DeltaNet/attn grad_norm은 안정적 (0.5 수준)
- **MMLU (full)**:
  | Group | Acc |
  |-------|-----|
  | Overall | 76.57% |
  | Humanities | 71.49% |
  | Other | 76.95% |
  | Social Sciences | 83.49% |
  | STEM | 75.23% |
- **Domain PPL**: Base 10.37 → FT 9.47 (**-8.6%**)
- **비고**: Run 6 CPT(-9.8%) 대비 소폭 낮음. DeltaNet adapter 추가가 벤치마크 성능 유지하면서 학습됨.

## Run 9 — Qwen3.5-4B CPT + DeltaNet adapter v2 (실패: backward gradient explosion)
- **GPU**: RTX 4090 24GB
- **설정**: r=32, alpha=32, use_rslora=true (scaling 5.66), lr=5e-6, max_grad_norm=3.0, weight_decay=0.01, warmup=0.1, batch=16, grad_accum=2, EMA off
- **데이터 확장**: 나무위키 경제 + 영어 경제 데이터 추가
  - namuwiki: 96,620 chunks (~43M tokens) — 나무위키 경제/시사/정책/지정학
  - en_econ: 224,636 chunks (~63M tokens) — NCERT Economics/Business/Accounting + Bloomberg 120k
  - 기존 4소스 ~18M + 신규 ~106M = **총 ~124M tokens**
  - 언어 비율: 한국어 ~50% / 영어 ~50%
- **Total steps**: 2361
- **관찰**:
  - step 510, 1010에서 loss spike (eval_steps=500 직후 + logging_steps=10)
  - step ~850부터 layer-wise grad_norm 분기:
    - Layer 31~11: grad_norm 감소 (loss에 가까워 아직 증폭 안 된 gradient)
    - Layer 10~6: 전환 구간
    - Layer 5~0: grad_norm 폭주 (32개 layer를 통과하며 누적 증폭된 gradient)
  - step 1010 spike에서 grad_norm은 오히려 **낮아지는** 방향
- **원인 분석**:
  1. **Loss spike (step 510, 1010)**: eval_steps=500 직후에 발생. eval 전후 모델 상태 전환(train→eval→train) 과정에서 gradient accumulation 또는 CUDA 메모리 상태 교란 가능성
  2. **Backward gradient 증폭**: LoRA weight 누적 성장(rsLoRA 5.66x) → 후반 layer Jacobian의 spectral norm > 1 → backward pass에서 gradient가 layer를 거슬러 올라갈수록 기하급수적 증폭 → 초반 layer에서 gradient explosion
  3. step 850이 분기점인 이유: 이 시점에서 LoRA perturbation이 충분히 커져 Jacobian이 gradient를 증폭하기 시작
- **교훈**: 후반 layer grad_norm 감소는 collapse가 아니라 "아직 증폭 안 된 gradient". 문제의 본질은 후반 layer의 LoRA weight 성장 → Jacobian 증폭 → 초반 layer gradient explosion. LLRD로 후반 layer weight 성장 억제 필요.

## Run 10 — Qwen3.5-4B CPT + DeltaNet adapter v3 (실패: catastrophic forgetting)
- **변경**:
  1. LLRD (Layer-wise LR Decay) 적용: decay=0.95, layer 0=5e-6(max), layer 31≈1e-6(min)
  2. Spike skip 안전장치 (실제로는 clipping 후 실행되어 무효)
  3. prediction_step override (eval OOM 방지)
- **설정**: lr=5e-6, max_grad_norm=3.0, 그 외 Run 9과 동일
- **방식**: Run 9의 checkpoint-499에서 resume
- **결과**: MMLU 76.6% → **38.0%** (-38.6%), Winogrande 63.5%
- **관찰**:
  - loss는 노이즈 많지만 하강 (2.38→2.28)
  - grad_norm 후반에 1→8까지 증가, 여전히 후반 layer 감소 + 초반 layer 증가 패턴
  - spike skip은 gradient clipping(3.0) 후에 실행되어 한 번도 trigger 안 됨
- **원인**: LLRD 방향이 반대였음
  - layer 0(초반) = max lr, layer 31(후반) = min lr로 설정
  - backward gradient amplification으로 초반 layer가 가장 큰 gradient를 받는데, 거기에 가장 큰 lr까지 적용
  - 큰 gradient × 큰 lr = 초반 layer의 일반 지식(MMLU) 파괴
  - resume로 인한 optimizer 불일치도 기여 가능성 있음
- **교훈**: 표준 LLRD는 초반 layer = min lr (보존), 후반 layer = max lr (적응). 이렇게 해야 backward gradient amplification을 상쇄.

## Run 11 — Qwen3.5-4B CPT + DeltaNet adapter v4 (gradient explosion 전 중단)
- **변경**:
  1. LLRD 방향 수정: decay=0.95, layer 0≈1e-6(min), layer 31=5e-6(max)
  2. Spike skip 제거 (clipping 후 실행이라 무효, max_grad_norm=3.0에 의존)
- **설정**: lr=5e-6, max_grad_norm=3.0, 그 외 Run 10과 동일
- **방식**: clean start (checkpoint resume 아님)
- **관찰**:
  - grad_norm이 후반 step에서 지수적으로 증가 — explosion 직전에 checkpoint-998에서 중단
  - LLRD 방향을 바꿔도 후반 layer grad_norm이 지수 증가하는 패턴은 해결되지 않음
- **결과 (checkpoint-998, explosion 전):**
  - 영어 벤치마크 (lm-eval, limit=400):
    | Task | Base | CPT (ckpt-998) | Diff |
    |------|------|----------------|------|
    | MMLU | 76.7% | 76.71% | +0.0% |
    | HellaSwag | 50.7% | 50.75% | +0.1% |
    | ARC-Easy | 81.2% | 81.00% | -0.2% |
    | ARC-Challenge | 50.5% | 52.75% | +2.3% |
    | WinoGrande | 71.8% | 71.00% | -0.8% |
  - 한국어 벤치마크 (lm-eval, limit=400):
    | Task | Base (Run 6, full) | CPT (ckpt-998) | Diff |
    |------|-------------------|----------------|------|
    | KMMLU | 48.9% | 49.01% | +0.1% |
    | KoBEST BoolQ | 78.5% | 78.25% | -0.3% |
    | KoBEST COPA | 70.3% | 70.50% | +0.2% |
    | KoBEST HellaSwag | 46.3% | 46.75% | +0.5% |
  - ⚠️ 영어/한국어 모두 base와 거의 동일. 998 step (~64M tokens) 학습했으나 벤치마크에 반영 없음. gradient explosion 전에 끊은 것이므로, 벤치마크 유지는 LLRD 효과가 아니라 단순히 아직 망가지기 전인 상태
- **교훈**: LLRD는 어느 방향으로 적용해도 gradient explosion을 근본적으로 해결하지 못함. max_grad_norm=3.0 + 7x 데이터(124M tokens)에서의 누적 update가 핵심 원인일 가능성.

## Run 12 — Qwen3.5-4B CPT + DeltaNet adapter v5 (explosion 해소, benchmark 하락)
- **목표**:
  1. 표준 LLRD 방향은 유지하면서 gradient explosion을 억제
  2. CPT 데이터 품질 개선으로 bad batch 요인을 제거
  3. standalone eval / benchmark 경로까지 안정적으로 재현 가능하게 만들기
- **설정**:
  - `optim=adamw_torch`
  - `llrd_decay=0.95`
  - `module_lr_multipliers={attn:1.0, mlp:1.0, deltanet:1.0, other:1.0}`
  - `gradient_accumulation_steps=4`
  - preprocessing 대폭 수정:
    - raw text에서 URL / 이메일 / bare domain 제거
    - `hard cut` 제거, 문장/문단 경계 기반 packing 유지
    - `padding 제외 마지막 non-pad = EOS` 보장
    - trailing newline, 괄호형 고지문, 날짜 단독 라인, 캡션/출처 라인, 긴 연표 tail block 제거
    - HF 원격 대신 로컬 dataset/tokenizer cache 우선 사용
    - 최종 packed dataset: `71,831 seqs x 2048`, `padding 1.0%`, `train 70,395 / val 1,436`
- **관찰**:
  - 이전처럼 gradient explosion은 재현되지 않음
  - train 종료 직후 ppl 출력은 정상, standalone eval / benchmark 경로도 별도 수정 후 안정화
  - 다만 domain ppl은 기대처럼 개선되지 않고 오히려 상승
  - benchmark는 catastrophic forgetting 수준까지는 아니지만, base 대비 눈에 띄게 하락
- **영어 벤치마크 (MMLU)**:
  | Group | Base | Run 11 (ckpt-998) | Run 12 | Diff vs Base |
  |-------|------|-------------------|--------|--------------|
  | Overall | 76.7% | 76.71% | **75.12%** | **-1.58%** |
  | Humanities | 71.49% | - | 70.12% | -1.37% |
  | Other | 76.95% | - | 75.59% | -1.36% |
  | Social Sciences | 83.49% | - | 81.99% | -1.50% |
  | STEM | 75.23% | - | 73.68% | -1.55% |
- **한국어 벤치마크 (KMMLU)**:
  | Group | Base | Run 11 (ckpt-998) | Run 12 | Diff vs Base |
  |-------|------|-------------------|--------|--------------|
  | Overall | 48.9% | 49.01% | **45.16%** | **-3.74%** |
  | Applied Science | - | - | 43.42% | - |
  | HUMSS | - | - | 45.68% | - |
  | Other | - | - | 44.50% | - |
  | STEM | - | - | 47.33% | - |
- **해석**:
  - 안정성 측면에서는 성공: `explosion -> no explosion`
  - 품질 측면에서는 기대 이하: `benchmark 유지 -> benchmark 하락`
  - 특히 한국어(KMMLU) 하락폭이 영어(MMLU)보다 큼
  - 즉 현재 세팅은 "학습은 안정화됐지만, 성능 보존/향상 측면에서 우세하지 않은 절충안"에 가깝다
- **교훈**:
  1. bad batch 문제를 제거해도, Qwen3.5 + DeltaNet CPT에서 장기 누적 update가 성능 저하를 완전히 막아주진 않음
  2. "explosion 방지"와 "benchmark 보존"은 별개 문제였음
  3. 다음 비교축은 Qwen 하이퍼파라미터 미세조정보다 Gemma-3 baseline을 먼저 확인하는 것이 더 효율적일 수 있음

## Run 13 — Gemma-3-4B CPT baseline (초기 benchmark 확인)
- **목표**:
  1. Qwen Run 12와 최대한 비슷한 학습/평가 조건에서 Gemma-3 baseline 확인
  2. `explosion` 없이 CPT가 안정적으로 진행되는지 점검
  3. 영어/한국어 벤치마크 보존력 비교
- **설정**:
  - config: `configs/stage1_gemma.yaml`
  - `optim=adamw_torch`
  - `learning_rate=5e-6`
  - `llrd_decay=0.95`
  - `module_lr_multipliers={attn:1.0, mlp:1.0, deltanet:1.0, other:1.0}`
  - `per_device_train_batch_size=16`
  - `gradient_accumulation_steps=4`
  - benchmark-only eval:
    - `python -m src.evaluate --model_path checkpoints/stage1_cpt_gemma --config configs/stage1_gemma.yaml --benchmarks_only --skip_base_benchmarks --batch_size 1`
- **결과 (영어/한국어 benchmark-only)**:
  - 상단 aggregate: `0.6950 ± 0.0230`
  - 영어 벤치마크 (MMLU):
    | Group | Gemma Base (Run 7) | Run 13 | Diff vs Base |
    |-------|---------------------|--------|--------------|
    | Overall | 60.0% | **59.15%** | **-0.85%** |
    | Humanities | - | 59.14% | - |
    | Other | - | 61.93% | - |
    | Social Sciences | - | 67.10% | - |
    | STEM | - | 49.89% | - |
  - 한국어 벤치마크 (KMMLU):
    | Group | Gemma Base (Run 7) | Run 13 | Diff vs Base |
    |-------|---------------------|--------|--------------|
    | Overall | 35.2% | **32.96%** | **-2.24%** |
    | Applied Science | - | 32.67% | - |
    | HUMSS | - | 29.96% | - |
    | Other | - | 34.42% | - |
    | STEM | - | 33.98% | - |
- **해석**:
  - 현재 시점 Gemma-3 baseline은 Qwen Run 12보다 영어/한국어 모두 낮음
  - Gemma 자기 기준으로 봐도 benchmark 보존이 안 좋음
  - `Gemma Base -> Run 13` 변화:
    - `MMLU 60.0% -> 59.15%` (`-0.85%p`)
    - `KMMLU 35.2% -> 32.96%` (`-2.24%p`)
  - 특히 한국어(KMMLU) 격차가 큼: `Qwen Run 12 45.16%` vs `Gemma Run 13 32.96%`
  - 영어(MMLU)도 차이가 큼: `Qwen Run 12 75.12%` vs `Gemma Run 13 59.15%`
  - 따라서 "Gemma로 바꾸면 benchmark 보존력이 더 좋을 것"이라는 가설은 현재 결과만 보면 지지되지 않음
- **주의**:
  - 이 결과는 `skip_base_benchmarks`로 측정한 CPT-only benchmark이며, 이후 Gemma base 직접 비교를 붙이면 해석이 더 명확해질 수 있음
  - benchmark eval 중 VRAM OOM이 간헐적으로 발생해 `batch_size=1`로 실행함
