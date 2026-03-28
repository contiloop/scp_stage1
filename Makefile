.PHONY: setup preprocess train train-resume eval eval-benchmarks merge push-to-hub clean

CONFIG ?= configs/stage1.yaml

setup:
	pip install -e . --no-deps -q
	pip install -U huggingface_hub -q
	pip install "transformers>=5.2.0,<=5.3.0" "trl>=0.15.0" --no-deps -q
	pip install --upgrade unsloth unsloth-zoo --no-deps -q
	# transformers, trl, huggingface_hub만 업그레이드. torch는 docker 이미지 버전 유지.
	python -c "import causal_conv1d" 2>/dev/null || pip install causal-conv1d -q
	python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" 2>/dev/null || pip install flash-linear-attention -q
	# flash-attn 또는 xformers 중 하나 확보
	python -c "import flash_attn" 2>/dev/null || pip install flash-attn --no-build-isolation -q 2>/dev/null || \
		(python -c "import xformers" 2>/dev/null || pip install xformers -q)
	pip install lm-eval -q 2>/dev/null || true

preprocess:
	python -m src.preprocess --config $(CONFIG)

train:
	python -m src.train --config $(CONFIG)

train-resume:
	python -m src.train --config $(CONFIG) --resume auto

eval:
	python -m src.evaluate --config $(CONFIG)

eval-benchmarks:
	python -m src.evaluate --config $(CONFIG) --benchmarks_only

merge:
	python -m src.merge --config $(CONFIG)

push-to-hub:
	python -m src.merge --push $(HF_REPO) --private

clean:
	rm -rf data/processed* checkpoints/ models/merged wandb/
