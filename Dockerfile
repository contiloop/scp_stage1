FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# system deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# python deps
RUN pip install --no-cache-dir \
    unsloth unsloth-zoo \
    transformers>=4.57.0 peft>=0.15.0 datasets>=3.0.0 \
    accelerate>=1.3.0 trl>=0.15.0 \
    wandb sentencepiece tiktoken safetensors pyyaml tqdm numpy

# CUDA kernels (slow build, cached in image)
RUN pip install --no-cache-dir causal-conv1d flash-linear-attention
RUN pip install --no-cache-dir flash-attn --no-build-isolation

WORKDIR /workspace
