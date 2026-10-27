DeepSeek V4.1 Flash 1017 — RTX96 inference image candidate

This repository contains a Linux amd64 Docker build recipe for RTX PRO 6000
Blackwell 96GB GPUs (SM120). A successful CPU-only build validates package
installation and static checks. It does not validate GPU kernels, model loading,
correct generation, latency or throughput. No performance claims are made.

The parent vLLM image is pinned by digest. The recipe preserves vLLM 0.32.0,
Torch 2.13.0+cu130 and FlashInfer 0.7.0.post1, retains matching inherited cubin and
SM120f JIT cache wheels after checking their installed RECORD hashes, and installs
missing wheels from hash-verified official release artifacts. Operational
Vast CLI dependencies are isolated in a separate, hash-locked environment.
The OS package additions are recorded but are not a complete reproducibility lock.

Build from this directory:
  docker build --platform linux/amd64 -t deepseek-rtx96:candidate .

No model weights, credentials, account information or runtime evidence are
included. Default startup is inert. The optional Vast SSH onstart command is:
  bash /opt/dsv41/init.sh
It checks package identity and records local hardware facts; it does not download
weights, arm deletion or launch inference. Vast SSH mode replaces ENTRYPOINT.

After publication, use the actual repository@sha256 registry manifest digest.
A tag, parent digest or local Docker config ID is insufficient. The admission
record distinguishes the parent runtime digest from the deployed derived digest.

The prepared full-model profiles are one host with four GPUs (TP4, CPU Engram)
and two hosts with four GPUs each (TP4/PP2, layers14/26, GPU Engram). They require
actual target-GPU, memory and collective gates. Two-host execution additionally
requires verified private routing; ordinary public port forwarding is insufficient.
Read runtime/OPERATOR.txt before deployment. That document describes explicit
operator actions; nothing in this repository authorizes rentals or spending.

The model reference is pqhaz/DeepSeek-V4.1-Flash-1017-heretic, revision
12d58ad0878b5008e80d720251adb969a0069412. Its model files and terms are hosted
separately; no checkpoint is redistributed by this image.

Sources:
  https://github.com/vllm-project/vllm
  https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.0.post1
  https://github.com/vast-ai/vast-cli
  https://docs.vast.ai/guides/instances/connect/networking
  https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic/tree/12d58ad0878b5008e80d720251adb969a0069412
