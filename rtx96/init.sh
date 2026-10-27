#!/usr/bin/env bash
# Vast onstart: bash /opt/dsv41/init.sh
set -euo pipefail
umask 077
# Generate missing per-container host identities; never ship private host keys.
ssh-keygen -A >/dev/null
python3 /opt/dsv41/verify-runtime.py
mkdir -p /workspace/dsv41-template/evidence /workspace/models /workspace/hf-metadata
# Only scripts/assets are copied; evidence and guard state are never overwritten.
cp -a /opt/dsv41/runtime/. /workspace/dsv41-template/
cat > /workspace/dsv41-template/environment.sh <<'ENV'
export MODEL=pqhaz/DeepSeek-V4.1-Flash-1017-heretic
export REV=12d58ad0878b5008e80d720251adb969a0069412
export MODEL_DIR=/workspace/models/dsv41-1017-heretic
export HF_HOME=/workspace/hf-metadata
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export TOKENIZERS_PARALLELISM=false
export TP_SIZE=4
export VAST_CLI=/opt/vast-cli/bin/vastai
ENV
python3 /workspace/dsv41-template/hardware_facts.py \
  > "/workspace/dsv41-template/evidence/hardware-$(date -u +%Y%m%dT%H%M%SZ).json"
printf '%s\n' 'Inventory complete. No weights downloaded or server started.' \
  'Read /workspace/dsv41-template/OPERATOR.txt and verify the deployed image digest.'
