#!/usr/bin/env bash
# vLLM 0.32.0 / native 1017. Preparation only; never called by template onstart.
set -euo pipefail
: "${BC_PROFILE:?Set two_node_gpu or one_node_cpu explicitly}"
MODEL_DIR=${MODEL_DIR:-/workspace/models/dsv41-1017-heretic}
DRY_RUN=${DRY_RUN:-0}
case "$BC_PROFILE" in
  two_node_gpu)
    : "${NODE_RANK:?Set 0 or 1}" "${LOCAL_IP:?Private routable IP}"
    : "${MASTER_IP:?Node0 private IP}" "${PRIVATE_IFACE:?Private interface}"
    [[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]] || exit 2
    export VLLM_PP_LAYER_PARTITION=14,26
    export VLLM_HOST_IP="$LOCAL_IP"
    export NCCL_SOCKET_IFNAME="=$PRIVATE_IFACE" GLOO_SOCKET_IFNAME="$PRIVATE_IFACE"
    parallel=(--tensor-parallel-size 4 --pipeline-parallel-size 2
      --nnodes 2 --node-rank "$NODE_RANK" --master-addr "$MASTER_IP"
      --master-port "${MASTER_PORT:-29500}")
    [[ "$NODE_RANK" == 0 ]] || parallel+=(--headless)
    engram='{"cpu_offload":false,"dp_shared_memory":false,"use_thp":false}'
    ;;
  one_node_cpu)
    unset VLLM_PP_LAYER_PARTITION
    NODE_RANK=0
    parallel=(--tensor-parallel-size 4 --pipeline-parallel-size 1)
    engram='{"cpu_offload":true,"dp_shared_memory":false,"use_thp":false}'
    ;;
  *) echo "Unknown BC_PROFILE" >&2; exit 2 ;;
esac
export NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO VLLM_ENABLE_GEMM_RS=0
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/hf-metadata HF_XET_CHUNK_CACHE_SIZE_BYTES=0
cmd=(vllm serve "$MODEL_DIR" --served-model-name dsv41-1017-heretic
  "${parallel[@]}" --distributed-executor-backend mp --data-parallel-backend mp
  --language-model-only --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41
  --engram-config "$engram"
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"fp8"}'
  --moe-backend marlin --kv-cache-dtype fp8 --block-size 64 --dtype bfloat16
  --load-format safetensors --safetensors-load-strategy lazy
  --max-model-len 4096 --max-num-seqs 4 --max-num-batched-tokens 1024
  --gpu-memory-utilization 0.90 --disable-custom-all-reduce
  --enable-prompt-tokens-details --host 127.0.0.1 --port 8000)
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
[[ "${BC_KERNEL_GATES_PASSED:-0}" == 1 && "${BC_CHECKPOINT_VALIDATED:-0}" == 1 ]] || {
  echo 'Kernel and exact-checkpoint gates are required.' >&2; exit 3; }
if [[ "$BC_PROFILE" == two_node_gpu ]]; then
  [[ "${BC_NETWORK_GATE_PASSED:-0}" == 1 ]] || { echo 'Eight-rank private NCCL gate required.' >&2; exit 3; }
else
  [[ "${BC_CPU_MEMORY_GATE_PASSED:-0}" == 1 ]] || { echo 'Four-rank Engram pinned-memory gate required.' >&2; exit 3; }
fi
: "${DEADLINE_EPOCH:?Set the original paid-run cleanup deadline; never reset on retries}"
remaining=$((DEADLINE_EPOCH - $(date +%s) - 120))
(( remaining > 1200 )) || { echo 'Insufficient time before the original deadline.' >&2; exit 4; }
exec timeout --signal=TERM --kill-after=30s "$remaining" "${cmd[@]}"
