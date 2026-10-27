#!/usr/bin/env python3
"""Real four-rank pinned Engram allocator/UVA admission test; no weights or server.

Run only on the selected one-host TP4 allocation, before downloading weights:
  timeout --signal=TERM --kill-after=30s 600 \
    torchrun --standalone --nproc-per-node=4 preflight_cpu_engram_tp4.py
Requires the original DEADLINE_EPOCH. All four PASS records are required.
This intentionally holds approximately264GiB of pinned RAM concurrently.
"""
import datetime
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import vllm
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.models.deepseek_v41.common.engram import EngramLayout, ParallelEngramEmbedding
from vllm.utils.platform_utils import is_uva_available


def available_ram():
    info = {k: int(v.split()[0]) * 1024 for k, v in
            (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    available = info['MemAvailable']
    facts = {'host_available_bytes': available, 'host_total_bytes': info['MemTotal']}
    for limit_path, used_path in (
        ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes'),
    ):
        limit_file, used_file = Path(limit_path), Path(used_path)
        if limit_file.exists() and used_file.exists():
            limit_text = limit_file.read_text().strip()
            if limit_text.isdigit():
                limit, used = int(limit_text), int(used_file.read_text())
                facts.update(cgroup_limit_bytes=limit, cgroup_used_bytes=used)
                available = min(available, limit - used)
                break
    facts['admission_available_bytes'] = available
    return available, facts


deadline = int(os.environ['DEADLINE_EPOCH'])
assert deadline - time.time() > 900, 'Insufficient remaining time for bounded memory gate'
assert vllm.__version__.startswith('0.32.0'), vllm.__version__
rank = int(os.environ['RANK'])
assert int(os.environ['WORLD_SIZE']) == 4
assert int(os.environ['LOCAL_WORLD_SIZE']) == 4 and int(os.environ['LOCAL_RANK']) == rank
assert torch.cuda.device_count() == 4
torch.cuda.set_device(rank)
torch.set_num_threads(2)
assert torch.cuda.get_device_capability(rank) == (12, 0)
assert is_uva_available(), 'CPU Engram requires UVA; do not use a full-copy fallback'
available, before = available_ram()
assert available >= 320 * 2**30, before
init_distributed_environment(world_size=4, rank=rank, local_rank=rank,
                             distributed_init_method='env://', backend='nccl',
                             timeout=datetime.timedelta(seconds=600))
initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
probe = torch.tensor([rank + 1.0], device=f'cuda:{rank}')
dist.all_reduce(probe)
assert probe.item() == 10, 'Local four-rank collective failed'

# Exact text-config Engram fields from the pinned native1017 checkpoint.
config = SimpleNamespace(engram_layer_ids=[1, 14],
    engram_num_embeddings=[384006168, 384016682], engram_max_ngram_size=4,
    engram_n_heads=8, engram_head_dim=256, engram_compressed_vocab_size=99092,
    engram_pad_token_id=2, engram_vocab_size=16000000)
layout = EngramLayout(config)
tables = []
for index, layer_id in enumerate(layout.layer_ids):
    assert time.time() < deadline - 300, 'Preserve cleanup margin'
    head_sizes = tuple(size for ngram in layout.primes[index] for size in ngram)
    table = ParallelEngramEmbedding(layout.num_embeddings[index], layout.head_dim,
        head_sizes=head_sizes, block_size=32, cpu_offload=True,
        dp_shared_memory=False, use_thp=False)
    assert table.part_n_hash_cols == 6
    for parameter in (table.weight, table.weight_scale_inv):
        assert parameter.is_pinned() and parameter.device.type == 'cpu'
        raw = parameter.detach().view(torch.uint8).reshape(-1)
        raw[::4096].fill_(1)  # Commit/touch every page, not only endpoints.
        raw[-1] = 1
    # Execute real GPU reads through the same UVA views used by the model.
    for view in table._storage():
        raw = view.view(torch.uint8).reshape(-1)
        indices = torch.tensor([0, raw.numel() - 1], device=f'cuda:{rank}')
        observed = raw.index_select(0, indices).cpu()
        assert observed.tolist() == [1, 1], 'Engram UVA read mismatch'
    torch.cuda.synchronize()
    tables.append(table)
    print(json.dumps({'allocated_layer': layer_id, 'rank': rank,
        'hash_heads': table.part_n_hash_cols, 'rows': table.part_num_embeddings,
        'logical_bytes': table.weight.numel() + table.weight_scale_inv.numel(),
        'default_pinned_planning_GiB': 33}), flush=True)

# Everyone retains BOTH tables until every rank has completed and read them.
dist.barrier()
_, after = available_ram()
print(json.dumps({'cpu_engram_tp4_gate': 'PASS', 'rank': rank,
    'concurrent_tables_per_rank': len(tables), 'default_pinned_planning_GiB_per_rank': 66,
    'default_pinned_planning_GiB_host': 264, 'before': before, 'after': after,
    'note': 'Real allocator, page touches and UVA reads; no model load or generation.'}), flush=True)
dist.barrier()
dist.destroy_process_group()
# Process exit releases CUDA contexts, CPU allocator caches and pinned tables.
