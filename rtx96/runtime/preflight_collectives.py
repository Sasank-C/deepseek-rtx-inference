#!/usr/bin/env python3
"""No-model Gloo/NCCL gate. Run with torchrun and an outer 240s kill timeout.

Single: --nnodes=1 --nproc-per-node=4, --profile tp4-single-cpu.
Two hosts: --nnodes=2 --nproc-per-node=4 --node-rank=N --master-addr=PRIVATE0,
--profile tp4-pp2-gpu. A verified private route must precede the two-host test.
Requires DEADLINE_EPOCH; produces one sanitized JSON PASS record per rank.
"""
import argparse
import datetime
import json
import os
import threading
import time


def topology(profile, rank, world, local_world):
    expected = 4 if profile == 'tp4-single-cpu' else 8
    if profile not in ('tp4-single-cpu', 'tp4-pp2-gpu'):
        raise ValueError('Unsupported profile')
    if world != expected or local_world != 4 or not 0 <= rank < world:
        raise ValueError('Require exactly four ranks per node and the expected world')
    return {'tp_groups': [list(range(start, start + 4)) for start in range(0, world, 4)],
            'pp_peer': (rank + 4 if rank < 4 else rank - 4) if world == 8 else None,
            'payload_bytes_per_rank': 40976 * 1024 // 4 if world == 8 else 0}


def abort():
    os.write(2, b'{"gate":"FAIL","reason":"collective_gate_walltime"}\n')
    os._exit(124)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True,
                        choices=['tp4-single-cpu', 'tp4-pp2-gpu'])
    args = parser.parse_args()
    deadline = int(os.environ['DEADLINE_EPOCH'])
    remaining = deadline - time.time() - 15
    if remaining < 120:
        raise SystemExit('Insufficient original deadline for collective gate')
    timer = threading.Timer(min(225, remaining), abort)
    timer.daemon = True
    timer.start()
    import torch
    import torch.distributed as dist
    local_rank = int(os.environ['LOCAL_RANK'])
    if torch.cuda.device_count() != 4 or local_rank not in range(4):
        raise RuntimeError('Exactly four visible CUDA devices required')
    torch.cuda.set_device(local_rank)
    # Gloo runs actual reverse/dynamic peer connections, rather than merely
    # proving the fixed TCP-store rendezvous port is reachable.
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    layout = topology(args.profile, rank, world, int(os.environ['LOCAL_WORLD_SIZE']))
    if rank // 4 != int(os.environ['GROUP_RANK']):
        raise RuntimeError('Unexpected rank-to-physical-node mapping')
    cpu_value = torch.tensor([rank], dtype=torch.int64)
    cpu_gather = [torch.empty_like(cpu_value) for _ in range(world)]
    dist.all_gather(cpu_gather, cpu_value)
    if [value.item() for value in cpu_gather] != list(range(world)):
        raise RuntimeError('Gloo rank/value mismatch')
    timeout = datetime.timedelta(seconds=90)
    all_nccl = dist.new_group(list(range(world)), backend='nccl', timeout=timeout)
    groups = [dist.new_group(ranks, backend='nccl', timeout=timeout)
              for ranks in layout['tp_groups']]
    group = groups[rank // 4]
    # One MiB per rank exercises more than scalar metadata on each TP4 host.
    value = torch.full((262144,), float(local_rank + 1), device='cuda')
    dist.all_reduce(value, group=group)
    if not torch.all(value == 10).item():
        raise RuntimeError('Local TP4 all_reduce mismatch')
    own = torch.tensor([rank], device='cuda', dtype=torch.int64)
    gathered = [torch.empty_like(own) for _ in range(4)]
    dist.all_gather(gathered, own, group=group)
    if [value.item() for value in gathered] != layout['tp_groups'][rank // 4]:
        raise RuntimeError('Local TP4 all_gather mismatch')
    peer = layout['pp_peer']
    elapsed = None
    if peer is not None:
        count = layout['payload_bytes_per_rank'] // 4
        payload = torch.full((count,), float(rank + 1), device='cuda')
        received = torch.empty_like(payload)
        dist.barrier(); torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(5):
            if rank < 4:
                dist.send(payload, dst=peer, group=all_nccl)
                dist.recv(received, src=peer, group=all_nccl)
            else:
                dist.recv(received, src=peer, group=all_nccl)
                dist.send(payload, dst=peer, group=all_nccl)
            if not torch.all(received == peer + 1).item():
                raise RuntimeError('Cross-host PP payload mismatch')
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
    dist.barrier(); torch.cuda.synchronize()
    print(json.dumps({'gate': 'PASS', 'profile': args.profile, 'rank': rank,
                      'local_rank': local_rank, 'world_size': world,
                      'gloo_world_pass': True, 'local_tp4_pass': True,
                      'pp_peer': peer, 'pp_roundtrips': 5 if peer is not None else 0,
                      'pp_payload_bytes_per_rank_per_direction_per_round':
                      layout['payload_bytes_per_rank'],
                      'pp_elapsed_s': elapsed, 'gpu': torch.cuda.get_device_name(),
                      'epoch': time.time()}), flush=True)
    dist.destroy_process_group()
    timer.cancel()


if __name__ == '__main__':
    main()
