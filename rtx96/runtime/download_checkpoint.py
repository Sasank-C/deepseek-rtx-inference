#!/usr/bin/env python3
"""Pinned full native checkpoint; one copy, informational ETA, absolute stop."""
import json
import os
from pathlib import Path
import shutil
import threading
import time
from huggingface_hub import HfApi, snapshot_download
from download_eta import download_decision

MODEL = 'pqhaz/DeepSeek-V4.1-Flash-1017-heretic'
REV = '12d58ad0878b5008e80d720251adb969a0069412'
root = Path(os.environ.get('MODEL_DIR','/workspace/models/dsv41-1017-heretic'))
deadline = int(os.environ['DEADLINE_EPOCH'])
reserve = int(os.environ.get('LOAD_BENCH_RESERVE_SECONDS','2100'))
assert os.environ.get('HF_XET_CHUNK_CACHE_SIZE_BYTES') == '0'
assert os.environ.get('NETWORK_GATE_PASSED') == '1', 'Require successful BOTH-node network/NCCL gate'
assert os.environ.get('KERNEL_GATE_PASSED') == '1', 'Require successful BOTH-node real kernel gate'
if os.environ.get('TRIAL_LAYOUT')=='sm120_cpu_pp20_tp1_uva60':
    assert os.environ.get('CPU_MEMORY_GATE_PASSED')=='1','Require BOTH-node pinned CPU memory gate'
assert time.time()+reserve+300 < deadline, 'Insufficient time left for full-model trial'
root.mkdir(parents=True,exist_ok=True)
info = HfApi().model_info(MODEL, revision=REV, files_metadata=True)
assert info.sha == REV
files = {f.rfilename:f.size for f in info.siblings
         if f.rfilename.startswith('model-') and f.rfilename.endswith('.safetensors') and '/' not in f.rfilename}
assert len(files)==48 and all(isinstance(v,int) for v in files.values())
total=sum(files.values())
existing=sum((root/k).stat().st_size for k in files if (root/k).exists())
free=shutil.disk_usage(root).free
assert existing or free>=600_000_000_000, 'Require600GB free after image/runtime for a fresh download'
assert free >= total-existing+65_000_000_000, (free,total,existing,'Insufficient post-image disk')
start=time.monotonic(); stop=threading.Event()
def completed_progress():
    # Xet can preallocate partials to full logical size. Only atomically
    # completed final shard names count as downloaded; ETA is conservative.
    total_local=0; names=set()
    for name,expected in files.items():
        p=root/name
        try:
            if p.is_file() and p.stat().st_size==expected:
                total_local+=expected; names.add(name)
        except FileNotFoundError:
            pass
    return total_local,names
initial,initial_names=completed_progress()
def monitor():
    while not stop.wait(20):
        now=time.time(); elapsed=time.monotonic()-start; have,names=completed_progress()
        decision=download_decision(now=now,elapsed=elapsed,have=have,initial=initial,
            total=total,new_shards=len(names-initial_names),deadline=deadline,reserve=reserve)
        print(json.dumps({'phase':'download','bytes_completed_shards':have,'bytes_target':total,
                          **decision,'seconds_left':deadline-now}),flush=True)
        if decision['stop_reason']:
            print('STOP: '+decision['stop_reason'],flush=True)
            os._exit(124)
threading.Thread(target=monitor,daemon=True).start()
try:
    snapshot_download(MODEL,revision=REV,local_dir=str(root),max_workers=4,
        allow_patterns=['model-*.safetensors','model.safetensors.index.json','config.json',
                        'tokenizer*','special_tokens_map.json','generation_config.json',
                        'encoding/*','chat_template*','LICENSE'])
finally:
    stop.set()
for name,size in files.items():
    assert (root/name).stat().st_size==size,(name,size)
(root/'TRIAL_PIN.json').write_text(json.dumps({'model':MODEL,'revision':REV,'files':files},indent=2))
print(json.dumps({'download':'PASS','shards':48,'bytes':total,'elapsed_s':time.monotonic()-start}),flush=True)
