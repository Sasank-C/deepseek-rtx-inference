#!/usr/bin/env python3
import json,os,struct
from pathlib import Path
root=Path(os.environ.get('MODEL_DIR','/workspace/models/dsv41-1017-heretic'))
pin=json.loads((root/'TRIAL_PIN.json').read_text())
assert pin['revision']=='12d58ad0878b5008e80d720251adb969a0069412'
index=json.loads((root/'model.safetensors.index.json').read_text())
assert set(index['weight_map'].values())==set(pin['files']) and len(pin['files'])==48
tensor_count=0
seen=set()
for name,size in pin['files'].items():
    p=root/name; assert p.stat().st_size==size,(name,'size')
    with p.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; assert n<50_000_000
        header=json.loads(f.read(n))
    for key,value in header.items():
        if key=='__metadata__':continue
        assert index['weight_map'][key]==name
        assert key not in seen,('duplicate tensor',key)
        seen.add(key)
        assert value['data_offsets'][1]+8+n<=size
        tensor_count+=1
assert seen==set(index['weight_map']), 'Index/header tensor coverage mismatch'
print(json.dumps({'checkpoint':'PASS','shards':48,'tensors':tensor_count,'revision':pin['revision']}))
