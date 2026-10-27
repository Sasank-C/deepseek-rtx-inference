#!/usr/bin/env python3
"""Execute small real kernels at the model's TP4 dimensions, before weight download.
Not a model benchmark or a guarantee of complete-model correctness/capacity.
API/source pins: vLLM v0.32.0; FlashInfer v0.7.0.post1.
"""
import gc
import importlib.metadata as metadata
import json
import os
import tempfile
import time
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import vllm
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.utils.flashinfer import (
    has_flashinfer_sparse_mla_sm120_config,
    flashinfer_trtllm_batch_decode_sparse_mla_dsv4,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod, resolve
from vllm.scalar_type import scalar_types

deadline=int(os.environ['DEADLINE_EPOCH'])
tp=int(os.environ.get('TP_SIZE','4'))
assert tp in (1,2,4)
heads=64//tp
assert vllm.__version__.startswith('0.32.0'),vllm.__version__
assert metadata.version('flashinfer-python')=='0.7.0.post1',metadata.version('flashinfer-python')
assert torch.cuda.device_count()==tp,torch.cuda.device_count()
assert all(has_flashinfer_sparse_mla_sm120_config(heads,t) for t in (128,512))
torch.set_default_dtype(torch.bfloat16)

def sparse_gate():
    # DSv4 FP8 footer cache: 448 FP8 values +64 BF16 RoPE values/token;
    # all data rows precede 8-byte/token exponent footers in each64-token page.
    nb,bs=8,64
    kv=torch.randn(nb,bs,512,device='cuda',dtype=torch.bfloat16)/8
    nope=kv[...,:448].to(torch.float8_e4m3fn)
    rope=kv[...,448:].contiguous()
    data=torch.cat((nope.view(torch.uint8),rope.view(torch.uint8)),dim=-1)
    footer=torch.full((nb,bs,8),127,device='cuda',dtype=torch.uint8)
    footer[...,-1]=0
    packed=torch.cat((data.reshape(nb,-1),footer.reshape(nb,-1)),dim=1).reshape(nb,bs,1,584)
    reference_kv=torch.cat((nope.to(torch.bfloat16),rope),dim=-1).reshape(-1,512).float()
    workspace=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
    for topk in (128,512):
        q=torch.randn(1,heads,512,device='cuda',dtype=torch.bfloat16)/8
        ids=torch.arange(topk,device='cuda',dtype=torch.int32).reshape(1,-1)
        lens=torch.full((1,),topk,device='cuda',dtype=torch.int32)
        out=flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q.unsqueeze(1),swa_kv_cache=packed,workspace_buffer=workspace,
            sparse_indices=ids,seq_lens=torch.full_like(lens,nb*bs),swa_topk_lens=lens,
            bmm1_scale=512**-0.5,kv_layout='NHD')
        selected=reference_kv[:topk]
        scores=(q.float()@selected.T)*(512**-0.5)
        ref=(scores.softmax(-1)@selected).to(torch.bfloat16)
        torch.testing.assert_close(out.reshape_as(ref),ref,atol=5e-2,rtol=5e-2)
        assert torch.isfinite(out).all().item()
    return f'SM120 sparse MLA{heads}heads, FP8KV topk128/512 executed'

def marlin_gate():
    # Native TP4 intermediate576 zero-padded to640, hidden5120, packed E2M1.
    e,k,logical=4,5120,2304//tp
    n=((logical+127)//128)*128
    w13=torch.zeros((e,2*n,k//2),device='cuda',dtype=torch.uint8)
    w2=torch.zeros((e,k,n//2),device='cuda',dtype=torch.uint8)
    w13[:,:logical]=0x22; w13[:,n:n+logical]=0x22
    w2[:,:,:logical//2]=0x22
    s13=torch.full((e,2*n,k//32),120,device='cuda',dtype=torch.uint8)
    s2=torch.full((e,k,n//32),120,device='cuda',dtype=torch.uint8)
    cpu_owners=[]
    use_uva=os.environ.get('TRIAL_LAYOUT')=='sm120_cpu_pp20_tp1_uva60'
    if use_uva:
        from vllm.utils.platform_utils import is_uva_available
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        assert is_uva_available(),'UVA unavailable; do not fall back to full parameter copies'
        def to_uva(tensor):
            owner=tensor.to('cpu').pin_memory()
            assert owner.is_pinned()
            cpu_owners.append(owner)
            return get_accelerator_view_from_cpu_tensor(owner)
        # Real loader path: quantized source parameters already reside inUVA
        # before Marlin repacks them. Exercise that read, not only finalGEMM.
        w13,w2=to_uva(w13),to_uva(w2)
    a,b,sa,sb,ba,bb=prepare_moe_mxfp4_layer_for_marlin(
        SimpleNamespace(params_dtype=torch.bfloat16),w13,w2,s13,s2,None,None)
    if use_uva:
        # The loader restoresUVA after repacking replaces parameter tensors.
        for name,tensor in (('a',a),('b',b)):
            if name=='a':a=to_uva(tensor)
            else:b=to_uva(tensor)
    for batch in (1,4):
        x=torch.full((batch,k),1/128,device='cuda',dtype=torch.bfloat16)
        ids=torch.tensor([[0,1]]*batch,device='cuda',dtype=torch.int32)
        weights=torch.full((batch,2),0.5,device='cuda',dtype=torch.float32)
        y=fused_marlin_moe(x,a,b,ba,bb,sa,sb,weights,ids,
                          scalar_types.float4_e2m1f.id,global_num_experts=e)
        gate=torch.tensor(k/(128*128),device='cuda',dtype=torch.float32)
        expected=(F.silu(gate)*gate*logical/128).item()
        torch.testing.assert_close(y.float(),torch.full_like(y.float(),expected),atol=0.015,rtol=0.05)
        assert torch.isfinite(y).all().item()
    return f'MXFP4 Marlin MoE5120x{logical} padded{n}, nonzero weights executed; UVA={bool(cpu_owners)}'

def fp8_linear_gate():
    names=[]
    # TP4 shared gate/up and down shapes, including the non-128 K576 case.
    for input_dim,output_dim in ((5120,2*(2304//tp)),(2304//tp,5120)):
        spec,ctx,fmt=resolve('MXFP8',None,'trial')
        method=ModelOptLinearMethod(spec,ctx,fmt)
        layer=torch.nn.Module()
        with torch.device('cuda'):
            method.create_weights(layer,input_dim,[output_dim],input_dim,output_dim,
                                  torch.bfloat16,weight_loader=lambda *a,**k:None)
        layer.weight.data.copy_(torch.full(layer.weight.shape,1/32,device='cuda',dtype=torch.bfloat16))
        # SerializedMXFP8 e8m0 exponent127 means scale1, not numeric uint8=1.
        layer.weight_scale.data.view(torch.uint8).fill_(127)
        method.process_weights_after_loading(layer)
        kernel=type(method.kernel).__name__
        assert 'emulat' not in kernel.lower(),('Refuse unbudgeted native-MXFP8 emulation',kernel)
        x=torch.full((4,input_dim),1/32,device='cuda',dtype=torch.bfloat16)
        y=method.apply(layer,x)
        ref=torch.full((4,output_dim),input_dim/1024,device='cuda',dtype=torch.bfloat16)
        torch.testing.assert_close(y,ref,atol=0.05,rtol=0.05)
        assert torch.isfinite(y).all().item()
        names.append(kernel)
    return names

with torch.inference_mode(),set_current_vllm_config(VllmConfig()):
    # ModelOpt parameters require TP rank metadata even in a standalone kernel
    # fixture. A singleton local Gloo/FileStore group performs no cross-host
    # communication and does not substitute for the separate NCCL network gate.
    torch.cuda.set_device(0)
    init_distributed_environment(world_size=1,rank=0,local_rank=0,backend='gloo',
        distributed_init_method='file://'+tempfile.mkdtemp(prefix='dsv41-kernel-')+'/store')
    initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1,
                              backend='gloo')
    for device in range(tp):
        assert time.time()<deadline-900,'Preserve at least15min after kernel probes'
        torch.cuda.set_device(device)
        assert torch.cuda.get_device_capability(device)==(12,0)
        props=torch.cuda.get_device_properties(device)
        started=time.monotonic()
        sparse=sparse_gate()
        print(json.dumps({'subgate':'sparse','result':'PASS','details':sparse}),flush=True)
        marlin=marlin_gate()
        print(json.dumps({'subgate':'marlin','result':'PASS','details':marlin}),flush=True)
        fp8=fp8_linear_gate()
        torch.cuda.synchronize()
        print(json.dumps({'kernel_gate':'PASS','gpu_index':device,'gpu':props.name,
                          'memory_bytes':props.total_memory,'sparse':sparse,'marlin':marlin,
                          'mxfp8_linear_kernels':fp8,'seconds':time.monotonic()-started}),flush=True)
        gc.collect(); torch.cuda.empty_cache()
print('ALL_KERNEL_GATES_PASS',flush=True)
