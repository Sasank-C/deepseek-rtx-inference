#!/usr/bin/env python3
"""Import pinned serving modules and parse the actual launcher, without weights.

Run in the pinned GPU image with BC_PROFILE=two_node_gpu or one_node_cpu.
Placeholder addresses are used only for parsing; no server/network is started.
"""
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shlex
import subprocess

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import vllm
assert vllm.__version__ == '0.32.0', vllm.__version__
assert metadata.version('flashinfer-python') == '0.7.0.post1'
for name in (
    'vllm.models.deepseek_v41.nvidia.model',
    'vllm.tokenizers.deepseek_v41',
    'vllm.reasoning.deepseek_v41_engine_reasoning_parser',
    'vllm.entrypoints.launchers.api_server.entry',
    'vllm.v1.executor.multiproc_executor',
):
    importlib.import_module(name)
    print(json.dumps({'dependency_import': 'PASS', 'module': name}), flush=True)
from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.argparse_utils import FlexibleArgumentParser

environment = os.environ.copy()
assert environment.get('BC_PROFILE') in ('two_node_gpu', 'one_node_cpu')
environment['DRY_RUN'] = '1'
environment.setdefault('NODE_RANK', '0')
environment.setdefault('LOCAL_IP', '192.0.2.10')
environment.setdefault('MASTER_IP', '192.0.2.10')
environment.setdefault('PRIVATE_IFACE', 'dryrun0')
command = subprocess.check_output(
    ['bash', str(Path(__file__).with_name('launch_dsv41.sh'))],
    env=environment, text=True, timeout=15)
arguments = shlex.split(command)
assert arguments[:2] == ['vllm', 'serve']
parsed = make_arg_parser(FlexibleArgumentParser(prog='vllm serve')).parse_args(arguments[2:])
validate_parsed_serve_args(parsed)
assert parsed.tensor_parallel_size == 4
assert parsed.max_model_len == 4096
assert parsed.max_num_seqs == 4
assert parsed.pipeline_parallel_size == (2 if environment['BC_PROFILE'] == 'two_node_gpu' else 1)
print(json.dumps({'launch_parser_gate': 'PASS', 'profile': environment['BC_PROFILE'],
    'node_rank': parsed.node_rank, 'headless': parsed.headless,
    'tp': parsed.tensor_parallel_size, 'pp': parsed.pipeline_parallel_size,
    'max_model_len': parsed.max_model_len, 'model_path': parsed.model_tag,
    'note': 'Imports and CLI parser only; no model config construction, download or serving.'}), flush=True)
