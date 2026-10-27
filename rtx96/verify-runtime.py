#!/usr/bin/env python3
"""Validate baked package identity; this does not execute GPU kernels."""
import argparse
import importlib.metadata as md
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build-only', action='store_true')
p.parse_args()
expected = {'vllm': '0.32.0', 'torch': '2.13.0+cu130',
            'flashinfer-python': '0.7.0.post1', 'flashinfer-cubin': '0.7.0.post1',
            'flashinfer-jit-cache': '0.7.0.post1+cu130',
            'flashinfer-jit-cache-sm120f': '0.7.0.post1+cu130'}
for name, version in expected.items():
    if md.version(name) != version:
        raise RuntimeError('Runtime version mismatch: ' + name)
manifest = json.loads(Path('/opt/dsv41/kernel-build-manifest.json').read_text())
for name, record in manifest['kernel_packages'].items():
    if md.version(name) != record['version']:
        raise RuntimeError('Kernel package version changed: ' + name)
print(json.dumps({'package_identity': 'PASS', 'gpu_execution': 'NOT_RUN',
                  'core': manifest['core'], 'target_sm': manifest['target_sm']}))
