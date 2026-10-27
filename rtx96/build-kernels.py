#!/usr/bin/env python3
"""Keep exact inherited kernels or install only missing, hash-pinned wheels.

Runs at image-build time without a GPU. Inherited wheel archives are not
available: their installed files are checked against RECORD, then SHA256 is
recorded. Archive verification is claimed only for wheels downloaded here.
"""
import base64
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlopen
from urllib.parse import urlparse, unquote

ROOT = Path('/opt/dsv41')
EXPECTED = {'vllm': '0.32.0', 'torch': '2.13.0+cu130',
            'flashinfer-python': '0.7.0.post1'}


def normalized(name):
    return name.lower().replace('_', '-').replace('.', '-')


def inventory():
    return {normalized(d.metadata['Name']): d.version for d in md.distributions()}


def core():
    actual = {name: md.version(name) for name in EXPECTED}
    if actual != EXPECTED:
        raise RuntimeError(f'Unexpected core runtime: {actual}')
    return actual


def missing_packages(packages, installed):
    missing = []
    for package in packages:
        actual = installed.get(normalized(package['name']))
        if actual is None:
            missing.append(package)
        elif actual != package['version']:
            raise RuntimeError(f"Refusing to change inherited {package['name']} "
                               f"from {actual} to {package['version']}")
    return missing


def verify_distribution(d):
    """Verify RECORD hashes/sizes and separately record SHA256 for each file."""
    name = normalized(d.metadata['Name'])
    if d.read_text('RECORD') is None:
        raise RuntimeError(f'No installed RECORD for {name}')
    entries, unhashed = {}, []
    verified_count = 0
    for item in d.files or []:
        p = Path(d.locate_file(item))
        if p.suffix == '.pyc':
            continue  # Generated artifacts are not wheel payloads.
        if not p.is_file():
            raise RuntimeError(f'Missing installed file: {name}: {item}')
        if item.size is not None and p.stat().st_size != item.size:
            raise RuntimeError(f'RECORD size mismatch: {name}: {item}')
        sha256 = hashlib.sha256()
        recorded = item.hash
        if recorded and recorded.mode not in ('sha256', 'sha384', 'sha512'):
            raise RuntimeError(f'Unsupported RECORD hash: {name}: {item}')
        record_hash = (sha256 if recorded and recorded.mode == 'sha256'
                       else hashlib.new(recorded.mode) if recorded else None)
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                sha256.update(chunk)
                if record_hash is not None and record_hash is not sha256:
                    record_hash.update(chunk)
        entries[str(item)] = sha256.hexdigest()
        if recorded:
            actual = base64.urlsafe_b64encode(record_hash.digest()).rstrip(b'=').decode()
            if actual != recorded.value:
                raise RuntimeError(f'RECORD hash mismatch: {name}: {item}')
            verified_count += 1
        else:
            # RECORD itself and pip-generated metadata often have no RECORD hash.
            unhashed.append(str(item))
    if not verified_count:
        raise RuntimeError(f'No RECORD-hashed installed files for {name}')
    return {'version': d.version, 'installed_files_sha256': entries,
            'record_hash_verified_files': verified_count,
            'record_unhashed_files': unhashed}


def main():
    core()
    before = inventory()
    lock = json.loads((ROOT / 'flashinfer-wheel-lock.json').read_text())
    missing = missing_packages(lock['packages'], before)
    downloads = []
    if missing:
        # Only missing providers are installed. --no-deps prevents dependency
        # resolution from replacing Torch, vLLM or any existing provider.
        with tempfile.TemporaryDirectory(prefix='dsv41-kernels-') as temporary:
            wheels = []
            for package in missing:
                wheel = Path(temporary) / unquote(Path(urlparse(package['url']).path).name)
                digest = hashlib.sha256()
                with urlopen(package['url'], timeout=120) as response, wheel.open('wb') as target:
                    while chunk := response.read(8 * 1024 * 1024):
                        target.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != package['sha256']:
                    raise RuntimeError('Wheel archive hash mismatch: ' + package['name'])
                downloads.append({**package, 'verified_archive_sha256': digest.hexdigest(),
                                  'archive_bytes': wheel.stat().st_size})
                wheels.append(str(wheel))
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                            '--no-index', *wheels], check=True)
    core()
    after = inventory()
    for name, version in before.items():
        if after.get(name) != version:
            raise RuntimeError(f'Kernel installation modified inherited {name}')
    if missing_packages(lock['packages'], after):
        raise RuntimeError('Expected exact cubin, JIT shim and SM120f provider absent')

    downloaded_names = {normalized(p['name']) for p in downloads}
    records = {}
    for d in md.distributions():
        name = normalized(d.metadata['Name'])
        if not name.startswith('flashinfer-'):
            continue
        if name == 'flashinfer-cubin' and d.version != '0.7.0.post1':
            raise RuntimeError('Wrong cubin version')
        if name.startswith('flashinfer-jit-cache') and d.version != '0.7.0.post1+cu130':
            raise RuntimeError('Wrong JIT cache version')
        record = verify_distribution(d)
        record['origin'] = ('downloaded_hash_pinned_wheel' if name in downloaded_names
                            else 'inherited_pinned_parent_image')
        record['original_wheel_archive_verified_here'] = name in downloaded_names
        records[name] = record

    (ROOT / 'kernel-build-manifest.json').write_text(json.dumps({
        'status': 'packages_verified_no_gpu_execution', 'core': core(),
        'target_sm': 'sm120f', 'before_packages': before, 'after_packages': after,
        'installation': 'missing_only' if downloads else 'skipped_all_required_exact',
        'locked_required_packages': lock['packages'],
        'verified_wheel_archives': downloads, 'kernel_packages': records,
        'hash_kind': 'installed RECORD verification and file SHA256; archive SHA256 only for downloaded wheels',
        'note': 'Inherited original wheel archives were not rehashed. No measured speedup or GPU execution.'
    }, indent=2) + '\n')
    print(json.dumps({'installation': 'missing_only' if downloads else 'skipped_all_required_exact',
                      'downloaded_wheels': len(downloads), 'verified_kernel_distributions': len(records),
                      'gpu_execution': 'NOT_RUN'}))


if __name__ == '__main__':
    main()
