#!/usr/bin/env python3
"""Read-only startup inventory. Never downloads weights or allocates GPU tensors."""
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"exit_code": p.returncode, "stdout": p.stdout[:20000],
                "stderr": p.stderr[:2000]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": type(e).__name__}


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def main():
    packages = {}
    for name in ("vllm", "torch", "flashinfer-python", "huggingface-hub", "hf-xet", "vastai"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    value = {
        "epoch": time.time(), "phase": "inventory_only_no_model_loaded",
        "instance_id": os.getenv("CONTAINER_ID"), "packages": packages,
        "gpu": command(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap", "--format=csv,noheader,nounits"]),
        "topology": command(["nvidia-smi", "topo", "-m"]),
        "meminfo": read("/proc/meminfo"),
        "cgroup_memory_max": read("/sys/fs/cgroup/memory.max"),
        "cgroup_memory_current": read("/sys/fs/cgroup/memory.current"),
        "cgroup_v1_limit": read("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        "memlock_limits_bytes": resource.getrlimit(resource.RLIMIT_MEMLOCK),
        "workspace_disk": dict(zip(("total", "used", "free"), shutil.disk_usage("/workspace"))),
        "shared_memory": dict(zip(("total", "used", "free"), shutil.disk_usage("/dev/shm"))),
    }
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
