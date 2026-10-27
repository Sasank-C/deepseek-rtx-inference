#!/usr/bin/env python3
"""Start the reviewed profile only with fresh evidence and an armed guard."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from download_gated import admit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", required=True)
    p.add_argument("--guard-status", required=True)
    a = p.parse_args()
    gate = json.loads(Path(a.gate).read_text())
    guard = json.loads(Path(a.guard_status).read_text())
    admit(gate, guard, time.time(), os.getenv("CONTAINER_ID"))
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / "validate_checkpoint.py")], check=True)
    env = os.environ.copy()
    for name in ("CONTAINER_API_KEY", "VAST_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(name, None)
    env.update({"BC_PROFILE": {"tp4-pp2-gpu": "two_node_gpu", "tp4-single-cpu": "one_node_cpu"}[gate["profile"]],
                "BC_KERNEL_GATES_PASSED": "1", "BC_CHECKPOINT_VALIDATED": "1",
                "BC_NETWORK_GATE_PASSED": "1", "BC_CPU_MEMORY_GATE_PASSED": "1",
                "DEADLINE_EPOCH": str(int(gate["deadline_epoch"]))})
    # Keep this parent alive to terminate its own workload if the independent
    # guard becomes stale. Destruction remains the guard/controller's job.
    process = subprocess.Popen(["bash", str(here / "launch_dsv41.sh")], env=env, start_new_session=True)
    try:
        while process.poll() is None:
            current = json.loads(Path(a.guard_status).read_text())
            if (current.get("status") != "armed" or
                current.get("instance_id") != guard["instance_id"] or
                current.get("deadline_epoch") != guard["deadline_epoch"] or
                not 0 <= time.time() - current.get("epoch", 0) < 15):
                raise RuntimeError("Cleanup guard heartbeat failed; terminating this model process group")
            time.sleep(3)
        raise SystemExit(process.returncode)
    finally:
        if process.poll() is None:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
