#!/usr/bin/env python3
"""Download the pinned checkpoint only after matching current-run evidence."""
import argparse
import json
import os
import re
from pathlib import Path
import runpy
import threading
import time


def admit(gate, guard, now, current_instance, deployed_digest=None):
    if gate.get("profile") not in ("tp4-pp2-gpu", "tp4-single-cpu"):
        raise ValueError("Select a reviewed profile")
    if not current_instance or str(gate.get("instance_id")) != current_instance:
        raise ValueError("Gate does not match this rental")
    if guard.get("status") != "armed" or str(guard.get("instance_id")) != current_instance:
        raise ValueError("Matching cleanup guard is not armed")
    stamp = guard.get("epoch", 0)
    if not isinstance(stamp, (int, float)) or not 0 <= now - stamp < 15:
        raise ValueError("Cleanup guard heartbeat is stale")
    if gate.get("deadline_epoch") != guard.get("deadline_epoch"):
        raise ValueError("Gate and guard deadline differ")
    parent = "sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34"
    if gate.get("runtime_base_digest") != parent:
        raise ValueError("Wrong parent runtime pin")
    deployed_digest = deployed_digest or os.getenv("DSV41_DEPLOYED_IMAGE_DIGEST", "")
    if (not re.fullmatch(r"sha256:[0-9a-f]{64}", deployed_digest)
            or deployed_digest == parent or gate.get("image_digest") != deployed_digest):
        raise ValueError("Require the actual derived image digest, verified from deployment metadata")
    required = ["hardware", "local_tp4_nccl", "kernel", "launch_parser", "external_cleanup_controller"]
    if gate["profile"] == "tp4-pp2-gpu":
        required += ["private_route", "bidirectional_transfer", "gloo_8rank", "nccl_8rank", "both_guards"]
    else:
        required += ["cpu_engram_pinned_memory"]
    for name in required:
        if gate.get("checks", {}).get(name) != "PASS":
            raise ValueError("Missing verified gate: " + name)
    if not isinstance(gate.get("completed_epoch"), (int, float)) or not 0 <= now - gate["completed_epoch"] < 900:
        raise ValueError("Gate evidence is stale")
    if not gate.get("evidence_files"):
        raise ValueError("Retain actual gate logs before signing admission")
    for path in gate["evidence_files"]:
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise ValueError("Missing evidence file: " + str(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", required=True, help="Operator-created admission record after reviewing real gate logs")
    p.add_argument("--guard-status", required=True)
    a = p.parse_args()
    gate = json.loads(Path(a.gate).read_text())
    guard = json.loads(Path(a.guard_status).read_text())
    admit(gate, guard, time.time(), os.getenv("CONTAINER_ID"))
    os.environ.update({"DEADLINE_EPOCH": str(int(gate["deadline_epoch"])),
                       "NETWORK_GATE_PASSED": "1", "KERNEL_GATE_PASSED": "1",
                       "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0"})
    # The low-level downloader checks the
    # absolute deadline and completed file sizes and uses one local_dir copy.
    # Credentials remain only in the independent guard, not model/download code.
    for name in ("CONTAINER_API_KEY", "VAST_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(name, None)
    def watch_guard():
        while True:
            time.sleep(3)
            try:
                current = json.loads(Path(a.guard_status).read_text())
                valid = (current.get("status") == "armed" and
                         current.get("instance_id") == guard["instance_id"] and
                         current.get("deadline_epoch") == guard["deadline_epoch"] and
                         0 <= time.time() - current.get("epoch", 0) < 15)
            except (OSError, ValueError, TypeError):
                valid = False
            if not valid:
                print("STOP: cleanup guard heartbeat unavailable", flush=True)
                os._exit(124)
    threading.Thread(target=watch_guard, daemon=True).start()
    runpy.run_path(str(Path(__file__).with_name("download_checkpoint.py")), run_name="__main__")


if __name__ == "__main__":
    main()
