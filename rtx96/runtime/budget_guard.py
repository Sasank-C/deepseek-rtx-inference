#!/usr/bin/env python3
"""Pinned, self-only Vast cleanup guard. Credentials never enter argv or logs.

Run inside an authorized rented container or VM. `check` is read-only. `watch` destroys ONLY the
validated current instance at the pinned deadline/cost/byte ceiling; it retries
and attempts stop as fallback. API or VM failure can defeat this guard: root must
also verify cleanup in the UI. It is not a provider-enforced spending limit.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

MAX_TRIAL_SECONDS = 4 * 3600
MAX_NODE_CAP = 14.73  # Conservative per-node template ceiling; quote all costs separately.


def require_scoped_identity(expected_instance=None, environment=None):
    """Validate injected self identity without returning or logging a key."""
    environment = os.environ if environment is None else environment
    current = environment.get("CONTAINER_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", current):
        raise ValueError("Required injected CONTAINER_ID is unavailable")
    if expected_instance is not None and current != str(expected_instance):
        raise ValueError("Instance ID does not match injected ID")
    scoped = environment.get("CONTAINER_API_KEY", "")
    if not scoped.strip() or "\x00" in scoped:
        raise ValueError("Required injected scoped credential is unavailable")


def reject_saved_cli_credentials(environment):
    # Current CLI can retry with a saved account key after a 2FA error, even
    # when VAST_API_KEY was supplied. It also imports the legacy key file.
    # Refuse these files without reading, moving or deleting their contents.
    guard_home = Path.home()
    configs = {guard_home / ".config"}
    if environment.get("XDG_CONFIG_HOME"):
        configs.add(Path(environment["XDG_CONFIG_HOME"]))
    candidates = [guard_home / ".vast_api_key"]
    candidates += [base / "vastai" / name for base in configs
                   for name in ("vast_api_key", "vast_tfa_key")]
    if any(path.exists() or path.is_symlink() for path in candidates):
        raise ValueError("Saved Vast credentials present; scoped-only guard refuses fallback")


def cli_environment():
    environment = os.environ.copy()
    require_scoped_identity(environment=environment)
    reject_saved_cli_credentials(environment)
    # Never fall back to inherited account authentication. Keep the scoped key
    # in process memory only; suppress CLI output and automatic update checks.
    environment["VAST_API_KEY"] = environment["CONTAINER_API_KEY"]
    environment["VASTAI_NO_UPDATE_CHECK"] = "1"
    return environment


def cli(executable, words, capture=False):
    try:
        r = subprocess.run([executable, *words], stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=12, check=False,
                           env=cli_environment())
        if r.returncode != 0: return None
        if not capture: return True
        if len(r.stdout) > 2_000_000: return None
        return json.loads(r.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def supports_noninteractive_destroy(executable):
    # Help is read-only. Current CLI otherwise prompts and can abort with exit 0.
    try:
        result = subprocess.run([executable, "destroy", "instance", "--help"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=12, check=False,
                                env=cli_environment())
        return result.returncode == 0 and b"--yes" in result.stdout
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return False


def self_info(executable, ident, expected_machine):
    # Raw instance metadata may contain credentials; only selected values escape.
    value = cli(executable, ["show", "instance", str(ident), "--raw"], True)
    if isinstance(value, dict) and isinstance(value.get("instances"), dict):
        value = value["instances"]
    if not isinstance(value, dict): return None
    try:
        valid = int(value.get("id", -1)) == ident and int(value.get("machine_id", -1)) == expected_machine
    except (ValueError, TypeError): return None
    if not valid: return None
    return {"instance_id": ident, "machine_id": expected_machine,
            "actual_status": str(value.get("actual_status", "unknown"))[:32]}


def counters(interface):
    base = Path("/sys/class/net") / interface / "statistics"
    return int((base / "rx_bytes").read_text()), int((base / "tx_bytes").read_text())


def estimated_cost(now, start, hourly, rx, tx, ingress, egress):
    # Decimal GB is deliberately conservative when provider uses GiB.
    return max(0, now - start) / 3600 * hourly + rx / 1e9 * ingress + tx / 1e9 * egress


def reason_at(now, state, rx, tx, prior_rx, prior_tx, mono_expired=False):
    if mono_expired or now >= state["deadline"]: return "deadline"
    if rx < prior_rx or tx < prior_tx: return "network_counter_reset"
    if rx > state["rx_limit"] or tx > state["tx_limit"]: return "transfer_limit"
    cost = estimated_cost(now, state["started"], state["hourly"], rx, tx,
                          state["ingress"], state["egress"])
    if cost >= state["node_cap"] - 0.25: return "estimated_cost_ceiling"
    return None


def write_status(path, value):
    try:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.chmod(temporary, 0o600); temporary.replace(path)
    except OSError:
        # A full/read-only disk must not suppress cleanup.
        pass


def cleanup(executable, ident, status_path, reason):
    attempts = 0
    while True:
        attempts += 1
        write_status(status_path, {"status": "cleanup_requested_unverified",
                     "reason": reason, "instance_id": ident,
                     "attempt": attempts, "epoch": time.time()})
        cli(executable, ["destroy", "instance", str(ident), "--yes"])
        # Allow transient destroy failures two retries before stop can terminate
        # this very watchdog. A stopped VM still needs deletion through the UI.
        # No exit code proves deletion, so never emit a successful-cleanup state.
        if attempts >= 3:
            cli(executable, ["stop", "instance", str(ident)])
        time.sleep(10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["check", "watch"])
    p.add_argument("--instance", type=int, required=True)
    p.add_argument("--machine", type=int, required=True)
    p.add_argument("--cli", default="vastai")
    p.add_argument("--interface", required=True, help="Actual external uplink, not the VPN")
    p.add_argument("--started", type=float, required=True, help="Original rental creation epoch")
    p.add_argument("--deadline", type=float, required=True, help="Original absolute deadline")
    p.add_argument("--hourly", type=float, required=True)
    p.add_argument("--ingress", type=float, required=True)
    p.add_argument("--egress", type=float, required=True)
    p.add_argument("--node-cap", type=float, default=MAX_NODE_CAP)
    p.add_argument("--rx-limit", type=int, default=700_000_000_000)
    p.add_argument("--tx-limit", type=int, default=25_000_000_000)
    p.add_argument("--state", default="/var/lib/dsv41-guard/state.json")
    a = p.parse_args(); now = time.time()
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", a.interface) or a.interface in ["lo", "dsv41"]:
        p.error("Specify the actual external uplink")
    if a.instance <= 0 or a.machine <= 0: p.error("Positive IDs required")
    for field in ["started", "deadline", "hourly", "ingress", "egress", "node_cap"]:
        if not math.isfinite(getattr(a, field)) or getattr(a, field) < 0: p.error("Invalid numeric bound")
    if not 0 < a.deadline - a.started <= MAX_TRIAL_SECONDS or a.started > now + 30:
        p.error("Deadline must be within 4 hours of original rental creation")
    if not 0 < a.node_cap <= MAX_NODE_CAP or a.rx_limit <= 0 or a.tx_limit <= 0:
        p.error("Node cap exceeds this template ceiling")
    try:
        require_scoped_identity(a.instance)
        reject_saved_cli_credentials(os.environ)
    except ValueError as exc:
        p.error(str(exc))
    executable = shutil.which(a.cli)
    if not executable: p.error("Official CLI unavailable; do not begin model work")
    state_path = Path(a.state)
    state_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if state_path.is_symlink(): p.error("State path cannot be a symlink")
    with open(str(state_path) + ".lock", "a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: p.error("Guard already active")
        state = {k: getattr(a, k) for k in ["instance", "machine", "interface", "started", "deadline", "hourly", "ingress", "egress", "node_cap", "rx_limit", "tx_limit"]}
        already_pinned = state_path.exists()
        if already_pinned:
            try: old = json.loads(state_path.read_text())
            except (ValueError, OSError): p.error("Invalid pinned state; do not overwrite")
            if old != state: p.error("Pinned state differs; never extend/reset an existing trial")
        if (a.mode == "check" or not already_pinned) and not supports_noninteractive_destroy(executable):
            p.error("CLI must support destroy instance --yes; do not begin model work")
        authenticated = self_info(executable, a.instance, a.machine) is not None
        if not authenticated:
            if a.mode == "watch" and already_pinned:
                # check previously authenticated these exact immutable bounds.
                # A later read failure must not disable best-effort destruction.
                cleanup(executable, a.instance, state_path.with_name("status.json"),
                        "self_auth_unavailable_on_restart")
                return
            p.error("Read-only self-authentication failed; do not begin model work")
        if not already_pinned:
            write_status(state_path, state)
            if not state_path.exists(): p.error("Could not persist pinned bounds")
        try:
            rx, tx = counters(a.interface)
        except (OSError, ValueError):
            if a.mode == "watch":
                cleanup(executable, a.instance, state_path.with_name("status.json"),
                        "counter_read_failed_at_startup")
                return
            p.error("Cannot read uplink counters; do not begin model work")
        # Authentication can take up to 12 seconds; use a fresh deadline check.
        now = time.time()
        reason = reason_at(now, state, rx, tx, rx, tx)
        if a.mode == "check":
            if reason: p.error("Budget already exhausted; clean up through root")
            print(json.dumps({"self_auth": "passed", "instance_id": a.instance,
                              "machine_id": a.machine, "deadline_epoch": a.deadline,
                              "remaining_seconds": max(0, a.deadline - now)}))
            return
        status_path = state_path.with_name("status.json")
        mono_deadline = time.monotonic() + max(0, a.deadline - now)
        prior_rx, prior_tx = rx, tx
        if status_path.exists():
            try:
                prior = json.loads(status_path.read_text())
                if prior.get("status") == "cleanup_requested_unverified":
                    reason = "resume_cleanup"
                elif prior.get("instance_id") == a.instance:
                    prior_rx = int(prior.get("rx_bytes", rx))
                    prior_tx = int(prior.get("tx_bytes", tx))
                    reason = reason_at(now, state, rx, tx, prior_rx, prior_tx)
            except (OSError, ValueError, TypeError):
                reason = "invalid_previous_status"
        while not reason:
            write_status(status_path, {"status": "armed", "instance_id": a.instance,
                         "deadline_epoch": a.deadline, "epoch": time.time(),
                         "rx_bytes": rx, "tx_bytes": tx,
                         "estimated_node_cost": estimated_cost(time.time(), a.started, a.hourly, rx, tx, a.ingress, a.egress)})
            time.sleep(3)
            try: rx, tx = counters(a.interface)
            except (OSError, ValueError): reason = "counter_read_failed"; break
            reason = reason_at(time.time(), state, rx, tx, prior_rx, prior_tx,
                               time.monotonic() >= mono_deadline)
            prior_rx, prior_tx = rx, tx
        cleanup(executable, a.instance, status_path, reason)


if __name__ == "__main__":
    main()
