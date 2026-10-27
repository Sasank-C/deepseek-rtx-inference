"""Pure absolute download-stop decision; completed-shard ETA is informational."""


def download_decision(*, now, elapsed, have, initial, total, new_shards, deadline, reserve):
    speed = max(0, have - initial) / max(1, elapsed)
    eta = max(0, total - have) / speed if speed > 0 else None
    # Completed-only progress is censored. This checkpoint has two final
    # ~101.5GB Engram shards; no fixed shard-count/time threshold makes that
    # estimate a reliable abort criterion. Preserve the absolute original stop.
    eta_ready = False
    reason = None
    if now >= deadline - reserve:
        reason = "absolute_download_deadline"
    return {"MB_s": speed / 1e6, "eta_s": eta, "eta_ready": eta_ready,
            "eta_mode": "completed_shards_informational_only",
            "new_completed_shards": new_shards, "stop_reason": reason}
