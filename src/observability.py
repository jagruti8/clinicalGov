"""Node-level latency timing and optional JSONL event logging."""
import functools
import json
import logging
import os
import time

logger = logging.getLogger("clinicalgov")


def log_event(event: dict) -> None:
    """Append an event to $PIPELINE_LOG_PATH (JSON lines) when that variable is set."""
    path = os.environ.get("PIPELINE_LOG_PATH")
    if not path:
        return
    try:
        with open(path, "a") as handle:
            handle.write(json.dumps({"ts": time.time(), **event}) + "\n")
    except OSError as exc:
        logger.warning("could not write pipeline log: %s", exc)


def timed(name: str, fn):
    """Wrap a graph node: records seconds spent under `name` in state['timings']."""
    @functools.wraps(fn)
    def wrapper(state):
        start = time.perf_counter()
        update = dict(fn(state) or {})
        seconds = time.perf_counter() - start
        timings = dict(state.get("timings") or {})
        timings[name] = round(timings.get(name, 0.0) + seconds, 4)
        update["timings"] = timings
        log_event({"node": name, "seconds": round(seconds, 4)})
        return update
    return wrapper
