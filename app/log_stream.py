"""
Global SSE log streaming module.

Provides a single shared queue that Python backend code pushes log messages
into. A Flask SSE endpoint drains this queue and streams messages to the
browser in real time.

Single-user design: one global queue, no session isolation.
"""

import json
import queue
import threading
import time
from typing import Generator


# ============================================================================
# Global State
# ============================================================================

# Thread-safe queue. maxsize=0 means unlimited.
_log_queue: queue.Queue = queue.Queue(maxsize=0)

# Track whether any SSE client is currently connected.
# Used to avoid filling the queue when no one is listening.
_client_connected: bool = False
_client_lock = threading.Lock()


# ============================================================================
# Public API
# ============================================================================

def push(message: str, level: str = "INFO", source: str = "") -> None:
    """
    Push a log message into the SSE stream queue.

    Call this from any Python backend code you want to appear in the browser.
    If no SSE client is connected, messages are silently discarded to avoid
    unbounded queue growth.
    """
    with _client_lock:
        connected = _client_connected

    if not connected:
        return

    payload = {
        "message": message.rstrip("\n"),
        "level": level.lower(),
        "source": source,
        "ts": time.strftime("%H:%M:%S"),
    }

    try:
        _log_queue.put_nowait(json.dumps(payload))
    except queue.Full:
        # Unlimited queue should not fill; keep the logging path non-fatal.
        pass


def push_separator(label: str = "") -> None:
    """Push a visual separator line into the stream."""
    msg = f"{'-' * 20} {label} {'-' * 20}" if label else "-" * 50
    push(msg, level="debug")


def clear() -> None:
    """Drain the queue without sending. Called when a new query starts."""
    while not _log_queue.empty():
        try:
            _log_queue.get_nowait()
        except queue.Empty:
            break


# ============================================================================
# SSE Generator (consumed by Flask route)
# ============================================================================

def stream_generator(
    poll_interval: float = 0.1,
    timeout: float = 600.0,
) -> Generator[str, None, None]:
    """
    Yield SSE-formatted strings from the log queue.

    Runs until a done sentinel is seen, timeout elapses, or the client
    disconnects.
    """
    global _client_connected

    with _client_lock:
        _client_connected = True

    clear()  # Discard stale messages from a previous run.

    deadline = time.time() + timeout
    last_activity = time.time()

    try:
        # Immediate keepalive confirms the stream is open to the browser.
        yield ": keepalive\n\n"

        while time.time() < deadline:
            try:
                raw = _log_queue.get(timeout=poll_interval)
            except queue.Empty:
                # Periodic ping keeps proxies/TCP connections alive.
                if time.time() - last_activity > 15:
                    yield ": ping\n\n"
                    last_activity = time.time()
                continue

            last_activity = time.time()

            # Sentinel: backend signaled operation completion.
            if raw == "__DONE__":
                yield "event: done\ndata: {}\n\n"
                break

            yield f"data: {raw}\n\n"

    finally:
        with _client_lock:
            _client_connected = False


def signal_done() -> None:
    """Signal the active SSE stream that the current operation is complete."""
    with _client_lock:
        connected = _client_connected

    # Avoid unbounded sentinel buildup when no browser stream is attached.
    if connected:
        _log_queue.put("__DONE__")


# ============================================================================
# stderr Interceptor
# ============================================================================

def _infer_level(line: str) -> str:
    """Infer a display level from message content."""
    lower = line.lower()
    if any(k in lower for k in ("error", "failed", "exception", "traceback")):
        return "error"
    if any(k in lower for k in ("warning", "warn")):
        return "warning"
    if any(k in lower for k in ("complete", "success", "done", "finished")):
        return "success"
    if any(k in lower for k in ("===", "starting", "begin", "---")):
        return "primary"
    return "info"


class StderrTee:
    """
    Wrap sys.stderr so that each write also forwards lines to log stream push().
    """

    # Only forward lines containing at least one of these strings.
    INCLUDE_PREFIXES = [
        "[ADVERSARIAL]",
        "[TIER0]",
        "[RAG]",
        "[LLM]",
        "[TIERED]",
        "[DEBUG]",
        "[INFO]",
        "[WARNING]",
        "[ERROR]",
    ]

    def __init__(self, original_stderr):
        self._original = original_stderr
        self._buffer = ""

    def write(self, text: str) -> int:
        # Preserve current CLI/stderr behavior.
        self._original.write(text)

        # Emit complete lines only.
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if not line:
                continue

            if self.INCLUDE_PREFIXES and not any(
                prefix in line for prefix in self.INCLUDE_PREFIXES
            ):
                continue

            push(line, level=_infer_level(line))

        return len(text)

    def flush(self) -> None:
        self._original.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()
