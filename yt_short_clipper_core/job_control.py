"""Thread-safe cooperative job cancellation and subprocess tracking."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from typing import Any

_SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class JobCancelledError(RuntimeError):
    """Raised when the user cancels an in-flight sidecar job."""


_cancel_event = threading.Event()
_procs_lock = threading.Lock()
_active_procs: set[subprocess.Popen[Any]] = set()


def reset_cancel() -> None:
    """Clear cancellation state at the start of a new job."""
    _cancel_event.clear()


def get_cancel_event() -> threading.Event:
    """Return the shared cancellation event."""
    return _cancel_event


def is_cancelled() -> bool:
    """Return True if cancellation has been requested."""
    return _cancel_event.is_set()


def check_cancelled(message: str = "Proses dibatalkan oleh pengguna.") -> None:
    """Raise ``JobCancelledError`` when cancellation has been requested."""
    if _cancel_event.is_set():
        raise JobCancelledError(message)


def request_cancel() -> int:
    """Signal cancellation and terminate all active child subprocesses.

    Returns the number of subprocesses that were signaled to terminate.
    """
    _cancel_event.set()
    killed = 0
    with _procs_lock:
        procs = list(_active_procs)
    for proc in procs:
        try:
            if proc.poll() is None:
                proc.terminate()
                killed += 1
        except OSError:
            pass
    return killed


def run_cancellable_subprocess(
    cmd: list[str],
    *,
    timeout: float | None = None,
    cwd: str | None = None,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> subprocess.CompletedProcess[str]:
    """Run *cmd* while respecting ``request_cancel()``.

    Registers the spawned process so ``request_cancel()`` can terminate it
    immediately without leaving orphan ``ffmpeg.exe`` processes behind.
    """
    check_cancelled()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding=encoding,
        errors=errors,
        cwd=cwd,
        creationflags=_SUBPROCESS_FLAGS,
    )
    with _procs_lock:
        _active_procs.add(proc)

    deadline = (time.monotonic() + timeout) if timeout is not None else None
    try:
        while True:
            if _cancel_event.is_set():
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                raise JobCancelledError("Proses dibatalkan oleh pengguna.")

            wait_slice = 0.2
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    proc.kill()
                    out, err = proc.communicate()
                    raise subprocess.TimeoutExpired(cmd, timeout or 0, output=out, stderr=err)
                wait_slice = min(wait_slice, remaining)

            try:
                out, err = proc.communicate(timeout=wait_slice)
                return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
            except subprocess.TimeoutExpired:
                continue
    finally:
        with _procs_lock:
            _active_procs.discard(proc)
