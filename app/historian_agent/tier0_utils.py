# app/historian_agent/tier0_utils.py
# Created: 2026-02-05
# Purpose: Tier 0 utilities (JSON parsing, logging, file saving)

"""
Tier 0 Utilities

Consolidates JSON cleaning, file saving, and logging patterns.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import threading

from config import APP_CONFIG


# ============================================================================
# JSON Parsing
# ============================================================================


def clean_json_response(response_text: str) -> str:
    """Clean LLM JSON response by removing markdown and control chars."""
    text = response_text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)

    if text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    elif text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            text = text[start:end + 1]

    return text


def parse_llm_json(response_text: str, default: Any = None) -> Any:
    """Parse JSON from LLM response with cleaning."""
    try:
        cleaned = clean_json_response(response_text)
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        if default is not None:
            return default
        raise ValueError(f"Failed to parse JSON: {exc}")


# ============================================================================
# File Saving
# ============================================================================


def save_with_timestamp(
    content: Any,
    base_dir: Path,
    filename_prefix: str,
    file_type: str = "json",
    subdirectory: Optional[str] = None,
) -> Path:
    """Save content to file with a timestamped filename."""
    if subdirectory:
        safe_subdir = subdirectory.replace("/", "_").replace(" ", "_")
        save_dir = base_dir / safe_subdir
    else:
        save_dir = base_dir

    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename_prefix}.{file_type}"
    filepath = save_dir / filename

    if isinstance(content, (dict, list)):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, default=str)
    else:
        filepath.write_text(str(content), encoding="utf-8")

    return filepath


# ============================================================================
# Logging
# ============================================================================


class Tier0Logger:
    """Tier 0 logger with file output when debug mode is enabled."""

    def __init__(self, log_dir: Path, log_prefix: str = "tier0") -> None:
        self.log_dir = Path(log_dir)
        self.log_prefix = log_prefix
        self.log_file = None
        self.debug_mode = APP_CONFIG.tier0.tier0_debug_mode or APP_CONFIG.debug_mode

        if self.debug_mode:
            self._init_log_file()

    def _init_log_file(self) -> None:
        if self.log_file is not None:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"{self.log_prefix}_{timestamp}.log"

        try:
            self.log_file = open(log_path, "w", encoding="utf-8")
            self.log_file.write(f"=== {self.log_prefix.upper()} Log ===\n")
            self.log_file.write(f"Started: {datetime.now().isoformat()}\n")
            self.log_file.write("=" * 60 + "\n\n")
            self.log_file.flush()

            sys.stderr.write(f"[TIER0] Logging to: {log_path}\n")
            sys.stderr.flush()
        except Exception as exc:
            sys.stderr.write(f"[TIER0] Failed to create log file: {exc}\n")
            self.log_file = None

    def log(self, step: str, detail: str = "", level: str = "INFO") -> None:
        if not self.debug_mode:
            return

        timestamp = time.strftime("%H:%M:%S")
        message = f"[{timestamp}] [{level}] {step}: {detail}\n"

        sys.stderr.write(message)
        sys.stderr.flush()

        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()


# ============================================================================
# Heartbeat
# ============================================================================


class Heartbeat:
    """Emit periodic heartbeat logs while a long-running task is active."""

    def __init__(
        self,
        logger: Tier0Logger,
        step: str,
        detail: str,
        interval_s: int,
    ) -> None:
        self.logger = logger
        self.step = step
        self.detail = detail
        self.interval_s = max(0, int(interval_s))
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self) -> "Heartbeat":
        if self.interval_s <= 0:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.interval_s <= 0:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        elapsed = 0
        while not self._stop.wait(self.interval_s):
            elapsed += self.interval_s
            self.logger.log("heartbeat", f"{self.step} {self.detail} elapsed={elapsed}s")


# ============================================================================
# Checkpoints
# ============================================================================


class CheckpointManager:
    """Save and load lightweight JSON checkpoints for long-running synthesis."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, stage: str, payload: Any, checksum: Optional[str] = None) -> Path:
        record = {
            "stage": stage,
            "checksum": checksum,
            "timestamp": datetime.now().isoformat(),
            "payload": payload,
        }
        latest_path = self.checkpoint_dir / f"{stage}_latest.json"
        latest_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.checkpoint_dir / f"{stage}_{timestamp}.json"
        archive_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return archive_path

    def load_latest(self, stage: str) -> Optional[Dict[str, Any]]:
        latest_path = self.checkpoint_dir / f"{stage}_latest.json"
        if not latest_path.exists():
            return None
        try:
            return json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
