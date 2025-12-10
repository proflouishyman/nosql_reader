"""Helpers for surfacing Docker volume mounts in the settings UI."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

# Added pattern to handle docker-compose style environment substitutions with defaults.
_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(:-([^}]*))?\}")


def _expand_compose_value(raw: str) -> str:
    """Expand ``${VAR:-default}`` expressions so targets resolve inside the container."""

    # Added manual expansion because os.path.expandvars does not understand default values.
    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(3) or ""
        return os.environ.get(name, default)

    expanded = _ENV_PATTERN.sub(_replace, raw)
    return os.path.expanduser(expanded)


def get_mounted_paths(compose_path: str = "docker-compose.yml") -> List[Tuple[str, str]]:
    """Return ``(source, target)`` tuples for the ``app`` service volumes."""

    # Added safeguard so the UI silently reports no mounts when compose metadata is absent.
    if not os.path.exists(compose_path):
        return []

    with open(compose_path, "r", encoding="utf-8") as handle:
        compose = yaml.safe_load(handle) or {}

    mounts: List[Tuple[str, str]] = []
    volumes: Iterable[object] = compose.get("services", {}).get("app", {}).get("volumes", [])
    for entry in volumes:
        if isinstance(entry, dict):
            source = _expand_compose_value(str(entry.get("source", "")))
            target = _expand_compose_value(str(entry.get("target", "")))
        elif isinstance(entry, str) and ":" in entry:
            source_raw, target_raw = entry.split(":", 1)
            source = _expand_compose_value(source_raw.strip())
            target = _expand_compose_value(target_raw.strip())
        else:
            # Added continue so unexpected entries (like tmpfs) do not break the response.
            continue

        if source or target:
            mounts.append((source, target))

    return mounts


def short_tree(root_path: str, depth: int = 2) -> Dict[str, Dict[str, List[str]]]:
    """Return a limited directory tree rooted at ``root_path``."""

    base = Path(root_path)
    # Added guard to keep the API predictable when mounts have not been created yet.
    if not base.exists():
        return {}

    tree: Dict[str, Dict[str, List[str]]] = {}
    for dirpath, dirnames, filenames in os.walk(base):
        relative = Path(dirpath).relative_to(base)
        level = len(relative.parts)
        if level > depth:
            # Added depth limiter so responses stay small for the settings UI.
            continue

        key = str(relative) if relative.parts else "."
        tree[key] = {
            "dirs": sorted(dirnames)[:5],  # Added slice to avoid overwhelming the UI with entries.
            "files": sorted(filenames)[:5],  # Added slice to avoid overwhelming the UI with entries.
        }

    return tree

