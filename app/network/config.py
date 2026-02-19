# app/network/config.py
"""
Configuration for the network analysis module.

All values are read from environment variables with sensible defaults.
Domain-agnostic: entity types, thresholds, and display settings are all
configurable rather than hard-coded.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Immutable configuration snapshot for network analysis."""

    # --- Feature flag ---
    enabled: bool = True

    # --- Edge construction ---
    # Empty means "derive all available types from linked_entities at runtime".
    entity_types: List[str] = field(default_factory=list)
    max_mention_count: int = 5000
    min_edge_weight: int = 2
    build_batch_size: int = 500

    # --- API defaults ---
    default_limit: int = 500
    default_min_weight: int = 3

    # --- Metrics cache ---
    metrics_cache_ttl: int = 86400  # seconds (24 hours)

    # --- Visualization ---
    max_display_nodes: int = 500

    @classmethod
    def from_env(cls, overrides: Optional[dict] = None) -> "NetworkConfig":
        """
        Build a NetworkConfig from environment variables.

        Environment variable names follow the pattern NETWORK_<UPPER_FIELD>.
        Comma-separated values are split into lists.

        Parameters
        ----------
        overrides : dict, optional
            Key-value pairs that take precedence over environment variables.
            Useful for CLI argument pass-through.
        """
        overrides = overrides or {}

        def _env(key: str, default):
            """Read from overrides first, then env, then default."""
            if key.lower().replace("network_", "") in overrides:
                return overrides[key.lower().replace("network_", "")]
            return os.environ.get(key, default)

        def _bool(val) -> bool:
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in ("1", "true", "yes", "y")

        def _int(val, fallback: int) -> int:
            try:
                return int(val)
            except (TypeError, ValueError):
                return fallback

        def _list(val) -> List[str]:
            if isinstance(val, list):
                return val
            if val is None:
                return []
            if isinstance(val, str) and not val.strip():
                return []
            return [v.strip() for v in str(val).split(",") if v.strip()]

        config = cls(
            enabled=_bool(_env("NETWORK_ANALYSIS_ENABLED", "true")),
            entity_types=_list(_env("NETWORK_ENTITY_TYPES", "")),
            max_mention_count=_int(_env("NETWORK_MAX_MENTION_COUNT", 5000), 5000),
            min_edge_weight=_int(_env("NETWORK_MIN_EDGE_WEIGHT", 2), 2),
            build_batch_size=_int(_env("NETWORK_BUILD_BATCH_SIZE", 500), 500),
            default_limit=_int(_env("NETWORK_DEFAULT_LIMIT", 500), 500),
            default_min_weight=_int(_env("NETWORK_DEFAULT_MIN_WEIGHT", 3), 3),
            metrics_cache_ttl=_int(_env("NETWORK_METRICS_CACHE_TTL", 86400), 86400),
            max_display_nodes=_int(_env("NETWORK_MAX_DISPLAY_NODES", 500), 500),
        )

        logger.debug(
            "NetworkConfig loaded: types=%s max_mention=%d min_weight=%d",
            config.entity_types,
            config.max_mention_count,
            config.min_edge_weight,
        )
        return config
