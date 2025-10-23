"""Utilities for deriving MongoDB connection information from environment variables.

Centralising this logic keeps individual scripts small and avoids repeated edits to
tracked environment files that often lead to merge conflicts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote_plus


def _first_env(*names: str) -> Optional[str]:
    """Return the first non-empty environment variable from *names*."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


@dataclass
class MongoCredentials:
    """Resolved Mongo connection details."""

    uri: Optional[str]
    username: Optional[str]
    password: Optional[str]
    host: str
    port: str
    auth_db: str


def resolve_credentials() -> MongoCredentials:
    """Gather MongoDB credentials from any supported variable names."""

    uri = _first_env("MONGO_URI", "APP_MONGO_URI")
    username = _first_env("MONGO_ROOT_USERNAME", "MONGO_INITDB_ROOT_USERNAME")
    password = _first_env("MONGO_ROOT_PASSWORD", "MONGO_INITDB_ROOT_PASSWORD")
    host = os.environ.get("MONGO_HOST", "mongodb")
    port = os.environ.get("MONGO_PORT", "27017")
    auth_db = os.environ.get("MONGO_AUTH_DB", "admin")

    return MongoCredentials(uri=uri, username=username, password=password, host=host, port=port, auth_db=auth_db)


def build_mongo_uri() -> str:
    """Return a usable MongoDB URI from the environment.

    Preference order:
    1. Explicit URI (`MONGO_URI` or `APP_MONGO_URI`).
    2. Root credentials paired with host/port/auth DB.
    3. Anonymous connection to the configured host/port/auth DB.
    """

    creds = resolve_credentials()
    if creds.uri:
        return creds.uri.strip().strip('"')

    if creds.username and creds.password:
        username = quote_plus(creds.username)
        password = quote_plus(creds.password)
        return f"mongodb://{username}:{password}@{creds.host}:{creds.port}/{creds.auth_db}"

    return f"mongodb://{creds.host}:{creds.port}/{creds.auth_db}"


def build_admin_auth_uri(host: str = "localhost", username: Optional[str] = None, password: Optional[str] = None) -> str:
    """Construct an admin-authenticated URI targeting *host* for maintenance scripts.

    Optional *username* and *password* overrides let callers supply prompted
    credentials without recomputing the rest of the connection string.
    """

    creds = resolve_credentials()
    resolved_username = username or creds.username or "admin"
    resolved_password = password or creds.password or "secret"
    return f"mongodb://{quote_plus(resolved_username)}:{quote_plus(resolved_password)}@{host}:{creds.port}/{creds.auth_db}"
