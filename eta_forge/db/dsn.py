"""libpq connection-string assembly.

Kept separate from the pool so it can be unit-tested without psycopg installed, and
so that anything printing a DSN (logs, ``--dry-run`` output, error messages) goes
through :func:`redacted_dsn` and cannot leak a password.
"""

from __future__ import annotations

from typing import Dict, Optional
from urllib.parse import quote

from ..config import DatabaseConfig

_REDACTED = "***"


def _params(cfg: DatabaseConfig, *, include_password: bool) -> Dict[str, str]:
    params: Dict[str, str] = {
        "host": cfg.host,
        "port": str(cfg.port),
        "dbname": cfg.database,
        "user": cfg.user,
        "sslmode": cfg.sslmode,
        "connect_timeout": str(int(max(1, round(cfg.connect_timeout_s)))),
        "application_name": cfg.application_name,
    }
    if cfg.password:
        params["password"] = cfg.password if include_password else _REDACTED
    # A statement timeout set on the connection is the only reliable guard against a
    # pathological routing query pinning a worker; per-request timeouts in the app
    # layer cannot cancel work already running inside the server.
    params["options"] = f"-c statement_timeout={int(cfg.statement_timeout_ms)}"
    return params


def _render(params: Dict[str, str]) -> str:
    parts = []
    for key, value in params.items():
        text = str(value)
        if text == "" or any(ch in text for ch in " '\\"):
            escaped = text.replace("\\", "\\\\").replace("'", "\\'")
            parts.append(f"{key}='{escaped}'")
        else:
            parts.append(f"{key}={text}")
    return " ".join(parts)


def build_dsn(cfg: DatabaseConfig, *, password_override: Optional[str] = None) -> str:
    """Return a keyword/value libpq connection string.

    ``password_override`` lets a caller inject a secret pulled from somewhere other
    than the config (a secrets manager, ``PGPASSWORD``) without mutating the config.
    """
    params = _params(cfg, include_password=True)
    if password_override is not None:
        if password_override:
            params["password"] = password_override
        else:
            params.pop("password", None)
    return _render(params)


def redacted_dsn(cfg: DatabaseConfig) -> str:
    """The same string with the password replaced — safe to log."""
    return _render(_params(cfg, include_password=False))


def build_uri(cfg: DatabaseConfig) -> str:
    """A ``postgresql://`` URI, for tools that will not take keyword/value form."""
    auth = quote(cfg.user, safe="")
    if cfg.password:
        auth += ":" + quote(cfg.password, safe="")
    return (
        f"postgresql://{auth}@{cfg.host}:{cfg.port}/{quote(cfg.database, safe='')}"
        f"?sslmode={cfg.sslmode}"
    )


__all__ = ["build_dsn", "redacted_dsn", "build_uri"]
