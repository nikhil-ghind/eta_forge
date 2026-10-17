"""A content-addressed, filesystem-only model registry.

Layout::

    <root>/
      residual/
        20260824T1200Z-a1b2c3d4/       one immutable version
          manifest.json
          residual.json, residual_q50.lgb, residual_q90.lgb, feature_state.json
        20260831T0900Z-9f8e7d6c/
        aliases.json                   {"production": {"version": ..., "history": [...]}}
      prep/
      dwell/

A version id is ``<name>/<UTC timestamp>-<first 8 hex of the content hash>``. The hash
covers every artifact file, so two versions with the same id are byte-identical and a
retrain that produced the same model does not mint a new version.

Every write is atomic. A version is assembled in a ``.staging-*`` directory and
renamed into place in one ``rename(2)``; the alias file is written to a temp file and
``os.replace``d over the old one. A serving process reading the registry concurrently
with a training job therefore sees either the old state or the new one, never a
half-written model — which is the whole job, and the reason this is not just
``shutil.copytree``.

No MLflow, no database: a registry that can be inspected with ``ls`` and ``cat`` and
rolled back with one command is worth more here than a UI.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..types import EtaForgeError, ModelNotFound

MANIFEST = "manifest.json"
ALIASES = "aliases.json"

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_VERSION_RE = re.compile(
    r"^(?P<name>[a-z][a-z0-9_-]{0,63})/(?P<stamp>\d{8}T\d{4}Z)-(?P<digest>[0-9a-f]{8})$"
)
_ALIAS_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class RegistryError(EtaForgeError):
    """A malformed reference, or an operation the registry state does not allow."""


@dataclass(frozen=True)
class ModelVersion:
    name: str
    version: str
    path: str
    manifest: Mapping[str, Any]

    @property
    def stamp(self) -> str:
        return self.version.split("/", 1)[1].split("-", 1)[0]

    @property
    def content_hash(self) -> str:
        return str(self.manifest.get("content_hash", ""))


def content_digest(directory: str) -> str:
    """SHA-256 over every file (path and bytes) except the manifest itself."""
    digest = hashlib.sha256()
    for base, dirs, files in os.walk(directory):
        dirs.sort()
        for filename in sorted(files):
            full = os.path.join(base, filename)
            rel = os.path.relpath(full, directory).replace(os.sep, "/")
            if rel == MANIFEST:
                continue
            digest.update(rel.encode("utf-8") + b"\0")
            with open(full, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            digest.update(b"\0")
    return digest.hexdigest()


def parse_version(version: str) -> Dict[str, str]:
    match = _VERSION_RE.match(version or "")
    if not match:
        raise RegistryError(
            f"malformed model version {version!r}; expected name/YYYYMMDDTHHMMZ-xxxxxxxx"
        )
    return match.groupdict()


def _atomic_write_json(path: str, payload: Any) -> None:
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".", suffix=".aliases.tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class ModelRegistry:
    def __init__(self, root: str):
        self.root = root

    # -- paths -------------------------------------------------------------------

    def _name_dir(self, name: str) -> str:
        if not _NAME_RE.match(name or ""):
            raise RegistryError(f"invalid model name {name!r}")
        return os.path.join(self.root, name)

    def _version_dir(self, version: str) -> str:
        parts = parse_version(version)
        return os.path.join(self._name_dir(parts["name"]), f"{parts['stamp']}-{parts['digest']}")

    # -- write -------------------------------------------------------------------

    def register(
        self,
        name: str,
        writer: Callable[[str], Optional[Mapping[str, Any]]],
        manifest: Mapping[str, Any],
        *,
        created_at: Optional[datetime] = None,
    ) -> ModelVersion:
        """Write a new version via ``writer(staging_dir)`` and publish it atomically.

        ``writer`` saves the artifacts into the directory it is given and may return a
        small dict of artifact metadata, stored under ``artifact`` in the manifest.
        """
        name_dir = self._name_dir(name)
        os.makedirs(name_dir, exist_ok=True)
        when = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        staging = tempfile.mkdtemp(prefix=".staging-", dir=name_dir)
        # mkdtemp is 0700; a serving container running as another uid must still be
        # able to read the published version.
        os.chmod(staging, 0o755)
        try:
            artifact = writer(staging) or {}
            digest = content_digest(staging)
            version_dir_name = f"{when:%Y%m%dT%H%MZ}-{digest[:8]}"
            version = f"{name}/{version_dir_name}"
            final = os.path.join(name_dir, version_dir_name)
            if os.path.isdir(final):
                shutil.rmtree(staging)
                return self.get(version)

            files = sorted(
                os.path.relpath(os.path.join(b, f), staging).replace(os.sep, "/")
                for b, _, fs in os.walk(staging)
                for f in fs
            )
            full_manifest = dict(manifest)
            full_manifest.update(
                {
                    "name": name,
                    "version": version,
                    "content_hash": digest,
                    "created_at": when.isoformat(),
                    "files": files,
                    "artifact": dict(artifact),
                }
            )
            with open(os.path.join(staging, MANIFEST), "w", encoding="utf-8") as handle:
                json.dump(full_manifest, handle, indent=2, sort_keys=True, default=str)
            os.rename(staging, final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return ModelVersion(name, version, final, full_manifest)

    # -- read --------------------------------------------------------------------

    def names(self) -> List[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted(
            entry for entry in os.listdir(self.root)
            if _NAME_RE.match(entry) and os.path.isdir(os.path.join(self.root, entry))
        )

    def list_versions(self, name: str) -> List[ModelVersion]:
        name_dir = self._name_dir(name)
        if not os.path.isdir(name_dir):
            return []
        out: List[ModelVersion] = []
        for entry in sorted(os.listdir(name_dir)):
            if entry.startswith(".") or not os.path.isfile(os.path.join(name_dir, entry, MANIFEST)):
                continue
            version = f"{name}/{entry}"
            if _VERSION_RE.match(version):
                out.append(self.get(version))
        return out

    def get(self, version: str) -> ModelVersion:
        path = self._version_dir(version)
        manifest_path = os.path.join(path, MANIFEST)
        if not os.path.isfile(manifest_path):
            raise ModelNotFound(f"model version {version} is not in the registry at {self.root}")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return ModelVersion(parse_version(version)["name"], version, path, manifest)

    def verify(self, version: str) -> bool:
        """Recompute the content hash; False means the artifacts were tampered with."""
        mv = self.get(version)
        return content_digest(mv.path) == mv.content_hash

    def resolve(self, ref: str, *, default_alias: str = "production") -> ModelVersion:
        """``name/stamp-hash`` (a version), ``name@alias``, or bare ``name`` (default alias)."""
        if "/" in ref:
            return self.get(ref)
        name, _, alias = ref.partition("@")
        alias = alias or default_alias
        current = self.aliases(name).get(alias, {}).get("version")
        if not current:
            raise ModelNotFound(f"model {name!r} has no {alias!r} alias")
        return self.get(current)

    def latest(self, name: str) -> ModelVersion:
        versions = self.list_versions(name)
        if not versions:
            raise ModelNotFound(f"model {name!r} has no versions")
        return max(versions, key=lambda v: (v.stamp, v.version))

    # -- aliases -----------------------------------------------------------------

    def aliases(self, name: str) -> Dict[str, Dict[str, Any]]:
        path = os.path.join(self._name_dir(name), ALIASES)
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def promote(
        self, version: str, *, alias: str = "production", reason: Optional[str] = None
    ) -> Optional[str]:
        """Point ``alias`` at ``version``; return the version it pointed at before."""
        if not _ALIAS_RE.match(alias):
            raise RegistryError(f"invalid alias {alias!r}")
        target = self.get(version)  # raises if missing
        aliases = self.aliases(target.name)
        entry = aliases.get(alias, {"version": None, "history": []})
        previous = entry.get("version")
        if previous == version:
            return previous
        history = list(entry.get("history", []))
        if previous:
            history.append(previous)
        aliases[alias] = {
            "version": version,
            "history": history[-50:],
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }
        _atomic_write_json(os.path.join(self._name_dir(target.name), ALIASES), aliases)
        return previous

    def rollback(self, name: str, *, alias: str = "production") -> str:
        """Re-point ``alias`` at the version it held before the current one."""
        aliases = self.aliases(name)
        entry = aliases.get(alias)
        if not entry or not entry.get("history"):
            raise RegistryError(f"{name}@{alias} has no previous version to roll back to")
        history = list(entry["history"])
        previous = history.pop()
        self.get(previous)  # must still exist
        aliases[alias] = {
            "version": previous,
            "history": history,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "reason": f"rollback from {entry.get('version')}",
        }
        _atomic_write_json(os.path.join(self._name_dir(name), ALIASES), aliases)
        return previous


__all__ = [
    "ModelRegistry",
    "ModelVersion",
    "RegistryError",
    "content_digest",
    "parse_version",
    "MANIFEST",
    "ALIASES",
]
