"""Checkpoint manifest: bind run metadata to checkpoints for auditable resume.

Each per-ticker checkpoint DB gets a companion ``{TICKER}.manifest.json``
file.  The file is a JSON object keyed by ``thread_id`` so multiple dates
for the same ticker coexist in one file.  Manifests are written atomically
(write-to-tmp + ``os.replace``) to survive mid-write crashes.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tradingagents.dataflows.utils import safe_ticker_component

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class RunManifest:
    """Immutable snapshot of configuration for a single checkpointed run."""

    version: int = MANIFEST_VERSION
    thread_id: str = ""
    ticker: str = ""
    trade_date: str = ""
    asset_type: str = "stock"
    selected_analysts: tuple = ()  # ordered, e.g. ("market", "social", "news")
    llm_provider: str = ""
    deep_think_llm: str = ""
    quick_think_llm: str = ""
    temperature: Optional[float] = None
    backend_url: Optional[str] = None
    last_completed_step: int = -1
    created_at: str = ""  # ISO-8601 UTC
    updated_at: str = ""  # ISO-8601 UTC

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict suitable for ``json.dumps``."""
        d = asdict(self)
        # asdict turns tuples into lists; convert back for consistency.
        d["selected_analysts"] = list(self.selected_analysts)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunManifest":
        """Deserialize from a dict (inverse of :meth:`to_dict`).

        Unknown keys are silently ignored so a manifest written by a
        future version with extra fields can still be loaded.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        # Restore tuple for frozen dataclass.
        if "selected_analysts" in filtered and isinstance(
            filtered["selected_analysts"], list
        ):
            filtered["selected_analysts"] = tuple(filtered["selected_analysts"])
        return cls(**filtered)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        ticker: str,
        trade_date: str,
        asset_type: str,
        selected_analysts: List[str],
        tid: str,
        step: int = -1,
    ) -> "RunManifest":
        """Build a manifest from the current runtime config.

        ``step`` is the last completed checkpoint step (``-1`` if starting
        fresh).  Timestamps are set to *now* in UTC.
        """
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            thread_id=tid,
            ticker=ticker,
            trade_date=str(trade_date),
            asset_type=asset_type,
            selected_analysts=tuple(selected_analysts),
            llm_provider=config.get("llm_provider", ""),
            deep_think_llm=config.get("deep_think_llm", ""),
            quick_think_llm=config.get("quick_think_llm", ""),
            temperature=config.get("temperature"),
            backend_url=config.get("backend_url"),
            last_completed_step=step,
            created_at=now,
            updated_at=now,
        )

    def with_step(self, step: int) -> "RunManifest":
        """Return a copy with ``last_completed_step`` and ``updated_at`` bumped."""
        now = datetime.now(timezone.utc).isoformat()
        return replace(self, last_completed_step=step, updated_at=now)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _manifest_path(data_dir: str | Path, ticker: str) -> Path:
    """Return the manifest JSON path for a ticker."""
    safe = safe_ticker_component(ticker).upper()
    p = Path(data_dir) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{safe}.manifest.json"


def _load_manifest_file(path: Path) -> Dict[str, Any]:
    """Load the entire manifest file as a dict of ``{thread_id: manifest_dict}``.

    Returns an empty dict when the file is missing, empty, or corrupt.
    """
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        return json.loads(text)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read manifest %s: %s", path, exc)
        return {}


def _save_manifest_file(path: Path, data: Dict[str, Any]) -> None:
    """Atomically write the manifest file (write-tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=path.stem + "_", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(path))
    except BaseException:
        # Clean up the temp file on any failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_manifest(
    data_dir: str | Path, ticker: str, tid: str
) -> Optional[RunManifest]:
    """Load the manifest for a specific thread, or ``None`` if absent/corrupt."""
    path = _manifest_path(data_dir, ticker)
    all_data = _load_manifest_file(path)
    entry = all_data.get(tid)
    if entry is None:
        return None
    try:
        return RunManifest.from_dict(entry)
    except (TypeError, KeyError) as exc:
        logger.warning("Manifest entry for thread %s is malformed: %s", tid, exc)
        return None


def save_manifest(data_dir: str | Path, manifest: RunManifest) -> None:
    """Persist a manifest entry (create or update). Atomic write."""
    path = _manifest_path(data_dir, manifest.ticker)
    all_data = _load_manifest_file(path)
    all_data[manifest.thread_id] = manifest.to_dict()
    _save_manifest_file(path, all_data)


def delete_manifest_entry(
    data_dir: str | Path, ticker: str, tid: str
) -> bool:
    """Remove the manifest entry for a specific thread.

    If the file becomes empty after removal, the file itself is deleted.
    Returns ``True`` if an entry was actually removed.
    """
    path = _manifest_path(data_dir, ticker)
    all_data = _load_manifest_file(path)
    if tid not in all_data:
        return False
    del all_data[tid]
    if not all_data:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    else:
        _save_manifest_file(path, all_data)
    return True


def clear_all_manifests(data_dir: str | Path) -> int:
    """Delete every ``*.manifest.json`` in the checkpoints directory.

    Returns the number of files deleted.
    """
    cp_dir = Path(data_dir) / "checkpoints"
    if not cp_dir.exists():
        return 0
    count = 0
    for f in cp_dir.glob("*.manifest.json"):
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    return count
