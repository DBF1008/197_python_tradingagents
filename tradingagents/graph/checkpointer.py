"""LangGraph checkpoint support for resumable analysis runs.

Per-ticker SQLite databases so concurrent tickers don't contend.

Path resolution delegates to :class:`~tradingagents.storage.ArtifactStore`
when a *store* argument is provided; otherwise the legacy ``data_dir``
parameter is used to build a minimal store internally, preserving full
backward compatibility for callers that have not yet migrated.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Generator

from langgraph.checkpoint.sqlite import SqliteSaver

from tradingagents.dataflows.utils import safe_ticker_component

if TYPE_CHECKING:
    from tradingagents.storage import ArtifactStore


def _resolve_store(
    data_dir: str | Path | None,
    store: ArtifactStore | None,
) -> ArtifactStore:
    """Return an ArtifactStore, preferring the explicit *store* argument."""
    if store is not None:
        return store
    from tradingagents.storage import ArtifactStore

    return ArtifactStore(data_cache_dir=data_dir)


def _db_path(data_dir: str | Path, ticker: str, *, store: ArtifactStore | None = None) -> Path:
    """Return the SQLite checkpoint DB path for a ticker."""
    s = _resolve_store(data_dir, store)
    return s.checkpoint_db(ticker)


def thread_id(ticker: str, date: str) -> str:
    """Deterministic thread ID for a ticker+date pair."""
    return hashlib.sha256(f"{ticker.upper()}:{date}".encode()).hexdigest()[:16]


@contextmanager
def get_checkpointer(
    data_dir: str | Path,
    ticker: str,
    *,
    store: ArtifactStore | None = None,
) -> Generator[SqliteSaver, None, None]:
    """Context manager yielding a SqliteSaver backed by a per-ticker DB."""
    db = _db_path(data_dir, ticker, store=store)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()


def has_checkpoint(
    data_dir: str | Path,
    ticker: str,
    date: str,
    *,
    store: ArtifactStore | None = None,
) -> bool:
    """Check whether a resumable checkpoint exists for ticker+date."""
    return checkpoint_step(data_dir, ticker, date, store=store) is not None


def checkpoint_step(
    data_dir: str | Path,
    ticker: str,
    date: str,
    *,
    store: ArtifactStore | None = None,
) -> int | None:
    """Return the step number of the latest checkpoint, or None if none exists."""
    db = _db_path(data_dir, ticker, store=store)
    if not db.exists():
        return None
    tid = thread_id(ticker, date)
    with get_checkpointer(data_dir, ticker, store=store) as saver:
        config = {"configurable": {"thread_id": tid}}
        cp = saver.get_tuple(config)
        if cp is None:
            return None
        return cp.metadata.get("step")


def clear_all_checkpoints(
    data_dir: str | Path | None = None,
    *,
    store: ArtifactStore | None = None,
) -> int:
    """Remove all checkpoint DBs. Returns number of files deleted."""
    s = _resolve_store(data_dir, store)
    return s.clear_all_checkpoints()


def clear_checkpoint(
    data_dir: str | Path,
    ticker: str,
    date: str,
    *,
    store: ArtifactStore | None = None,
) -> None:
    """Remove checkpoint for a specific ticker+date by deleting the thread's rows."""
    db = _db_path(data_dir, ticker, store=store)
    if not db.exists():
        return
    tid = thread_id(ticker, date)
    conn = sqlite3.connect(str(db))
    try:
        for table in ("writes", "checkpoints"):
            conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()
