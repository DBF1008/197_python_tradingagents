"""LangGraph checkpoint support for resumable analysis runs.

Per-ticker SQLite databases so concurrent tickers don't contend.

On top of LangGraph's own checkpoint tables, each per-ticker DB also carries a
``run_manifest`` table: one row per thread recording *which run* produced the
checkpoint (selected analysts, asset type, a fingerprint of the key LLM config).
This is the auditable layer that lets us preview a resume and refuse to resume
when the configuration has drifted — see :mod:`tradingagents.graph.resume_policy`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Tuple

from langgraph.checkpoint.sqlite import SqliteSaver

from tradingagents.dataflows.utils import safe_ticker_component

from .resume_policy import (
    RunManifest,
    ResumeDecision,
    evaluate_resume,
    normalize_analysts,
)


def _db_path(data_dir: str | Path, ticker: str) -> Path:
    """Return the SQLite checkpoint DB path for a ticker."""
    # Reject ticker values that would escape the checkpoints directory.
    safe = safe_ticker_component(ticker).upper()
    p = Path(data_dir) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{safe}.db"


def thread_id(ticker: str, date: str, analysts=None) -> str:
    """Deterministic thread ID for a run.

    With ``analysts=None`` this reproduces the legacy ticker+date hash byte for
    byte (preserving existing checkpoints and callers). When an analyst list is
    given, the sorted/de-duped analyst set is folded into the hash so different
    analyst combinations on the same ticker+date get *physically separate*
    threads — they can coexist without overwriting each other.
    """
    base = f"{ticker.upper()}:{date}"
    if analysts is None:
        return hashlib.sha256(base.encode()).hexdigest()[:16]
    key = ",".join(normalize_analysts(analysts))
    return hashlib.sha256(f"{base}|{key}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# run_manifest table
# ---------------------------------------------------------------------------

_MANIFEST_DDL = """
CREATE TABLE IF NOT EXISTS run_manifest (
    thread_id        TEXT PRIMARY KEY,
    ticker           TEXT NOT NULL,
    trade_date       TEXT NOT NULL,
    asset_type       TEXT NOT NULL,
    analysts         TEXT NOT NULL,
    llm_config       TEXT NOT NULL,
    last_step        INTEGER,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    manifest_version INTEGER NOT NULL
)
"""


def _ensure_manifest_table(conn: sqlite3.Connection) -> None:
    conn.execute(_MANIFEST_DDL)
    conn.commit()


@contextmanager
def _manifest_conn(data_dir: str | Path, ticker: str) -> Generator[sqlite3.Connection, None, None]:
    """Short-lived connection for manifest ops, with the table ensured."""
    db = _db_path(data_dir, ticker)
    conn = sqlite3.connect(str(db))
    # Tolerate brief contention with an active saver on the same file.
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        _ensure_manifest_table(conn)
        yield conn
    finally:
        conn.close()


@contextmanager
def get_checkpointer(data_dir: str | Path, ticker: str) -> Generator[SqliteSaver, None, None]:
    """Context manager yielding a SqliteSaver backed by a per-ticker DB."""
    db = _db_path(data_dir, ticker)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        _ensure_manifest_table(conn)
        yield saver
    finally:
        conn.close()


def write_manifest(data_dir: str | Path, ticker: str, manifest: RunManifest) -> None:
    """Upsert a run manifest, preserving the original ``created_at``."""
    with _manifest_conn(data_dir, ticker) as conn:
        conn.execute(
            """
            INSERT INTO run_manifest (
                thread_id, ticker, trade_date, asset_type, analysts, llm_config,
                last_step, created_at, updated_at, manifest_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                ticker=excluded.ticker,
                trade_date=excluded.trade_date,
                asset_type=excluded.asset_type,
                analysts=excluded.analysts,
                llm_config=excluded.llm_config,
                last_step=excluded.last_step,
                updated_at=excluded.updated_at,
                manifest_version=excluded.manifest_version
            """,
            (
                manifest.thread_id,
                manifest.ticker,
                manifest.trade_date,
                manifest.asset_type,
                json.dumps(list(manifest.analysts)),
                json.dumps(manifest.llm_config),
                manifest.last_step,
                manifest.created_at,
                manifest.updated_at,
                manifest.manifest_version,
            ),
        )
        conn.commit()


def read_manifest(
    data_dir: str | Path, ticker: str, tid: str
) -> Optional[RunManifest]:
    """Read the manifest for a thread, or ``None`` if absent."""
    db = _db_path(data_dir, ticker)
    if not db.exists():
        return None
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            """
            SELECT thread_id, ticker, trade_date, asset_type, analysts,
                   llm_config, last_step, created_at, updated_at, manifest_version
            FROM run_manifest WHERE thread_id = ?
            """,
            (tid,),
        ).fetchone()
    except sqlite3.OperationalError:
        # Table doesn't exist yet (no manifest ever written for this DB).
        return None
    finally:
        conn.close()

    if row is None:
        return None
    payload = {
        "thread_id": row[0],
        "ticker": row[1],
        "trade_date": row[2],
        "asset_type": row[3],
        "analysts": json.loads(row[4]) if row[4] else [],
        "llm_config": json.loads(row[5]) if row[5] else {},
        "last_step": row[6],
        "created_at": row[7],
        "updated_at": row[8],
        "manifest_version": row[9],
    }
    return RunManifest.from_payload(payload)


def has_checkpoint(
    data_dir: str | Path, ticker: str, date: str, analysts=None
) -> bool:
    """Check whether a resumable checkpoint exists for ticker+date(+analysts)."""
    return checkpoint_step(data_dir, ticker, date, analysts) is not None


def checkpoint_step(
    data_dir: str | Path, ticker: str, date: str, analysts=None
) -> Optional[int]:
    """Return the step number of the latest checkpoint, or None if none exists."""
    db = _db_path(data_dir, ticker)
    if not db.exists():
        return None
    tid = thread_id(ticker, date, analysts)
    with get_checkpointer(data_dir, ticker) as saver:
        config = {"configurable": {"thread_id": tid}}
        cp = saver.get_tuple(config)
        if cp is None:
            return None
        return cp.metadata.get("step")


def clear_thread(data_dir: str | Path, ticker: str, tid: str) -> None:
    """Targeted clear of a single thread's state (checkpoint rows + manifest).

    Version-robust: instead of hard-coding LangGraph's table names (which differ
    across versions), discover every table that has a ``thread_id`` column and
    delete this thread's rows from it. This naturally covers the manifest table
    too and skips bookkeeping tables like ``checkpoint_migrations``.
    """
    db = _db_path(data_dir, ticker)
    if not db.exists():
        return
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        ]
        for table in tables:
            cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
            if "thread_id" in cols:
                conn.execute(f'DELETE FROM "{table}" WHERE thread_id = ?', (tid,))
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()


def clear_all_checkpoints(data_dir: str | Path) -> int:
    """Remove all checkpoint DBs. Returns number of files deleted."""
    cp_dir = Path(data_dir) / "checkpoints"
    if not cp_dir.exists():
        return 0
    dbs = list(cp_dir.glob("*.db"))
    for db in dbs:
        db.unlink()
    return len(dbs)


def clear_checkpoint(
    data_dir: str | Path, ticker: str, date: str, analysts=None
) -> None:
    """Remove checkpoint + manifest for a specific ticker+date(+analysts)."""
    clear_thread(data_dir, ticker, thread_id(ticker, date, analysts))


def plan_resume(
    data_dir: str | Path,
    ticker: str,
    trade_date: str,
    asset_type: str,
    analysts,
    config: dict,
    *,
    strict: bool = False,
) -> Tuple[ResumeDecision, str]:
    """Compute the resume decision for an intended run.

    This is the single bridge between the on-disk state and the pure resume
    policy: it computes the analyst-aware thread id, reads the live LangGraph
    step and the stored manifest, builds the intended manifest, and evaluates
    compatibility. Returns ``(decision, thread_id)``.
    """
    tid = thread_id(ticker, str(trade_date), analysts)
    live_step = checkpoint_step(data_dir, ticker, str(trade_date), analysts)
    stored = read_manifest(data_dir, ticker, tid)
    intended = RunManifest.build(
        thread_id=tid,
        ticker=ticker,
        trade_date=str(trade_date),
        asset_type=asset_type,
        analysts=analysts,
        config=config,
        last_step=live_step,
    )
    decision = evaluate_resume(
        stored,
        intended,
        has_live_checkpoint=live_step is not None,
        strict_unverified=strict,
    )
    decision.last_step = live_step
    return decision, tid
