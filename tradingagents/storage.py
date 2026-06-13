"""Unified path policy for TradingAgents run artifacts.

Run products — LangGraph checkpoint SQLite DBs, per-ticker state-log JSON, the
CLI's per-run report directory + ``message_tool.log``, and the append-only
memory markdown — used to assemble their own paths, create their own
directories, and (inconsistently) validate the ticker in four different
modules. ``ArtifactStore`` is the single place that:

* locates each artifact under the configured roots,
* creates the directory that holds it (opt-in via ``create=``),
* applies the ticker path-safety check on *every* write path (previously the
  CLI run directory took the raw ticker, letting a crafted value escape
  ``results_dir``), and
* removes artifacts for targeted cleanup.

The directory layout and user-visible filenames are unchanged for valid
tickers — ``safe_ticker_component`` returns valid tickers verbatim, so routing
the previously-unguarded CLI path through here is byte-identical except that a
traversal attempt now raises instead of escaping the tree.

Roots (defaults from :mod:`tradingagents.default_config`):

    checkpoints   {data_cache_dir}/checkpoints/{TICKER}.db          (ticker upper-cased)
    state logs    {results_dir}/{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json
    CLI run dir   {results_dir}/{ticker}/{date}/  (+ reports/{section}.md, message_tool.log)
    memory log    {memory_log_path}  (+ sibling {stem}.tmp for atomic writes)

This module imports only :mod:`tradingagents.dataflows.utils` so that the
modules wiring through it (checkpointer, trading_graph, cli, memory) never
create an import cycle.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from tradingagents.dataflows.utils import safe_ticker_component


def _as_path(value) -> Optional[Path]:
    """Coerce a configured root to ``Path``; ``None``/empty stays ``None``."""
    if value is None or value == "":
        return None
    return Path(value)


class ArtifactStore:
    """Locate, create, validate, and clean TradingAgents run artifacts.

    Construct from a config dict via :meth:`from_config`, or directly with any
    subset of roots — methods that need a root that wasn't supplied raise a
    clear ``ValueError`` rather than producing a path under ``None``.
    """

    # Directory / filename conventions — the single source of truth for the
    # on-disk layout. Changing one of these moves every artifact of that kind.
    CHECKPOINTS_DIRNAME = "checkpoints"
    STATE_LOGS_DIRNAME = "TradingAgentsStrategy_logs"
    REPORTS_DIRNAME = "reports"
    MESSAGE_LOG_FILENAME = "message_tool.log"
    STATE_LOG_PREFIX = "full_states_log_"  # state-log file = PREFIX + date + ".json"

    def __init__(
        self,
        *,
        results_dir=None,
        cache_dir=None,
        memory_log_path=None,
    ) -> None:
        self._results_dir = _as_path(results_dir)
        self._cache_dir = _as_path(cache_dir)
        # Kept unexpanded; expanduser() is applied at access time so a "~/..."
        # config value resolves to the home dir rather than a literal "~".
        self._memory_log_path = _as_path(memory_log_path)

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "ArtifactStore":
        """Build a store from a TradingAgents config dict (``None`` -> empty)."""
        cfg = config or {}
        return cls(
            results_dir=cfg.get("results_dir"),
            cache_dir=cfg.get("data_cache_dir"),
            memory_log_path=cfg.get("memory_log_path"),
        )

    # --- safety chokepoint -------------------------------------------------

    @staticmethod
    def safe_component(ticker: str) -> str:
        """Validate ``ticker`` is safe to use as a path component.

        Single entry point for the run-artifact layer; raises ``ValueError``
        on anything that could traverse out of a configured directory.
        """
        return safe_ticker_component(ticker)

    # --- internal helpers --------------------------------------------------

    @staticmethod
    def _ensure(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _require(root: Optional[Path], label: str) -> Path:
        if root is None:
            raise ValueError(f"ArtifactStore: {label} is not configured")
        return root

    def ensure_base_dirs(self) -> None:
        """Create the configured ``results_dir`` and ``cache_dir`` (if set).

        Mirrors the two ``os.makedirs`` calls ``TradingAgentsGraph.__init__``
        used to make; the memory-log parent is created lazily on first use, not
        here, to keep parity with the previous behavior.
        """
        for root in (self._results_dir, self._cache_dir):
            if root is not None:
                self._ensure(root)

    # --- checkpoints (under cache_dir) -------------------------------------

    def checkpoints_dir(self, *, create: bool = False) -> Path:
        root = self._require(self._cache_dir, "cache_dir")
        d = root / self.CHECKPOINTS_DIRNAME
        return self._ensure(d) if create else d

    def checkpoint_db(self, ticker: str, *, create: bool = True) -> Path:
        """Per-ticker checkpoint DB path. Ticker is upper-cased (legacy convention).

        ``create`` defaults to ``True`` because the SQLite connection opened by
        the checkpointer fails if ``checkpoints/`` does not yet exist.
        """
        safe = self.safe_component(ticker).upper()
        return self.checkpoints_dir(create=create) / f"{safe}.db"

    def clear_checkpoints(self) -> int:
        """Delete every ``*.db`` under ``checkpoints/``; return the count.

        A missing directory yields ``0`` without creating an empty one.
        """
        d = self.checkpoints_dir(create=False)
        if not d.exists():
            return 0
        dbs = list(d.glob("*.db"))
        for db in dbs:
            db.unlink()
        return len(dbs)

    # --- state logs (under results_dir, case preserved) --------------------

    def state_logs_dir(self, ticker: str, *, create: bool = False) -> Path:
        root = self._require(self._results_dir, "results_dir")
        safe = self.safe_component(ticker)
        d = root / safe / self.STATE_LOGS_DIRNAME
        return self._ensure(d) if create else d

    def state_log_file(self, ticker: str, date, *, create: bool = True) -> Path:
        d = self.state_logs_dir(ticker, create=create)
        return d / f"{self.STATE_LOG_PREFIX}{date}.json"

    # --- CLI run artifacts (under results_dir, case preserved) -------------

    def run_dir(self, ticker: str, date, *, create: bool = False) -> Path:
        root = self._require(self._results_dir, "results_dir")
        safe = self.safe_component(ticker)
        d = root / safe / str(date)
        return self._ensure(d) if create else d

    def reports_dir(self, ticker: str, date, *, create: bool = False) -> Path:
        d = self.run_dir(ticker, date, create=create) / self.REPORTS_DIRNAME
        return self._ensure(d) if create else d

    def report_file(self, ticker: str, date, section: str, *, create: bool = False) -> Path:
        return self.reports_dir(ticker, date, create=create) / f"{section}.md"

    def message_log(self, ticker: str, date, *, create: bool = False) -> Path:
        return self.run_dir(ticker, date, create=create) / self.MESSAGE_LOG_FILENAME

    def clear_run(self, ticker: str, date) -> bool:
        """Remove the entire run directory for ``ticker``/``date``.

        Returns ``True`` when a directory was removed, ``False`` when none
        existed. The ticker is still validated, so a traversal value raises.
        """
        d = self.run_dir(ticker, date, create=False)
        if d.exists():
            shutil.rmtree(d)
            return True
        return False

    # --- memory log (single configured file path) --------------------------

    def memory_log_file(self, *, create: bool = False) -> Optional[Path]:
        """Resolved memory-log path, or ``None`` when no path is configured.

        ``create=True`` makes the parent directory (not the file). The path is
        ``expanduser``-ed so a ``~/...`` config value resolves to the home dir.
        """
        if self._memory_log_path is None:
            return None
        p = self._memory_log_path.expanduser()
        if create:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def memory_log_tmp_file(self) -> Optional[Path]:
        """Sibling temp path used for atomic rewrites (``<stem>.tmp``)."""
        p = self.memory_log_file()
        return p.with_suffix(".tmp") if p is not None else None
