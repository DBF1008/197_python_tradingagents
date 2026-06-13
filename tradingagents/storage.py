"""Unified artifact path policy for TradingAgents.

All runtime artifacts — checkpoint SQLite DBs, state JSON logs, CLI
message logs, report sections, memory markdown, and OHLCV CSV caches —
flow through :class:`ArtifactStore`.  The class is a *path policy engine*:
it produces validated :class:`~pathlib.Path` objects, ensures their parent
directories exist, and provides targeted cleanup.  It does **not** perform
application-level I/O (read/write/open); callers still handle file content.

Design goals:

* **Always safe** — every method accepting a ticker routes through
  :func:`~tradingagents.dataflows.utils.safe_ticker_component`.
* **Backward-compatible** — produced paths are identical to the previous
  per-module implementations for valid inputs.
* **Centralized cleanup** — targeted removal of checkpoints, results,
  and caches through a single API.
"""

from __future__ import annotations

import os
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from tradingagents.dataflows.utils import safe_ticker_component

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")


class ArtifactStore:
    """Centralized path resolution, directory creation, and cleanup policy.

    Parameters
    ----------
    results_dir:
        Root directory for analysis results (state logs, message logs,
        report sections).  Falls back to ``$TRADINGAGENTS_RESULTS_DIR``
        or ``~/.tradingagents/logs``.
    data_cache_dir:
        Root directory for cached data (checkpoint DBs, OHLCV CSVs).
        Falls back to ``$TRADINGAGENTS_CACHE_DIR``
        or ``~/.tradingagents/cache``.
    memory_log_path:
        Path to the memory markdown log file.
        Falls back to ``$TRADINGAGENTS_MEMORY_LOG_PATH``
        or ``~/.tradingagents/memory/trading_memory.md``.
    """

    def __init__(
        self,
        results_dir: str | Path | None = None,
        data_cache_dir: str | Path | None = None,
        memory_log_path: str | Path | None = None,
    ) -> None:
        self.results_dir = Path(
            results_dir
            or os.environ.get("TRADINGAGENTS_RESULTS_DIR")
            or os.path.join(_TRADINGAGENTS_HOME, "logs")
        ).resolve()

        self.data_cache_dir = Path(
            data_cache_dir
            or os.environ.get("TRADINGAGENTS_CACHE_DIR")
            or os.path.join(_TRADINGAGENTS_HOME, "cache")
        ).resolve()

        self.memory_log_path = Path(
            memory_log_path
            or os.environ.get("TRADINGAGENTS_MEMORY_LOG_PATH")
            or os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")
        ).resolve()

    # ------------------------------------------------------------------
    # Directory bootstrap
    # ------------------------------------------------------------------

    def ensure_base_dirs(self) -> None:
        """Create all base directories (idempotent).

        Replaces the scattered ``os.makedirs`` / ``Path.mkdir`` calls that
        were previously in ``TradingAgentsGraph.__init__``,
        ``TradingMemoryLog.__init__``, ``checkpointer._db_path``, etc.
        """
        for d in (self.results_dir, self.data_cache_dir, self.memory_log_path.parent):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _date_str(analysis_date: date | str) -> str:
        """Coerce *analysis_date* to a ``YYYY-MM-DD`` string."""
        if isinstance(analysis_date, str):
            return analysis_date
        return analysis_date.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Checkpoint artifacts
    # ------------------------------------------------------------------

    def checkpoint_db(self, ticker: str) -> Path:
        """Return the per-ticker SQLite checkpoint path.

        Path format: ``{data_cache_dir}/checkpoints/{TICKER_UPPER}.db``

        The checkpoints subdirectory is created if it does not exist.
        """
        safe = safe_ticker_component(ticker).upper()
        p = self.data_cache_dir / "checkpoints" / f"{safe}.db"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # State log artifacts
    # ------------------------------------------------------------------

    def state_log(self, ticker: str, analysis_date: date | str) -> Path:
        """Return the full-state JSON log path.

        Path format:
        ``{results_dir}/{safe_ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json``
        """
        safe = safe_ticker_component(ticker)
        date_s = self._date_str(analysis_date)
        p = (
            self.results_dir
            / safe
            / "TradingAgentsStrategy_logs"
            / f"full_states_log_{date_s}.json"
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # CLI message log artifacts
    # ------------------------------------------------------------------

    def message_log(self, ticker: str, analysis_date: date | str) -> Path:
        """Return the CLI ``message_tool.log`` path.

        Path format: ``{results_dir}/{safe_ticker}/{date}/message_tool.log``

        **Security**: the raw CLI ticker is now routed through
        ``safe_ticker_component`` — previously it was interpolated directly.
        """
        safe = safe_ticker_component(ticker)
        date_s = self._date_str(analysis_date)
        p = self.results_dir / safe / date_s / "message_tool.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # CLI report artifacts
    # ------------------------------------------------------------------

    def report_dir(self, ticker: str, analysis_date: date | str) -> Path:
        """Return the per-analysis report directory.

        Path format: ``{results_dir}/{safe_ticker}/{date}/reports/``
        """
        safe = safe_ticker_component(ticker)
        date_s = self._date_str(analysis_date)
        p = self.results_dir / safe / date_s / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_report_dir(
        self,
        ticker: str,
        timestamp: str | datetime,
        base: Path | None = None,
    ) -> Path:
        """Return the CLI ``save_report_to_disk`` directory.

        Path format: ``{base}/reports/{safe_ticker}_{timestamp}/``

        ``base`` defaults to ``Path.cwd()`` (matching existing behaviour),
        **not** ``results_dir``.
        """
        safe = safe_ticker_component(ticker)
        ts_str = (
            timestamp.strftime("%Y%m%d_%H%M%S")
            if isinstance(timestamp, datetime)
            else timestamp
        )
        root = (base or Path.cwd()) / "reports"
        p = root / f"{safe}_{ts_str}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # Memory artifact
    # ------------------------------------------------------------------

    def memory_log(self) -> Path:
        """Return the memory log path, ensuring its parent directory exists."""
        self.memory_log_path.parent.mkdir(parents=True, exist_ok=True)
        return self.memory_log_path

    # ------------------------------------------------------------------
    # OHLCV cache artifact
    # ------------------------------------------------------------------

    def ohlcv_cache(self, symbol: str, start: str, end: str) -> Path:
        """Return the OHLCV CSV cache path.

        Path format: ``{data_cache_dir}/{safe_symbol}-YFin-data-{start}-{end}.csv``
        """
        safe = safe_ticker_component(symbol)
        p = self.data_cache_dir / f"{safe}-YFin-data-{start}-{end}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # Targeted cleanup
    # ------------------------------------------------------------------

    def clear_checkpoint(self, ticker: str) -> int:
        """Remove the checkpoint DB for a specific ticker.

        Returns 1 if a file was deleted, 0 otherwise.
        """
        safe = safe_ticker_component(ticker).upper()
        target = self.data_cache_dir / "checkpoints" / f"{safe}.db"
        if target.exists():
            target.unlink()
            return 1
        return 0

    def clear_all_checkpoints(self) -> int:
        """Remove all checkpoint DBs.  Returns the number of files deleted."""
        cp_dir = self.data_cache_dir / "checkpoints"
        if not cp_dir.exists():
            return 0
        count = 0
        for db in cp_dir.glob("*.db"):
            db.unlink()
            count += 1
        return count

    def clear_results(
        self,
        ticker: str,
        analysis_date: Optional[date | str] = None,
    ) -> int:
        """Remove result artifacts for *ticker*.

        If *analysis_date* is ``None``, removes the entire per-ticker tree
        (``{results_dir}/{safe_ticker}/``).  Otherwise removes only the
        per-date subtree (``{results_dir}/{safe_ticker}/{date}/``).

        Returns 1 if a directory was removed, 0 otherwise.
        """
        safe = safe_ticker_component(ticker)
        if analysis_date is None:
            target = self.results_dir / safe
        else:
            date_s = self._date_str(analysis_date)
            target = self.results_dir / safe / date_s
        if not target.exists():
            return 0
        shutil.rmtree(target)
        return 1

    def clear_ohlcv_cache(self, symbol: Optional[str] = None) -> int:
        """Remove cached OHLCV CSVs.

        If *symbol* is given, only that symbol's caches are removed.
        Otherwise all ``*-YFin-data-*.csv`` files are deleted.

        Returns the number of files deleted.
        """
        if symbol is not None:
            safe = safe_ticker_component(symbol)
            pattern = f"{safe}-YFin-data-*.csv"
        else:
            pattern = "*-YFin-data-*.csv"
        count = 0
        for f in self.data_cache_dir.glob(pattern):
            f.unlink()
            count += 1
        return count

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ArtifactStore(results_dir={self.results_dir!r}, "
            f"data_cache_dir={self.data_cache_dir!r}, "
            f"memory_log_path={self.memory_log_path!r})"
        )


def store_from_config(config: dict | None = None) -> ArtifactStore:
    """Build an :class:`ArtifactStore` from a config dict.

    Reads ``results_dir``, ``data_cache_dir``, and ``memory_log_path``
    keys.  Missing or ``None`` values fall through to environment variables
    and ultimately the built-in defaults.
    """
    cfg = config or {}
    return ArtifactStore(
        results_dir=cfg.get("results_dir"),
        data_cache_dir=cfg.get("data_cache_dir"),
        memory_log_path=cfg.get("memory_log_path"),
    )
