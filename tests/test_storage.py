"""Tests for the unified ArtifactStore path policy layer.

Covers ticker safety validation, directory creation, targeted cleanup,
backward path compatibility, store_from_config, and edge cases.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import pytest

from tradingagents.storage import ArtifactStore, store_from_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> ArtifactStore:
    """ArtifactStore rooted in *tmp_path* for full isolation."""
    return ArtifactStore(
        results_dir=tmp_path / "logs",
        data_cache_dir=tmp_path / "cache",
        memory_log_path=tmp_path / "memory" / "trading_memory.md",
    )


# ===========================================================================
# Ticker safety validation
# ===========================================================================


@pytest.mark.unit
class TestTickerSafety:
    """Every path method must reject dangerous ticker values."""

    DANGEROUS = (
        "..",
        ".",
        "../etc",
        "a/b",
        "a\\b",
        "/abs",
        "..\\..\\x",
        "AAPL\x00",
        "AAPL\n",
        "\tAAPL",
        "",
        "..."
    )

    VALID = ("AAPL", "BRK-B", "BRK.A", "0700.HK", "^GSPC", "GC=F", "XAUUSD+")

    @pytest.mark.parametrize(
        "method",
        [
            "checkpoint_db",
            "state_log",
            "message_log",
        ],
    )
    def test_rejects_dangerous_tickers(self, store: ArtifactStore, method: str):
        fn = getattr(store, method)
        for bad in self.DANGEROUS:
            with pytest.raises(ValueError):
                # Some methods require a date argument
                if method in ("state_log", "message_log"):
                    fn(bad, "2024-01-01")
                else:
                    fn(bad)

    def test_report_dir_rejects_dangerous_tickers(self, store: ArtifactStore):
        for bad in self.DANGEROUS:
            with pytest.raises(ValueError):
                store.report_dir(bad, "2024-01-01")

    def test_save_report_dir_rejects_dangerous_tickers(self, store: ArtifactStore):
        for bad in self.DANGEROUS:
            with pytest.raises(ValueError):
                store.save_report_dir(bad, "20240115_120000")

    def test_ohlcv_cache_rejects_dangerous_symbols(self, store: ArtifactStore):
        for bad in self.DANGEROUS:
            with pytest.raises(ValueError):
                store.ohlcv_cache(bad, "2024-01-01", "2024-01-31")

    @pytest.mark.parametrize("ticker", VALID)
    def test_accepts_valid_tickers(self, store: ArtifactStore, ticker: str):
        # checkpoint_db is the simplest — just needs a ticker
        p = store.checkpoint_db(ticker)
        assert p.name  # should not raise


# ===========================================================================
# Directory creation
# ===========================================================================


@pytest.mark.unit
class TestDirectoryCreation:
    """Each path method must create its parent directory tree."""

    def test_ensure_base_dirs(self, store: ArtifactStore):
        store.ensure_base_dirs()
        assert store.results_dir.is_dir()
        assert store.data_cache_dir.is_dir()
        assert store.memory_log_path.parent.is_dir()

    def test_ensure_base_dirs_idempotent(self, store: ArtifactStore):
        store.ensure_base_dirs()
        store.ensure_base_dirs()  # must not raise

    def test_checkpoint_db_creates_parent(self, store: ArtifactStore):
        p = store.checkpoint_db("AAPL")
        assert p.parent.is_dir()

    def test_state_log_creates_parent(self, store: ArtifactStore):
        p = store.state_log("AAPL", date(2024, 3, 15))
        assert p.parent.is_dir()

    def test_message_log_creates_parent(self, store: ArtifactStore):
        p = store.message_log("AAPL", "2024-03-15")
        assert p.parent.is_dir()

    def test_report_dir_creates_dir(self, store: ArtifactStore):
        p = store.report_dir("AAPL", "2024-03-15")
        assert p.is_dir()

    def test_save_report_dir_creates_dir(self, store: ArtifactStore, tmp_path: Path):
        p = store.save_report_dir("AAPL", "20240115_120000", base=tmp_path)
        assert p.is_dir()

    def test_memory_log_creates_parent(self, store: ArtifactStore):
        p = store.memory_log()
        assert p.parent.is_dir()

    def test_ohlcv_cache_creates_parent(self, store: ArtifactStore):
        p = store.ohlcv_cache("SPY", "2024-01-01", "2024-06-30")
        assert p.parent.is_dir()


# ===========================================================================
# Path format — backward compatibility
# ===========================================================================


@pytest.mark.unit
class TestBackwardCompatibility:
    """Verify exact path format matches previous per-module implementations."""

    def test_checkpoint_db_format(self, store: ArtifactStore):
        """Previous: {data_cache_dir}/checkpoints/{TICKER_UPPER}.db"""
        p = store.checkpoint_db("Brk-B")
        assert p == store.data_cache_dir / "checkpoints" / "BRK-B.db"

    def test_checkpoint_db_uppercases(self, store: ArtifactStore):
        p = store.checkpoint_db("aapl")
        assert p.name == "AAPL.db"

    def test_state_log_format(self, store: ArtifactStore):
        """Previous: {results_dir}/{safe}/TradingAgentsStrategy_logs/full_states_log_{date}.json"""
        p = store.state_log("AAPL", date(2024, 3, 15))
        assert p == (
            store.results_dir
            / "AAPL"
            / "TradingAgentsStrategy_logs"
            / "full_states_log_2024-03-15.json"
        )

    def test_state_log_string_date(self, store: ArtifactStore):
        p = store.state_log("AAPL", "2024-03-15")
        assert p.name == "full_states_log_2024-03-15.json"

    def test_message_log_format(self, store: ArtifactStore):
        """Previous: {results_dir}/{ticker}/{date}/message_tool.log (but raw ticker)"""
        p = store.message_log("AAPL", "2024-03-15")
        assert p == store.results_dir / "AAPL" / "2024-03-15" / "message_tool.log"

    def test_report_dir_format(self, store: ArtifactStore):
        """Previous: {results_dir}/{ticker}/{date}/reports/"""
        p = store.report_dir("AAPL", "2024-03-15")
        assert p == store.results_dir / "AAPL" / "2024-03-15" / "reports"

    def test_save_report_dir_format(self, store: ArtifactStore, tmp_path: Path):
        """Previous: {cwd}/reports/{ticker}_{timestamp}/"""
        p = store.save_report_dir("AAPL", "20240115_120000", base=tmp_path)
        assert p == tmp_path / "reports" / "AAPL_20240115_120000"

    def test_save_report_dir_datetime(self, store: ArtifactStore, tmp_path: Path):
        dt = datetime(2024, 1, 15, 12, 0, 0)
        p = store.save_report_dir("AAPL", dt, base=tmp_path)
        assert p.name == "AAPL_20240115_120000"

    def test_memory_log_returns_configured_path(self, store: ArtifactStore):
        assert store.memory_log() == store.memory_log_path

    def test_ohlcv_cache_format(self, store: ArtifactStore):
        """Previous: {cache}/{safe}-YFin-data-{start}-{end}.csv"""
        p = store.ohlcv_cache("SPY", "2024-01-01", "2024-06-30")
        assert p == store.data_cache_dir / "SPY-YFin-data-2024-01-01-2024-06-30.csv"

    def test_message_log_date_object(self, store: ArtifactStore):
        """date objects should be formatted as YYYY-MM-DD."""
        p = store.message_log("AAPL", date(2024, 3, 15))
        assert "2024-03-15" in str(p)


# ===========================================================================
# Targeted cleanup
# ===========================================================================


@pytest.mark.unit
class TestCleanup:
    """Targeted cleanup for checkpoints, results, and OHLCV caches."""

    # -- Checkpoints --

    def test_clear_checkpoint_single(self, store: ArtifactStore):
        p = store.checkpoint_db("AAPL")
        p.touch()
        assert store.clear_checkpoint("AAPL") == 1
        assert not p.exists()

    def test_clear_checkpoint_nonexistent(self, store: ArtifactStore):
        assert store.clear_checkpoint("NOPE") == 0

    def test_clear_all_checkpoints(self, store: ArtifactStore):
        for t in ("AAPL", "MSFT", "GOOG"):
            store.checkpoint_db(t).touch()
        assert store.clear_all_checkpoints() == 3
        assert list((store.data_cache_dir / "checkpoints").glob("*.db")) == []

    def test_clear_all_checkpoints_empty(self, store: ArtifactStore):
        assert store.clear_all_checkpoints() == 0

    def test_clear_all_checkpoints_no_dir(self, store: ArtifactStore):
        # checkpoints dir doesn't even exist yet
        assert store.clear_all_checkpoints() == 0

    # -- Results --

    def test_clear_results_full_ticker(self, store: ArtifactStore):
        store.ensure_base_dirs()
        # Create some result artifacts
        d1 = store.message_log("AAPL", "2024-01-01")
        d2 = store.message_log("AAPL", "2024-01-02")
        (d1.parent).mkdir(parents=True, exist_ok=True)
        (d2.parent).mkdir(parents=True, exist_ok=True)
        d1.touch()
        d2.touch()
        assert (store.results_dir / "AAPL").is_dir()
        assert store.clear_results("AAPL") == 1
        assert not (store.results_dir / "AAPL").exists()

    def test_clear_results_specific_date(self, store: ArtifactStore):
        store.ensure_base_dirs()
        d1_parent = store.message_log("AAPL", "2024-01-01").parent
        d2_parent = store.message_log("AAPL", "2024-01-02").parent
        d1_parent.mkdir(parents=True, exist_ok=True)
        d2_parent.mkdir(parents=True, exist_ok=True)
        assert store.clear_results("AAPL", "2024-01-01") == 1
        assert not d1_parent.exists()
        assert d2_parent.exists()

    def test_clear_results_nonexistent(self, store: ArtifactStore):
        store.ensure_base_dirs()
        assert store.clear_results("NOPE") == 0

    def test_clear_results_date_object(self, store: ArtifactStore):
        store.ensure_base_dirs()
        d_parent = store.message_log("AAPL", date(2024, 1, 1)).parent
        d_parent.mkdir(parents=True, exist_ok=True)
        assert store.clear_results("AAPL", date(2024, 1, 1)) == 1

    # -- OHLCV cache --

    def test_clear_ohlcv_by_symbol(self, store: ArtifactStore):
        store.ohlcv_cache("AAPL", "2024-01-01", "2024-01-31").touch()
        store.ohlcv_cache("MSFT", "2024-01-01", "2024-01-31").touch()
        assert store.clear_ohlcv_cache("AAPL") == 1
        # MSFT untouched
        assert list(store.data_cache_dir.glob("*MSFT*"))

    def test_clear_ohlcv_all(self, store: ArtifactStore):
        store.ohlcv_cache("AAPL", "2024-01-01", "2024-01-31").touch()
        store.ohlcv_cache("MSFT", "2024-01-01", "2024-01-31").touch()
        assert store.clear_ohlcv_cache() == 2

    def test_clear_ohlcv_empty(self, store: ArtifactStore):
        store.ensure_base_dirs()
        assert store.clear_ohlcv_cache() == 0


# ===========================================================================
# store_from_config
# ===========================================================================


@pytest.mark.unit
class TestStoreFromConfig:

    def test_reads_config_keys(self, tmp_path: Path):
        cfg = {
            "results_dir": str(tmp_path / "r"),
            "data_cache_dir": str(tmp_path / "c"),
            "memory_log_path": str(tmp_path / "m" / "mem.md"),
        }
        s = store_from_config(cfg)
        assert s.results_dir == (tmp_path / "r").resolve()
        assert s.data_cache_dir == (tmp_path / "c").resolve()
        assert s.memory_log_path == (tmp_path / "m" / "mem.md").resolve()

    def test_falls_back_to_defaults(self):
        s = store_from_config({})
        home = Path.home()
        assert s.results_dir == (home / ".tradingagents" / "logs").resolve()
        assert s.data_cache_dir == (home / ".tradingagents" / "cache").resolve()

    def test_none_config(self):
        s = store_from_config(None)
        home = Path.home()
        assert s.results_dir == (home / ".tradingagents" / "logs").resolve()

    def test_partial_config(self, tmp_path: Path):
        cfg = {"results_dir": str(tmp_path / "custom_logs")}
        s = store_from_config(cfg)
        assert s.results_dir == (tmp_path / "custom_logs").resolve()
        # Other keys fall through to defaults
        home = Path.home()
        assert s.data_cache_dir == (home / ".tradingagents" / "cache").resolve()


# ===========================================================================
# Repr
# ===========================================================================


@pytest.mark.unit
class TestRepr:

    def test_repr(self, store: ArtifactStore):
        r = repr(store)
        assert "ArtifactStore" in r
        assert "results_dir=" in r
        assert "data_cache_dir=" in r
        assert "memory_log_path=" in r


# ===========================================================================
# Edge cases
# ===========================================================================


@pytest.mark.unit
class TestEdgeCases:

    def test_path_resolution(self, tmp_path: Path):
        """Paths should be resolved to absolute."""
        s = ArtifactStore(
            results_dir=tmp_path / "logs",
            data_cache_dir=tmp_path / "cache",
            memory_log_path=tmp_path / "memory" / "mem.md",
        )
        assert s.results_dir.is_absolute()
        assert s.data_cache_dir.is_absolute()
        assert s.memory_log_path.is_absolute()

    def test_string_and_path_args(self, tmp_path: Path):
        """Accept both str and Path for all constructor args."""
        s = ArtifactStore(
            results_dir=str(tmp_path / "a"),
            data_cache_dir=Path(tmp_path / "b"),
            memory_log_path=str(tmp_path / "c" / "m.md"),
        )
        assert s.results_dir == (tmp_path / "a").resolve()

    def test_checkpoint_db_rejects_overlong_ticker(self, store: ArtifactStore):
        with pytest.raises(ValueError, match="exceeds"):
            store.checkpoint_db("A" * 33)

    def test_message_log_traversal_fix(self, store: ArtifactStore):
        """Regression: CLI previously used raw ticker without validation."""
        with pytest.raises(ValueError):
            store.message_log("../malicious", "2024-01-01")

    def test_report_dir_traversal_fix(self, store: ArtifactStore):
        with pytest.raises(ValueError):
            store.report_dir("../../../etc", "2024-01-01")

    def test_save_report_dir_defaults_to_cwd(self, store: ArtifactStore, monkeypatch, tmp_path: Path):
        """When base is None, defaults to Path.cwd()."""
        monkeypatch.chdir(tmp_path)
        p = store.save_report_dir("AAPL", "20240115_120000")
        assert p.parent == tmp_path / "reports"
