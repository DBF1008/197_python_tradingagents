"""Tests for ArtifactStore — the unified run-artifact path policy.

Coverage maps to the four guarantees of the storage layer:
  * path compatibility — the on-disk layout is byte-identical to the
    pre-refactor layout for valid tickers (the "do not change the default
    directory layout" invariant);
  * ticker safety — every write-path method runs the traversal check
    (previously the CLI run directory took the raw ticker);
  * directory creation — opt-in ``create=`` makes the right parents, and the
    ``expanduser`` behaviour the memory log relies on is preserved;
  * targeted cleanup — ``clear_run`` and ``clear_checkpoints`` remove only
    their own artifacts and leave siblings intact.
"""

import os

import pytest

from tradingagents.storage import ArtifactStore


def _store(tmp_path):
    """A fully-configured store rooted under a pytest tmp dir."""
    return ArtifactStore(
        results_dir=tmp_path / "logs",
        cache_dir=tmp_path / "cache",
        memory_log_path=tmp_path / "memory" / "trading_memory.md",
    )


# ---------------------------------------------------------------------------
# Path compatibility — exact-string guards on the on-disk layout
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPathCompatibility:

    def test_checkpoint_db_is_uppercased_under_cache(self, tmp_path):
        s = _store(tmp_path)
        expected = tmp_path / "cache" / "checkpoints" / "AAPL.db"
        # Checkpoints upper-case the ticker (legacy convention) — both cases land
        # on the same file.
        assert s.checkpoint_db("aapl") == expected
        assert s.checkpoint_db("AAPL") == expected

    def test_state_log_preserves_ticker_case(self, tmp_path):
        s = _store(tmp_path)
        expected = (
            tmp_path / "logs" / "brk.b" / "TradingAgentsStrategy_logs"
            / "full_states_log_2026-01-10.json"
        )
        assert s.state_log_file("brk.b", "2026-01-10") == expected

    def test_run_dir_reports_and_message_log(self, tmp_path):
        s = _store(tmp_path)
        run = tmp_path / "logs" / "AAPL" / "2026-01-10"
        assert s.run_dir("AAPL", "2026-01-10") == run
        assert s.reports_dir("AAPL", "2026-01-10") == run / "reports"
        assert s.report_file("AAPL", "2026-01-10", "market_report") == run / "reports" / "market_report.md"
        assert s.message_log("AAPL", "2026-01-10") == run / "message_tool.log"

    def test_memory_log_and_tmp_sibling(self, tmp_path):
        s = _store(tmp_path)
        assert s.memory_log_file() == tmp_path / "memory" / "trading_memory.md"
        # Atomic-write sibling must be "<stem>.tmp" (the memory log relies on it).
        assert s.memory_log_tmp_file() == tmp_path / "memory" / "trading_memory.tmp"

    def test_matches_legacy_formulas(self, tmp_path):
        """Cross-check against the literal pre-refactor path expressions."""
        s = _store(tmp_path)
        cache, logs = tmp_path / "cache", tmp_path / "logs"
        # checkpointer._db_path: Path(data_dir)/"checkpoints"/f"{safe.upper()}.db"
        assert s.checkpoint_db("AAPL", create=False) == cache / "checkpoints" / "AAPL.db"
        # trading_graph._log_state path build
        assert (
            s.state_log_file("AAPL", "2026-01-10", create=False)
            == logs / "AAPL" / "TradingAgentsStrategy_logs" / "full_states_log_2026-01-10.json"
        )
        # cli/main run_analysis run dir + message log
        assert s.run_dir("AAPL", "2026-01-10") == logs / "AAPL" / "2026-01-10"
        assert s.message_log("AAPL", "2026-01-10") == logs / "AAPL" / "2026-01-10" / "message_tool.log"


# ---------------------------------------------------------------------------
# Ticker safety — every write path rejects traversal values
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTickerSafety:

    BAD_TICKERS = ["../etc", "a/b", "a\\b", "..", ".", "", "A" * 33]

    @pytest.mark.parametrize("bad", BAD_TICKERS)
    def test_every_write_path_rejects_unsafe_ticker(self, tmp_path, bad):
        s = _store(tmp_path)
        calls = (
            lambda: s.checkpoint_db(bad),
            lambda: s.state_log_file(bad, "2026-01-10"),
            lambda: s.state_logs_dir(bad),
            lambda: s.run_dir(bad, "2026-01-10"),
            lambda: s.reports_dir(bad, "2026-01-10"),
            lambda: s.report_file(bad, "2026-01-10", "market_report"),
            lambda: s.message_log(bad, "2026-01-10"),
            lambda: s.clear_run(bad, "2026-01-10"),
        )
        for call in calls:
            with pytest.raises(ValueError):
                call()

    def test_unsafe_ticker_creates_no_directories(self, tmp_path):
        """Validation happens before any mkdir, so a rejected ticker leaves no trace."""
        s = _store(tmp_path)
        with pytest.raises(ValueError):
            s.checkpoint_db("../escape")  # create=True default
        assert not (tmp_path / "cache" / "checkpoints").exists()

    def test_safe_component_passthrough(self):
        # Valid tickers are returned verbatim — this is what keeps the layout
        # unchanged when the previously-unguarded CLI path now validates.
        for t in ("AAPL", "BRK.B", "0700.HK", "GC=F", "XAUUSD+", "^GSPC"):
            assert ArtifactStore.safe_component(t) == t


# ---------------------------------------------------------------------------
# Directory creation — the create= flag and expanduser
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDirectoryCreation:

    def test_checkpoint_db_creates_dir_by_default(self, tmp_path):
        s = _store(tmp_path)
        assert not (tmp_path / "cache" / "checkpoints").exists()
        db = s.checkpoint_db("AAPL")  # create=True default — sqlite needs the dir
        assert db.parent.exists()
        assert not db.exists()  # the .db file itself is the caller's job

    def test_checkpoints_dir_no_create_default(self, tmp_path):
        s = _store(tmp_path)
        s.checkpoints_dir()  # create=False
        assert not (tmp_path / "cache" / "checkpoints").exists()

    def test_state_log_file_creates_nested_parents(self, tmp_path):
        s = _store(tmp_path)
        f = s.state_log_file("AAPL", "2026-01-10")  # create=True default
        assert f.parent.exists()
        assert f.parent == tmp_path / "logs" / "AAPL" / "TradingAgentsStrategy_logs"

    def test_run_and_reports_create_flag(self, tmp_path):
        s = _store(tmp_path)
        assert not s.run_dir("AAPL", "2026-01-10").exists()  # create=False default
        rd = s.reports_dir("AAPL", "2026-01-10", create=True)
        assert rd.exists()
        assert (tmp_path / "logs" / "AAPL" / "2026-01-10").exists()  # parent made too

    def test_message_log_create_makes_run_dir(self, tmp_path):
        s = _store(tmp_path)
        p = s.message_log("AAPL", "2026-01-10", create=True)
        assert p.parent.exists()
        assert not p.exists()

    def test_ensure_base_dirs_makes_results_and_cache_only(self, tmp_path):
        s = _store(tmp_path)
        s.ensure_base_dirs()
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "cache").exists()
        # Memory parent is created lazily, not by ensure_base_dirs (parity with
        # the old TradingAgentsGraph.__init__).
        assert not (tmp_path / "memory").exists()

    def test_memory_log_file_creates_parent_only(self, tmp_path):
        s = _store(tmp_path)
        assert not (tmp_path / "memory").exists()
        p = s.memory_log_file(create=True)
        assert p.parent.exists()
        assert not p.exists()

    def test_memory_log_file_no_create_default(self, tmp_path):
        s = _store(tmp_path)
        s.memory_log_file()
        assert not (tmp_path / "memory").exists()

    def test_memory_log_path_is_expanduser_ed(self):
        # A "~/..." config value must resolve to the home dir, not a literal "~".
        # Use create=False so the test never touches the real home directory.
        s = ArtifactStore(memory_log_path="~/.tradingagents_pathtest/mem.md")
        p = s.memory_log_file()
        assert "~" not in str(p)
        assert str(p).startswith(os.path.expanduser("~"))


# ---------------------------------------------------------------------------
# Targeted cleanup — clear_run and clear_checkpoints
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCleanup:

    def test_clear_run_removes_only_target(self, tmp_path):
        s = _store(tmp_path)
        target = s.run_dir("AAPL", "2026-01-10", create=True)
        sibling = s.run_dir("AAPL", "2026-01-11", create=True)
        (target / "x.txt").write_text("a", encoding="utf-8")
        (sibling / "y.txt").write_text("b", encoding="utf-8")
        cp = s.checkpoints_dir(create=True)

        assert s.clear_run("AAPL", "2026-01-10") is True
        assert not target.exists()
        assert sibling.exists() and (sibling / "y.txt").exists()  # sibling untouched
        assert cp.exists()  # checkpoints untouched

    def test_clear_run_absent_returns_false(self, tmp_path):
        s = _store(tmp_path)
        assert s.clear_run("AAPL", "2099-01-01") is False

    def test_clear_checkpoints_removes_only_db_files(self, tmp_path):
        s = _store(tmp_path)
        cp = s.checkpoint_db("AAPL").parent  # creates the dir
        (cp / "AAPL.db").write_text("x", encoding="utf-8")
        (cp / "MSFT.db").write_text("y", encoding="utf-8")
        (cp / "keep.txt").write_text("z", encoding="utf-8")
        # A state-log tree under results_dir must survive checkpoint cleanup.
        state_log = s.state_log_file("AAPL", "2026-01-10")
        state_log.write_text("{}", encoding="utf-8")

        assert s.clear_checkpoints() == 2
        assert not (cp / "AAPL.db").exists()
        assert not (cp / "MSFT.db").exists()
        assert (cp / "keep.txt").exists()  # non-db left in place
        assert state_log.exists()  # state-log tree intact

    def test_clear_checkpoints_missing_dir_returns_zero(self, tmp_path):
        s = _store(tmp_path)
        assert s.clear_checkpoints() == 0
        assert not (tmp_path / "cache" / "checkpoints").exists()  # no empty dir created


# ---------------------------------------------------------------------------
# from_config / None handling
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFromConfigAndMissingRoots:

    def test_from_config_reads_all_keys(self, tmp_path):
        cfg = {
            "results_dir": str(tmp_path / "logs"),
            "data_cache_dir": str(tmp_path / "cache"),
            "memory_log_path": str(tmp_path / "m" / "x.md"),
        }
        s = ArtifactStore.from_config(cfg)
        assert s.checkpoint_db("AAPL", create=False) == tmp_path / "cache" / "checkpoints" / "AAPL.db"
        assert s.state_log_file("AAPL", "2026-01-10", create=False).parent.parent == tmp_path / "logs" / "AAPL"
        assert s.memory_log_file() == tmp_path / "m" / "x.md"

    def test_from_config_none_and_empty_have_no_memory_path(self):
        for cfg in (None, {}):
            s = ArtifactStore.from_config(cfg)
            assert s.memory_log_file() is None
            assert s.memory_log_tmp_file() is None

    def test_missing_root_raises_clear_error(self):
        s = ArtifactStore.from_config({})  # no results_dir / cache_dir configured
        with pytest.raises(ValueError):
            s.checkpoint_db("AAPL")
        with pytest.raises(ValueError):
            s.state_log_file("AAPL", "2026-01-10")
        with pytest.raises(ValueError):
            s.run_dir("AAPL", "2026-01-10")


# ---------------------------------------------------------------------------
# Checkpointer wiring — the public functions route through the store
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckpointerRoutesThroughStore:

    def test_db_path_matches_store_and_creates_dir(self, tmp_path):
        from tradingagents.graph.checkpointer import _db_path
        data_dir = tmp_path / "cache"
        got = _db_path(str(data_dir), "aapl")
        assert got == ArtifactStore(cache_dir=data_dir).checkpoint_db("aapl")
        assert got == data_dir / "checkpoints" / "AAPL.db"  # preserved layout
        assert got.parent.exists()  # sqlite connect needs the dir to pre-exist

    def test_clear_all_checkpoints_matches_store(self, tmp_path):
        from tradingagents.graph.checkpointer import _db_path, clear_all_checkpoints
        data_dir = tmp_path / "cache"
        db = _db_path(str(data_dir), "AAPL")
        db.write_text("x", encoding="utf-8")
        assert clear_all_checkpoints(str(data_dir)) == 1
        assert not db.exists()
