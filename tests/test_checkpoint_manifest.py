"""Tests for checkpoint_manifest: RunManifest data model and storage I/O."""

import json
import tempfile
import unittest
from pathlib import Path

from tradingagents.graph.checkpoint_manifest import (
    MANIFEST_VERSION,
    RunManifest,
    _manifest_path,
    clear_all_manifests,
    delete_manifest_entry,
    load_manifest,
    save_manifest,
)


def _make_manifest(**overrides) -> RunManifest:
    """Helper: build a RunManifest with sensible defaults, overriding any field."""
    defaults = dict(
        thread_id="abc123",
        ticker="TEST",
        trade_date="2026-04-20",
        asset_type="stock",
        selected_analysts=("market", "social", "news", "fundamentals"),
        llm_provider="openai",
        deep_think_llm="gpt-5.5",
        quick_think_llm="gpt-5.4-mini",
        temperature=0.2,
        backend_url=None,
        last_completed_step=3,
        created_at="2026-04-20T10:00:00+00:00",
        updated_at="2026-04-20T10:00:00+00:00",
    )
    defaults.update(overrides)
    return RunManifest(**defaults)


class TestRunManifestSerialization(unittest.TestCase):
    """Roundtrip and edge-case tests for RunManifest."""

    def test_save_and_load_roundtrip(self):
        """to_dict → json → from_dict preserves all fields."""
        m = _make_manifest()
        d = m.to_dict()
        restored = RunManifest.from_dict(d)
        self.assertEqual(m, restored)

    def test_from_dict_ignores_unknown_keys(self):
        """Extra keys from a future version are silently dropped."""
        d = _make_manifest().to_dict()
        d["future_field"] = "should be ignored"
        restored = RunManifest.from_dict(d)
        self.assertEqual(restored.version, MANIFEST_VERSION)
        self.assertFalse(hasattr(restored, "future_field"))

    def test_from_config_extraction(self):
        """from_config pulls the right keys from a DEFAULT_CONFIG-like dict."""
        config = {
            "llm_provider": "anthropic",
            "deep_think_llm": "claude-4-sonnet",
            "quick_think_llm": "claude-4-haiku",
            "temperature": 0.5,
            "backend_url": "https://api.example.com",
        }
        m = RunManifest.from_config(
            config=config,
            ticker="AAPL",
            trade_date="2026-06-01",
            asset_type="stock",
            selected_analysts=["market", "news"],
            tid="thread123",
            step=5,
        )
        self.assertEqual(m.llm_provider, "anthropic")
        self.assertEqual(m.deep_think_llm, "claude-4-sonnet")
        self.assertEqual(m.quick_think_llm, "claude-4-haiku")
        self.assertEqual(m.temperature, 0.5)
        self.assertEqual(m.backend_url, "https://api.example.com")
        self.assertEqual(m.selected_analysts, ("market", "news"))
        self.assertEqual(m.thread_id, "thread123")
        self.assertEqual(m.last_completed_step, 5)
        self.assertEqual(m.ticker, "AAPL")
        self.assertEqual(m.asset_type, "stock")

    def test_with_step_returns_copy(self):
        """with_step returns a new manifest with updated step and timestamp."""
        m = _make_manifest(last_completed_step=1, updated_at="old")
        m2 = m.with_step(7)
        self.assertEqual(m.last_completed_step, 1)  # original unchanged
        self.assertEqual(m2.last_completed_step, 7)
        self.assertNotEqual(m2.updated_at, "old")

    def test_json_roundtrip_via_file(self):
        """Manifests survive json.dumps/loads."""
        m = _make_manifest()
        text = json.dumps(m.to_dict())
        restored = RunManifest.from_dict(json.loads(text))
        self.assertEqual(m, restored)


class TestManifestStorage(unittest.TestCase):
    """Tests for the file-backed save/load/delete API."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ticker = "TEST"

    def test_manifest_path_location(self):
        """Path is {data_dir}/checkpoints/{TICKER}.manifest.json."""
        path = _manifest_path(self.tmpdir, self.ticker)
        expected = Path(self.tmpdir) / "checkpoints" / "TEST.manifest.json"
        self.assertEqual(path, expected)

    def test_load_missing_returns_none(self):
        """No file → None."""
        result = load_manifest(self.tmpdir, self.ticker, "nonexistent")
        self.assertIsNone(result)

    def test_load_corrupt_json_returns_none(self):
        """Corrupt JSON file → None (graceful)."""
        path = _manifest_path(self.tmpdir, self.ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{{{not valid json", encoding="utf-8")
        result = load_manifest(self.tmpdir, self.ticker, "any_thread")
        self.assertIsNone(result)

    def test_load_empty_file_returns_none(self):
        """Empty file → None."""
        path = _manifest_path(self.tmpdir, self.ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        result = load_manifest(self.tmpdir, self.ticker, "any_thread")
        self.assertIsNone(result)

    def test_save_and_load(self):
        """Save → Load roundtrip through the file system."""
        m = _make_manifest(ticker=self.ticker)
        save_manifest(self.tmpdir, m)
        loaded = load_manifest(self.tmpdir, self.ticker, m.thread_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded, m)

    def test_multi_thread_same_ticker(self):
        """Two different threads (dates) coexist in one manifest file."""
        m1 = _make_manifest(thread_id="thread_a", trade_date="2026-04-20")
        m2 = _make_manifest(thread_id="thread_b", trade_date="2026-04-21")
        save_manifest(self.tmpdir, m1)
        save_manifest(self.tmpdir, m2)

        loaded_a = load_manifest(self.tmpdir, self.ticker, "thread_a")
        loaded_b = load_manifest(self.tmpdir, self.ticker, "thread_b")

        self.assertEqual(loaded_a.trade_date, "2026-04-20")
        self.assertEqual(loaded_b.trade_date, "2026-04-21")

    def test_save_overwrites_same_thread(self):
        """Saving the same thread_id updates the entry in place."""
        m1 = _make_manifest(last_completed_step=1)
        m2 = _make_manifest(last_completed_step=5)
        save_manifest(self.tmpdir, m1)
        save_manifest(self.tmpdir, m2)

        loaded = load_manifest(self.tmpdir, self.ticker, m1.thread_id)
        self.assertEqual(loaded.last_completed_step, 5)

    def test_delete_single_entry(self):
        """Deleting one thread leaves the other intact."""
        m1 = _make_manifest(thread_id="keep_me")
        m2 = _make_manifest(thread_id="delete_me")
        save_manifest(self.tmpdir, m1)
        save_manifest(self.tmpdir, m2)

        removed = delete_manifest_entry(self.tmpdir, self.ticker, "delete_me")
        self.assertTrue(removed)

        self.assertIsNotNone(load_manifest(self.tmpdir, self.ticker, "keep_me"))
        self.assertIsNone(load_manifest(self.tmpdir, self.ticker, "delete_me"))

    def test_delete_last_entry_removes_file(self):
        """When the last entry is deleted, the manifest file itself is removed."""
        m = _make_manifest()
        save_manifest(self.tmpdir, m)
        path = _manifest_path(self.tmpdir, self.ticker)
        self.assertTrue(path.exists())

        delete_manifest_entry(self.tmpdir, self.ticker, m.thread_id)
        self.assertFalse(path.exists())

    def test_delete_nonexistent_returns_false(self):
        """Deleting a thread that doesn't exist returns False."""
        result = delete_manifest_entry(self.tmpdir, self.ticker, "ghost")
        self.assertFalse(result)

    def test_atomic_write_preserves_old_data(self):
        """A failed save attempt does not corrupt existing data.

        We simulate failure by patching os.replace to raise, then verify
        the old manifest is still intact.
        """
        m1 = _make_manifest(last_completed_step=1)
        save_manifest(self.tmpdir, m1)

        import tradingagents.graph.checkpoint_manifest as mod

        original_replace = mod.os.replace

        def failing_replace(*args, **kwargs):
            raise OSError("simulated disk failure")

        mod.os.replace = failing_replace
        try:
            m2 = _make_manifest(last_completed_step=99)
            with self.assertRaises(OSError):
                save_manifest(self.tmpdir, m2)
        finally:
            mod.os.replace = original_replace

        # Old data must still be intact.
        loaded = load_manifest(self.tmpdir, self.ticker, m1.thread_id)
        self.assertEqual(loaded.last_completed_step, 1)

    def test_clear_all_manifests(self):
        """Deletes every *.manifest.json, returns count."""
        # Create manifests for two tickers.
        save_manifest(self.tmpdir, _make_manifest(ticker="AAA"))
        save_manifest(self.tmpdir, _make_manifest(ticker="BBB"))
        save_manifest(self.tmpdir, _make_manifest(ticker="CCC"))

        count = clear_all_manifests(self.tmpdir)
        self.assertEqual(count, 3)

        # All gone.
        self.assertIsNone(load_manifest(self.tmpdir, "AAA", "abc123"))
        self.assertIsNone(load_manifest(self.tmpdir, "BBB", "abc123"))

    def test_clear_all_manifests_empty_dir(self):
        """No manifest files → returns 0."""
        count = clear_all_manifests(self.tmpdir)
        self.assertEqual(count, 0)

    def test_clear_all_manifests_no_dir(self):
        """Checkpoints dir doesn't exist → returns 0."""
        count = clear_all_manifests(Path(self.tmpdir) / "nonexistent")
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
