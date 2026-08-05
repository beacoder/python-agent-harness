import os
import tempfile
import time
import unittest

from python_agent_harness.cache import ToolCache


class TestCache(unittest.TestCase):
    def setUp(self):
        self.cache = ToolCache()

    def test_miss_then_hit(self):
        key = ("read", "/tmp/x.py", None, None)
        self.assertIsNone(self.cache.get(key, "/tmp/x.py"))
        self.cache.store(key, "content", "/tmp/x.py")
        self.cache.mark_seen(key)
        # seen -> dedup marker on repeat
        result = self.cache.get(key, "/tmp/x.py")
        self.assertIn("[Cached: Read", result)
        self.assertIn("same as earlier call, see above", result)

    def test_epoch_reset(self):
        key = ("glob", "*.py", "/tmp", None)
        self.cache.store(key, "a.py")
        self.cache.mark_seen(key)
        first = self.cache.get(key)
        self.assertIn("[Cached: Glob", first)
        self.assertIn("same as earlier call, see above", first)
        self.cache.reset_epoch()
        # full result again
        result = self.cache.get(key)
        self.assertEqual(result, "a.py")

    def test_mtime_invalidation(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("v1")
            path = f.name
        try:
            key = ("read", path, None, None)
            self.cache.store(key, "v1", path)
            self.cache.mark_seen(key)
            time.sleep(0.02)
            with open(path, "w") as f:
                f.write("v2")
            # mtime changed -> stale
            self.assertIsNone(self.cache.get(key, path))
        finally:
            os.unlink(path)

    def test_ttl_for_directories(self):
        self.cache.ttl = 0
        key = ("glob", "*.py", "/tmp", None)
        self.cache.store(key, "x.py")
        time.sleep(0.01)
        self.assertIsNone(self.cache.get(key, "/tmp"))

    def test_write_through_invalidation(self):
        key = ("grep", "foo", "/tmp/dir", None, None)
        self.cache.store(key, "hit", "/tmp/dir")
        self.cache.invalidate_path("/tmp/dir/file.py")
        self.assertNotIn(key, self.cache.table)

    def test_eviction(self):
        cache = ToolCache(max_entries=2)
        cache.store(("a", 1), "r1")
        cache.store(("b", 2), "r2")
        cache.store(("c", 3), "r3")
        self.assertEqual(len(cache.table), 2)
        self.assertNotIn(("a", 1), cache.table)

    def test_cacheable(self):
        self.assertFalse(ToolCache.cacheable_p(""))
        self.assertFalse(ToolCache.cacheable_p("Error: boom"))
        self.assertFalse(ToolCache.cacheable_p("x failed with exit code 1"))
        self.assertTrue(ToolCache.cacheable_p("ok"))

    def test_dedup_message_formats(self):
        msg = self.cache.dedup_message("read", ("/tmp/a.py", 1, 10), "x" * 50)
        self.assertIn('Read "/tmp/a.py" lines 1-10 (50 chars)', msg)
        msg = self.cache.dedup_message("glob", ("*.py", "/tmp"), "x" * 3)
        self.assertIn('Glob "*.py" in /tmp (3 chars)', msg)


if __name__ == "__main__":
    unittest.main()
