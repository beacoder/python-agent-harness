"""Tests for clipboard image capture (python_agent_harness.clipboard)."""

import os
import subprocess
import unittest
import unittest.mock as mock

from python_agent_harness import clipboard

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _completed(returncode=0, stdout=b""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=b"")


class TestGrabClipboardImage(unittest.TestCase):
    def tearDown(self):
        # drop any temp files the capture tracked, so tests don't leak
        from python_agent_harness.tools import filesystem

        filesystem.cleanup_spooled_files()

    def test_capture_writes_tracked_temp_png(self):
        """A valid PNG on the clipboard is written to a temp file whose
        path is returned and tracked for cleanup."""
        from python_agent_harness.tools import filesystem

        with (
            mock.patch("shutil.which", return_value="/usr/bin/xclip"),
            mock.patch("subprocess.run", return_value=_completed(0, _PNG)),
            mock.patch.object(clipboard.sys, "platform", "linux"),
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("WAYLAND_DISPLAY", None)
            path = clipboard.grab_clipboard_image()
        self.assertIsNotNone(path)
        assert path is not None
        self.assertTrue(os.path.isfile(path))
        with open(path, "rb") as f:
            self.assertTrue(f.read().startswith(b"\x89PNG"))
        with filesystem._spooled_files_lock:
            self.assertIn(path, filesystem._spooled_files)

    def test_no_tool_returns_none(self):
        with (
            mock.patch("shutil.which", return_value=None),
            mock.patch.object(clipboard.sys, "platform", "linux"),
        ):
            os.environ.pop("WAYLAND_DISPLAY", None)
            self.assertIsNone(clipboard.grab_clipboard_image())

    def test_tool_nonzero_exit_returns_none(self):
        """Tool exits non-zero (no image on clipboard) -> None."""
        with (
            mock.patch("shutil.which", return_value="/usr/bin/xclip"),
            mock.patch("subprocess.run", return_value=_completed(1, b"")),
            mock.patch.object(clipboard.sys, "platform", "linux"),
        ):
            os.environ.pop("WAYLAND_DISPLAY", None)
            self.assertIsNone(clipboard.grab_clipboard_image())

    def test_non_png_output_rejected(self):
        """Output that isn't PNG-signed (e.g. text) is rejected."""
        with (
            mock.patch("shutil.which", return_value="/usr/bin/xclip"),
            mock.patch("subprocess.run", return_value=_completed(0, b"not an image")),
            mock.patch.object(clipboard.sys, "platform", "linux"),
        ):
            os.environ.pop("WAYLAND_DISPLAY", None)
            self.assertIsNone(clipboard.grab_clipboard_image())

    def test_oversized_capture_rejected(self):
        big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (clipboard._MAX_CAPTURE + 1)
        with (
            mock.patch("shutil.which", return_value="/usr/bin/xclip"),
            mock.patch("subprocess.run", return_value=_completed(0, big)),
            mock.patch.object(clipboard.sys, "platform", "linux"),
        ):
            os.environ.pop("WAYLAND_DISPLAY", None)
            self.assertIsNone(clipboard.grab_clipboard_image())

    def test_subprocess_error_returns_none(self):
        with (
            mock.patch("shutil.which", return_value="/usr/bin/xclip"),
            mock.patch("subprocess.run", side_effect=OSError("boom")),
            mock.patch.object(clipboard.sys, "platform", "linux"),
        ):
            os.environ.pop("WAYLAND_DISPLAY", None)
            self.assertIsNone(clipboard.grab_clipboard_image())

    def test_timeout_returns_none(self):
        with (
            mock.patch("shutil.which", return_value="/usr/bin/xclip"),
            mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("xclip", 5)),
            mock.patch.object(clipboard.sys, "platform", "linux"),
        ):
            os.environ.pop("WAYLAND_DISPLAY", None)
            self.assertIsNone(clipboard.grab_clipboard_image())

    def test_wayland_preferred_when_env_set(self):
        """WAYLAND_DISPLAY hint tries wl-paste first."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd[0])
            if cmd[0] == "wl-paste":
                return _completed(0, _PNG)
            return _completed(1, b"")

        with (
            mock.patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"),
            mock.patch("subprocess.run", side_effect=fake_run),
            mock.patch.object(clipboard.sys, "platform", "linux"),
            mock.patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}),
        ):
            path = clipboard.grab_clipboard_image()
        self.assertIsNotNone(path)
        self.assertEqual(calls[0], "wl-paste")

    def test_macos_uses_pngpaste(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd[0])
            return _completed(0, _PNG)

        with (
            mock.patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"),
            mock.patch("subprocess.run", side_effect=fake_run),
            mock.patch.object(clipboard.sys, "platform", "darwin"),
        ):
            path = clipboard.grab_clipboard_image()
        self.assertIsNotNone(path)
        self.assertEqual(calls, ["pngpaste"])


if __name__ == "__main__":
    unittest.main()
