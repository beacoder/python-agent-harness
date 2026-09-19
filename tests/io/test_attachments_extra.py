"""Extra attachment tests: signature check edge, path resolution, and
the unreadable/oversized branches of text and unknown-extension files."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from python_agent_harness.io.attachments import (
    MAX_TEXT_SIZE,
    AttachmentError,
    ParsedAttachment,
    _has_image_signature,
    _resolve_path,
    _validate_and_create,
)


class TestHasImageSignature(unittest.TestCase):
    def test_unknown_media_type_accepted(self):
        self.assertTrue(_has_image_signature(b"", "image/unknown-format"))

    def test_known_signature_matches(self):
        self.assertTrue(_has_image_signature(b"\x89PNG\r\n\x1a\nrest", "image/png"))

    def test_too_short_data_rejected(self):
        self.assertFalse(_has_image_signature(b"\x89P", "image/png"))


class TestResolvePath(unittest.TestCase):
    def test_absolute_path_returned_as_is(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(_resolve_path(d, "/some/project"), d)

    def test_relative_path_joined_with_project(self):
        self.assertEqual(
            _resolve_path("sub/file.txt", os.path.join("proj", "root")),
            os.path.join("proj", "root", "sub/file.txt"),
        )


class TestValidateAndCreate(unittest.TestCase):
    def _file(self, d: str, name: str, content: bytes = b"data\n") -> str:
        path = os.path.join(d, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def _huge_file(self, d: str, name: str) -> str:
        path = os.path.join(d, name)
        with open(path, "wb") as f:
            f.truncate(MAX_TEXT_SIZE + 1)
        return path

    def test_text_file_read_failure_reported(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._file(d, "notes.txt")
            with mock.patch("builtins.open", side_effect=OSError("denied")):
                result = _validate_and_create(path, "notes.txt")
        self.assertIsInstance(result, AttachmentError)
        self.assertIn("cannot read file", result.message)

    def test_text_file_too_large_reported(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._huge_file(d, "big.txt")
            result = _validate_and_create(path, "big.txt")
        self.assertIsInstance(result, AttachmentError)
        self.assertIn("file too large", result.message)

    def test_unknown_extension_read_as_text(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._file(d, "script.zzz", b"plain content\n")
            result = _validate_and_create(path, "script.zzz")
        self.assertIsInstance(result, ParsedAttachment)
        self.assertIn("plain content", result.part.text)

    def test_unknown_extension_too_large_reported(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._huge_file(d, "blob.zzz")
            result = _validate_and_create(path, "blob.zzz")
        self.assertIsInstance(result, AttachmentError)
        self.assertIn("file too large", result.message)

    def test_unknown_extension_read_failure_reported(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._file(d, "mystery.zzz")
            with mock.patch("builtins.open", side_effect=OSError("denied")):
                result = _validate_and_create(path, "mystery.zzz")
        self.assertIsInstance(result, AttachmentError)
        self.assertIn("cannot read file", result.message)


if __name__ == "__main__":
    unittest.main()
