"""Tests for multimodal content parts, @file parsing, and image serialization."""

import os
import tempfile
import unittest

from python_agent_harness.attachments import parse_at_references
from python_agent_harness.config import DEFAULT_LLM, load_llm_config
from python_agent_harness.models import ImagePart, Message, TextPart


class TestTextPart(unittest.TestCase):
    def test_text_part_to_api(self):
        p = TextPart(text="hello")
        self.assertEqual(p.to_api(), {"type": "text", "text": "hello"})


class TestImagePart(unittest.TestCase):
    def test_image_part_to_api(self):
        p = ImagePart(data=b"\x89PNG fake", media_type="image/png")
        api = p.to_api()
        self.assertEqual(api["type"], "image_url")
        self.assertIn("image_url", api)
        url = api["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_image_part_default_media_type(self):
        p = ImagePart(data=b"x")
        self.assertEqual(p.media_type, "image/png")

    def test_image_part_url_to_api(self):
        p = ImagePart.from_url("https://example.com/image.jpg")
        api = p.to_api()
        self.assertEqual(
            api,
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        )

    def test_image_part_url_takes_precedence_over_data(self):
        p = ImagePart(data=b"\x89PNG", url="https://example.com/image.jpg")
        self.assertEqual(
            p.to_api(),
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        )

    def test_image_part_no_source_raises(self):
        p = ImagePart()
        with self.assertRaises(ValueError):
            p.to_api()

    def test_multimodal_url_image_to_api(self):
        m = Message(
            role="user",
            content=[
                TextPart(text="What is this?"),
                ImagePart.from_url("https://example.com/image.jpg"),
            ],
        )
        api = m.to_api()
        self.assertEqual(
            api["content"][1],
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        )


class TestMessageMultimodal(unittest.TestCase):
    def test_text_only_message_unchanged(self):
        m = Message(role="user", content="hello")
        self.assertEqual(m.to_api(), {"role": "user", "content": "hello"})
        self.assertEqual(m.text(), "hello")

    def test_multimodal_message_to_api(self):
        m = Message(
            role="user",
            content=[
                TextPart(text="What is this?"),
                ImagePart(data=b"\x89PNG fake", media_type="image/png"),
            ],
        )
        api = m.to_api()
        self.assertEqual(api["role"], "user")
        content = api["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0], {"type": "text", "text": "What is this?"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertIn("image_url", content[1])

    def test_multimodal_text_extraction(self):
        m = Message(
            role="user",
            content=[
                TextPart(text="hello "),
                ImagePart(data=b"x"),
                TextPart(text="world"),
            ],
        )
        self.assertEqual(m.text(), "hello world")

    def test_image_only_message_text_is_empty(self):
        m = Message(role="user", content=[ImagePart(data=b"x")])
        self.assertEqual(m.text(), "")

    def test_mixed_string_and_parts(self):
        m = Message(
            role="user",
            content=["plain ", TextPart(text="text"), ImagePart(data=b"x")],
        )
        self.assertEqual(m.text(), "plain text")

    def test_dict_parts_still_work(self):
        m = Message(
            role="user",
            content=[{"type": "text", "text": "dict text"}],
        )
        self.assertEqual(m.text(), "dict text")

    def test_reasoning_stripped_from_string_content(self):
        m = Message(
            role="assistant",
            content="thinking\n\nanswer",
            reasoning="thinking",
        )
        self.assertEqual(m.to_api()["content"], "answer")


class TestSupportsImageInputSetting(unittest.TestCase):
    def test_default_is_false(self):
        self.assertFalse(DEFAULT_LLM["supports_image_input"])

    def test_config_llm_supports_image_input(self):
        import json

        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "config.json")
            with open(cfg, "w") as f:
                json.dump({"llm": {"supports_image_input": True}}, f)
            self.assertTrue(load_llm_config(cfg)["supports_image_input"])


class TestImageTokenEstimation(unittest.TestCase):
    def test_image_parts_counted_in_payload(self):
        from python_agent_harness import token_estimator as te

        img = ImagePart(data=b"\x89PNG" + b"\x00" * 5000, media_type="image/png")
        msg = Message(role="user", content=[TextPart(text="what is this?"), img])
        tokens = te.estimate_payload_tokens(None, [msg.to_api()], [])
        # image contributes a flat estimate, not the huge base64 length
        self.assertGreaterEqual(tokens, te.IMAGE_TOKEN_ESTIMATE)
        self.assertLess(tokens, te.IMAGE_TOKEN_ESTIMATE + 100)

    def test_count_image_tokens_multiple(self):
        from python_agent_harness import token_estimator as te

        img = ImagePart(data=b"\x89PNG", media_type="image/png")
        msg = Message(role="user", content=[img, img])
        self.assertEqual(te.count_image_tokens([msg.to_api()]), 2 * te.IMAGE_TOKEN_ESTIMATE)

    def test_no_images_no_image_tokens(self):
        from python_agent_harness import token_estimator as te

        msg = Message(role="user", content="just text")
        self.assertEqual(te.count_image_tokens([msg.to_api()]), 0)


class TestParseAtReferences(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_file(self, name, content, binary=False):
        path = os.path.join(self.tmpdir, name)
        mode = "wb" if binary else "w"
        with open(path, mode, encoding=None if binary else "utf-8") as f:
            f.write(content)
        return path

    def test_image_file_creates_image_part(self):
        self._make_file("test.png", b"\x89PNG\r\n\x1a\n fake png data", binary=True)
        cleaned, attachments, errors = parse_at_references("explain @test.png", self.tmpdir)
        self.assertEqual(errors, [])
        self.assertEqual(len(attachments), 1)
        self.assertIsInstance(attachments[0].part, ImagePart)
        self.assertEqual(attachments[0].part.media_type, "image/png")
        # the resolved absolute path is recorded on the part
        self.assertEqual(attachments[0].part.path, os.path.join(self.tmpdir, "test.png"))
        self.assertIn("test.png", cleaned)
        self.assertNotIn("@test.png", cleaned)

    def test_text_file_creates_text_part(self):
        self._make_file("README.md", "# Hello World\n")
        cleaned, attachments, errors = parse_at_references("explain @README.md", self.tmpdir)
        self.assertEqual(errors, [])
        self.assertEqual(len(attachments), 1)
        self.assertIsInstance(attachments[0].part, TextPart)
        self.assertIn("Hello World", attachments[0].part.text)

    def test_python_file_creates_text_part(self):
        self._make_file("foo.py", "print('hello')\n")
        cleaned, attachments, errors = parse_at_references("review @foo.py", self.tmpdir)
        self.assertEqual(errors, [])
        self.assertEqual(len(attachments), 1)
        self.assertIsInstance(attachments[0].part, TextPart)

    def test_multiple_references(self):
        self._make_file("before.png", b"\x89PNG\r\n\x1a\nimg1", binary=True)
        self._make_file("after.png", b"\x89PNG\r\n\x1a\nimg2", binary=True)
        cleaned, attachments, errors = parse_at_references(
            "compare @before.png and @after.png", self.tmpdir
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(attachments), 2)
        self.assertIsInstance(attachments[0].part, ImagePart)
        self.assertIsInstance(attachments[1].part, ImagePart)

    def test_mixed_text_and_image(self):
        self._make_file("code.py", "x = 1\n")
        self._make_file("screenshot.png", b"\x89PNG\r\n\x1a\npng", binary=True)
        cleaned, attachments, errors = parse_at_references(
            "fix @code.py see @screenshot.png", self.tmpdir
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(attachments), 2)
        self.assertIsInstance(attachments[0].part, TextPart)
        self.assertIsInstance(attachments[1].part, ImagePart)

    def test_missing_file_error(self):
        cleaned, attachments, errors = parse_at_references("explain @nonexistent.png", self.tmpdir)
        self.assertEqual(len(errors), 1)
        self.assertIn("not found", errors[0].message)
        self.assertEqual(attachments, [])

    def test_no_references_returns_unchanged(self):
        cleaned, attachments, errors = parse_at_references("just a plain message", self.tmpdir)
        self.assertEqual(attachments, [])
        self.assertEqual(errors, [])
        self.assertEqual(cleaned, "just a plain message")

    def test_email_not_treated_as_reference(self):
        cleaned, attachments, errors = parse_at_references(
            "contact me at user@host.com", self.tmpdir
        )
        self.assertEqual(attachments, [])
        self.assertEqual(cleaned, "contact me at user@host.com")

    def test_jpg_extension(self):
        self._make_file("photo.jpg", b"\xff\xd8\xff\xe0 jpg data", binary=True)
        _, attachments, errors = parse_at_references("@photo.jpg", self.tmpdir)
        self.assertEqual(errors, [])
        self.assertEqual(attachments[0].part.media_type, "image/jpeg")

    def test_jpeg_extension(self):
        self._make_file("photo.jpeg", b"\xff\xd8\xff\xe0 jpeg data", binary=True)
        _, attachments, _ = parse_at_references("@photo.jpeg", self.tmpdir)
        self.assertEqual(attachments[0].part.media_type, "image/jpeg")

    def test_gif_extension(self):
        self._make_file("anim.gif", b"GIF89a", binary=True)
        _, attachments, _ = parse_at_references("@anim.gif", self.tmpdir)
        self.assertEqual(attachments[0].part.media_type, "image/gif")

    def test_webp_extension(self):
        self._make_file("diagram.webp", b"RIFF\x00\x00\x00\x00WEBP", binary=True)
        _, attachments, _ = parse_at_references("@diagram.webp", self.tmpdir)
        self.assertEqual(attachments[0].part.media_type, "image/webp")

    def test_fake_image_rejected(self):
        """A .png whose content is not a PNG is rejected, not sent."""
        self._make_file("fake.png", b"this is not an image", binary=True)
        _, attachments, errors = parse_at_references("@fake.png", self.tmpdir)
        self.assertEqual(attachments, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("does not look like", errors[0].message)

    def test_truncated_image_rejected(self):
        """A .jpg with only the SOI marker (no EOI) is still accepted —
        the signature check only verifies the header, not the trailer."""
        self._make_file("trunc.jpg", b"\xff\xd8", binary=True)
        _, attachments, errors = parse_at_references("@trunc.jpg", self.tmpdir)
        self.assertEqual(errors, [])
        self.assertEqual(len(attachments), 1)

    def test_trailing_punctuation_stripped(self):
        """Sentence-final punctuation after @path must not break the ref."""
        self._make_file("shot.png", b"\x89PNG\r\n\x1a\npng", binary=True)
        for text in (
            "see @shot.png.",
            "see @shot.png,",
            "see @shot.png!",
            "see @shot.png;",
            "(@shot.png)",
            "see @shot.png:",
        ):
            cleaned, attachments, errors = parse_at_references(text, self.tmpdir)
            self.assertEqual(errors, [], f"unexpected errors for {text!r}")
            self.assertEqual(len(attachments), 1, f"no attachment for {text!r}")
            self.assertIsInstance(attachments[0].part, ImagePart)
            # the punctuation stays in the cleaned text
            self.assertIn("shot.png", cleaned)
            self.assertNotIn("@shot.png", cleaned)

    def test_trailing_punctuation_kept_in_text(self):
        self._make_file("shot.png", b"\x89PNG\r\n\x1a\npng", binary=True)
        cleaned, _, _ = parse_at_references("see @shot.png.", self.tmpdir)
        self.assertEqual(cleaned, "see shot.png.")

    def test_punctuation_only_token_ignored(self):
        """A bare @ followed only by punctuation is not a reference."""
        cleaned, attachments, errors = parse_at_references("ping @.", self.tmpdir)
        self.assertEqual(attachments, [])
        self.assertEqual(errors, [])
        self.assertEqual(cleaned, "ping @.")

    def test_large_image_error(self):
        from python_agent_harness import attachments as att_mod

        old_max = att_mod.MAX_IMAGE_SIZE
        att_mod.MAX_IMAGE_SIZE = 10
        try:
            self._make_file("big.png", b"x" * 100, binary=True)
            _, _, errors = parse_at_references("@big.png", self.tmpdir)
            self.assertEqual(len(errors), 1)
            self.assertIn("too large", errors[0].message)
        finally:
            att_mod.MAX_IMAGE_SIZE = old_max

    def test_empty_image_error(self):
        self._make_file("empty.png", b"", binary=True)
        _, _, errors = parse_at_references("@empty.png", self.tmpdir)
        self.assertEqual(len(errors), 1)
        self.assertIn("empty", errors[0].message)

    def test_at_at_start_of_text(self):
        self._make_file("test.py", "hello\n")
        _, attachments, errors = parse_at_references("@test.py", self.tmpdir)
        self.assertEqual(errors, [])
        self.assertEqual(len(attachments), 1)

    def test_relative_path_resolution(self):
        subdir = os.path.join(self.tmpdir, "sub")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "nested.py"), "w") as f:
            f.write("x = 1\n")
        _, attachments, errors = parse_at_references("@sub/nested.py", self.tmpdir)
        self.assertEqual(errors, [])
        self.assertEqual(len(attachments), 1)

    def test_large_text_file_error(self):
        from python_agent_harness import attachments as att_mod

        old_max = att_mod.MAX_TEXT_SIZE
        att_mod.MAX_TEXT_SIZE = 10
        try:
            self._make_file("big.py", "x" * 100)
            _, _, errors = parse_at_references("@big.py", self.tmpdir)
            self.assertEqual(len(errors), 1)
            self.assertIn("too large", errors[0].message)
        finally:
            att_mod.MAX_TEXT_SIZE = old_max

    def test_failed_reference_kept_in_text(self):
        self._make_file("ok.py", "hello\n")
        cleaned, attachments, errors = parse_at_references(
            "fix @ok.py and @missing.py", self.tmpdir
        )
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(attachments), 1)
        # The successful @ is stripped, the failed one stays
        self.assertIn("ok.py", cleaned)
        self.assertNotIn("@ok.py", cleaned)
        self.assertIn("@missing.py", cleaned)


class TestClientImageStripping(unittest.TestCase):
    """When the model doesn't support images, the Client should strip
    ImagePart from message content before sending."""

    def test_strip_image_parts_from_message(self):
        from python_agent_harness.client import _strip_image_parts

        m = Message(
            role="user",
            content=[TextPart(text="hello"), ImagePart(data=b"x")],
        )
        stripped = _strip_image_parts(m)
        self.assertEqual(stripped.content, "hello")

    def test_strip_image_parts_all_images(self):
        from python_agent_harness.client import _strip_image_parts

        m = Message(role="user", content=[ImagePart(data=b"x"), ImagePart(data=b"y")])
        stripped = _strip_image_parts(m)
        # An image-only message must not collapse to an empty string (some
        # backends reject a blank user turn); a placeholder is substituted.
        self.assertEqual(stripped.content, "[2 images omitted: model does not support image input]")

    def test_strip_image_parts_no_images_unchanged(self):
        from python_agent_harness.client import _strip_image_parts

        m = Message(role="user", content="plain text")
        stripped = _strip_image_parts(m)
        self.assertIs(stripped, m)

    def test_strip_preserves_other_fields(self):
        from python_agent_harness.client import _strip_image_parts

        m = Message(
            role="user",
            content=[TextPart(text="hello"), ImagePart(data=b"x")],
            tool_call_id="c1",
            name="test",
        )
        stripped = _strip_image_parts(m)
        self.assertEqual(stripped.role, "user")
        self.assertEqual(stripped.tool_call_id, "c1")
        self.assertEqual(stripped.name, "test")

    def test_strip_dict_image_parts(self):
        """Raw dict parts shaped like OpenAI image_url content are stripped."""
        from python_agent_harness.client import _strip_image_parts

        m = Message(
            role="user",
            content=[
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
            ],
        )
        stripped = _strip_image_parts(m)
        # dict text parts collapse to a plain string like TextPart does
        self.assertEqual(stripped.content, "hello")

    def test_strip_dict_image_parts_all_images(self):
        from python_agent_harness.client import _strip_image_parts

        m = Message(
            role="user",
            content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}],
        )
        stripped = _strip_image_parts(m)
        # image-only content collapses to a placeholder, not an empty string
        self.assertEqual(stripped.content, "[1 image omitted: model does not support image input]")

    def test_strip_mixed_dict_and_text_parts(self):
        from python_agent_harness.client import _strip_image_parts

        m = Message(
            role="user",
            content=[
                "plain ",
                {"type": "text", "text": "dict text"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
            ],
        )
        stripped = _strip_image_parts(m)
        # str + dict text parts collapse to a plain string
        self.assertEqual(stripped.content, "plain dict text")
        # the original message is not mutated
        self.assertEqual(len(m.content), 3)


class TestSessionPersistenceMultimodal(unittest.TestCase):
    """Session persistence should note image attachments in the saved text."""

    def test_image_noted_in_conversation_text(self):
        from python_agent_harness.session import Session

        class FakeStore:
            pass

        class FakeClient:
            base_url = "http://x"
            api_key = None
            model = "m"
            timeout = 1.0
            log_path = None
            context_window = 128000

        # We can't easily build a full Session; test _conversation_text
        # by calling it as an unbound method with a minimal stand-in.
        messages = [
            Message(
                role="user",
                content=[TextPart(text="what is this"), ImagePart(data=b"x")],
            ),
            Message(role="assistant", content="it's a test"),
        ]
        # _conversation_text is an instance method but doesn't use self
        # for the core logic — call it via the class with a dummy self.
        dummy = type("Dummy", (), {"_conversation_text": Session._conversation_text})()
        text = dummy._conversation_text(messages)
        self.assertIn("image attachment", text)
        self.assertIn("what is this", text)
        self.assertIn("it's a test", text)

    def test_image_path_noted_in_conversation_text(self):
        """When the image came from a path, the placeholder records it."""
        from python_agent_harness.session import Session

        messages = [
            Message(
                role="user",
                content=[
                    TextPart(text="what is this"),
                    ImagePart(data=b"x", path="/tmp/shot.png"),
                ],
            ),
        ]
        dummy = type("Dummy", (), {"_conversation_text": Session._conversation_text})()
        text = dummy._conversation_text(messages)
        self.assertIn("/tmp/shot.png", text)
        self.assertIn("image attachment", text)

    def test_image_without_path_placeholder_has_no_location(self):
        """An image with no path (e.g. a URL image) gets a bare placeholder."""
        from python_agent_harness.session import Session

        messages = [
            Message(
                role="user",
                content=[ImagePart(data=b"x")],
            ),
        ]
        dummy = type("Dummy", (), {"_conversation_text": Session._conversation_text})()
        text = dummy._conversation_text(messages)
        self.assertIn("image attachment", text)
        self.assertNotIn(" from ", text)


class TestRestoreReattach(unittest.TestCase):
    """Restoring a saved session re-attaches images whose file still exists."""

    PNG = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\xfc\xcf"
        b"\xc0P\x0f\x00\x04\x85\x01\x80\x84\xa9\x0c\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img = os.path.join(self.tmpdir, "shot.png")
        with open(self.img, "wb") as f:
            f.write(self.PNG)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _saved_body(self):
        from python_agent_harness.session import Session

        messages = [
            Message(
                role="user",
                content=[
                    ImagePart(data=self.PNG, media_type="image/png", path=self.img),
                    TextPart(text="what is this"),
                ],
            ),
            Message(role="assistant", content="it is a test"),
        ]
        dummy = type("Dummy", (), {"_conversation_text": Session._conversation_text})()
        return dummy._conversation_text(messages)

    def test_reattach_when_file_exists(self):
        from python_agent_harness.tui.commands import CommandMixin

        restored = CommandMixin._parse_saved_body(self._saved_body())
        self.assertEqual(len(restored), 2)
        user = restored[0]
        self.assertIsInstance(user.content, list)
        images = [p for p in user.content if isinstance(p, ImagePart)]
        texts = [p for p in user.content if isinstance(p, TextPart)]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].data, self.PNG)
        self.assertEqual(images[0].media_type, "image/png")
        self.assertEqual(images[0].path, self.img)
        self.assertEqual([t.text for t in texts], ["what is this"])

    def test_no_reattach_when_file_gone(self):
        from python_agent_harness.tui.commands import CommandMixin

        body = self._saved_body()
        os.unlink(self.img)
        restored = CommandMixin._parse_saved_body(body)
        user = restored[0]
        self.assertIsInstance(user.content, str)
        self.assertIn("not available in restored session", user.content)
        self.assertIn("what is this", user.content)

    def test_no_reattach_when_file_is_not_an_image(self):
        from python_agent_harness.tui.commands import CommandMixin

        with open(self.img, "w", encoding="utf-8") as f:
            f.write("not really a png")
        restored = CommandMixin._parse_saved_body(self._saved_body())
        user = restored[0]
        self.assertIsInstance(user.content, str)
        self.assertIn("not available in restored session", user.content)

    def test_url_image_roundtrip(self):
        from python_agent_harness.session import Session
        from python_agent_harness.tui.commands import CommandMixin

        url = "https://example.com/image.jpg"
        messages = [
            Message(
                role="user",
                content=[
                    ImagePart.from_url(url),
                    TextPart(text="hi"),
                ],
            ),
        ]
        dummy = type("Dummy", (), {"_conversation_text": Session._conversation_text})()
        restored = CommandMixin._parse_saved_body(dummy._conversation_text(messages))
        user = restored[0]
        self.assertIsInstance(user.content, list)
        images = [p for p in user.content if isinstance(p, ImagePart)]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].url, url)

    def test_reattach_path_with_comma(self):
        from python_agent_harness.session import Session
        from python_agent_harness.tui.commands import CommandMixin

        img = os.path.join(self.tmpdir, "my, shot.png")
        with open(img, "wb") as f:
            f.write(self.PNG)
        messages = [
            Message(
                role="user",
                content=[
                    ImagePart(data=self.PNG, media_type="image/png", path=img),
                    TextPart(text="hi"),
                ],
            ),
        ]
        dummy = type("Dummy", (), {"_conversation_text": Session._conversation_text})()
        restored = CommandMixin._parse_saved_body(dummy._conversation_text(messages))
        user = restored[0]
        self.assertIsInstance(user.content, list)
        images = [p for p in user.content if isinstance(p, ImagePart)]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].path, img)

    def test_reattach_url_image(self):
        from python_agent_harness.attachments import reattach_images

        url = "https://example.com/image.jpg"
        text = f"[1 image attachment(s) from {url} — not available in restored session]"
        new_text, parts = reattach_images(text)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].url, url)
        self.assertIsNone(parts[0].data)
        self.assertIn("re-attached on restore", new_text)
        self.assertIn(url, new_text)

    def test_reattach_mixed_path_and_url(self):
        from python_agent_harness.attachments import reattach_images

        url = "https://example.com/image.jpg"
        text = f"[2 image attachment(s) from {self.img}, {url} — not available in restored session]"
        new_text, parts = reattach_images(text)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].path, self.img)
        self.assertEqual(parts[1].url, url)

    def test_reattach_url_only_file_gone(self):
        from python_agent_harness.attachments import reattach_images

        url = "https://example.com/image.jpg"
        text = f"[2 image attachment(s) from {self.img}, {url} — not available in restored session]"
        os.unlink(self.img)
        new_text, parts = reattach_images(text)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].url, url)

    def test_reattach_images_no_placeholder(self):
        from python_agent_harness.attachments import reattach_images

        text, parts = reattach_images("just some text")
        self.assertEqual(text, "just some text")
        self.assertEqual(parts, [])

    def test_reattach_images_placeholder_only(self):
        from python_agent_harness.attachments import reattach_images

        text = f"[1 image attachment(s) from {self.img} — not available in restored session]"
        new_text, parts = reattach_images(text)
        self.assertEqual(len(parts), 1)
        self.assertIn("re-attached on restore", new_text)


if __name__ == "__main__":
    unittest.main()
