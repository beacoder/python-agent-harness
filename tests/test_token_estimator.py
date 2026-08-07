import unittest

from python_agent_harness.token_estimator import (
    TokenCalibrator,
    context_window_for,
    estimate_payload_tokens,
    estimate_tokens,
    is_cjk_char,
)


class TestTokenizer(unittest.TestCase):
    def test_latin_estimate(self):
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertEqual(estimate_tokens("abcdefgh"), 2)

    def test_cjk_estimate(self):
        self.assertTrue(is_cjk_char("中"))
        self.assertEqual(estimate_tokens("中文"), 1)
        # mixed: 8 latin (2 tok) + 2 cjk (1 tok)
        self.assertEqual(estimate_tokens("abcdefgh中文"), 3)

    def test_context_window_matching(self):
        self.assertEqual(context_window_for("deepseek-v4"), 1_000_000)
        self.assertEqual(context_window_for("gpt-5-mini"), 128000)
        self.assertEqual(context_window_for("claude-sonnet"), 200000)
        self.assertEqual(context_window_for("unknown-model"), 32768)

    def test_calibrator(self):
        c = TokenCalibrator()
        c.last_raw_estimate = 1000
        c.update(2000)
        self.assertAlmostEqual(c.factor, 2.0)
        self.assertEqual(c.calibrate(1000), 2000)
        # clamping
        c.update(10000)
        self.assertEqual(c.factor, 3.0)
        c.update(1)
        self.assertEqual(c.factor, 0.5)
        # no raw estimate -> no update
        c2 = TokenCalibrator()
        c2.update(500)
        self.assertEqual(c2.factor, 1.0)

    def test_payload_tokens(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "tool_calls": [
                {"function": {"name": "read", "arguments": '{"file_path": "a.py"}'}}
            ]},
        ]
        tools = [{"type": "function", "function": {"name": "read"}}]
        self.assertGreater(estimate_payload_tokens("system", msgs, tools), 5)


if __name__ == "__main__":
    unittest.main()
