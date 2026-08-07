"""Configuration-file tests (LLM settings via TOML, no env vars required)."""

import os
import tempfile
import unittest
from pathlib import Path

from python_agent_harness import config

ENV_KEYS = [
    "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL",
    "OPENAI_BACKEND", "PYTHON_AGENT_HARNESS_CONFIG",
]


class TestConfigFile(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_missing_file_uses_defaults(self):
        settings = config.load_llm_config("/no/such/file.json")
        self.assertEqual(settings["base_url"], config.DEFAULT_LLM["base_url"])
        self.assertEqual(settings["model"], config.DEFAULT_LLM["model"])
        self.assertIsNone(settings["api_key"])

    def test_file_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"llm": {'
                '"base_url": "https://api.example.com/v1", '
                '"api_key": "sk-test", '
                '"model": "custom-model", '
                '"backend": "Example", '
                '"temperature": 0.2, '
                '"max_tokens": 2048, '
                '"timeout": 30.0, '
                '"reasoning_effort": "high"}}',
                encoding="utf-8",
            )
            settings = config.load_llm_config(p)
            self.assertEqual(settings["base_url"], "https://api.example.com/v1")
            self.assertEqual(settings["api_key"], "sk-test")
            self.assertEqual(settings["model"], "custom-model")
            self.assertEqual(settings["backend"], "Example")
            self.assertEqual(settings["temperature"], 0.2)
            self.assertEqual(settings["max_tokens"], 2048)
            self.assertEqual(settings["timeout"], 30.0)
            self.assertEqual(settings["reasoning_effort"], "high")

    def test_reasoning_effort_default_none(self):
        settings = config.load_llm_config("/no/such/file.json")
        self.assertIsNone(settings["reasoning_effort"])

    def test_reasoning_effort_unset_omitted_from_payload(self):
        from python_agent_harness.client import Client

        c = Client(base_url="http://x/v1", api_key="k", model="m")
        payload = c._payload([], stream=False)
        self.assertNotIn("reasoning_effort", payload)
        c2 = Client(base_url="http://x/v1", api_key="k", model="m")
        payload2 = c2._payload([], stream=False, reasoning_effort="medium")
        self.assertEqual(payload2["reasoning_effort"], "medium")

    def test_partial_file_keeps_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {"model": "only-model"}}', encoding="utf-8")
            settings = config.load_llm_config(p)
            self.assertEqual(settings["model"], "only-model")
            self.assertEqual(settings["base_url"], config.DEFAULT_LLM["base_url"])
            self.assertIsNone(settings["api_key"])

    def test_env_still_wins(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {"model": "file-model"}}', encoding="utf-8")
            os.environ["OPENAI_MODEL"] = "env-model"
            settings = config.load_llm_config(p)
            self.assertEqual(settings["model"], "env-model")

    def test_bad_json_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text("not {valid json", encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load_llm_config(p)

    def test_env_path_env_var(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "custom.json"
            p.write_text('{"llm": {"model": "via-env-path"}}', encoding="utf-8")
            os.environ["PYTHON_AGENT_HARNESS_CONFIG"] = str(p)
            settings = config.load_llm_config()
            self.assertEqual(settings["model"], "via-env-path")

    def test_mask_secret(self):
        self.assertEqual(config.mask_secret("sk-123"), "****")
        self.assertEqual(config.mask_secret(None), "(unset)")
        self.assertEqual(config.mask_secret(""), "(unset)")

    def test_template_contains_llm(self):
        self.assertIn('"llm"', config.CONFIG_TEMPLATE)
        self.assertIn("base_url", config.CONFIG_TEMPLATE)
        self.assertIn("reasoning_effort", config.CONFIG_TEMPLATE)


class TestConfigCli(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_config_init_and_show(self):
        from python_agent_harness.cli import main

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            rc = main(["config", "--init", "--path", str(p)])
            self.assertEqual(rc, 0)
            self.assertTrue(p.exists())
            rc = main(["config", "--path", str(p)])
            self.assertEqual(rc, 0)

    def test_config_init_refuses_overwrite(self):
        from python_agent_harness.cli import main

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {}}', encoding="utf-8")
            rc = main(["config", "--init", "--path", str(p)])
            self.assertEqual(rc, 1)
            rc = main(["config", "--init", "--force", "--path", str(p)])
            self.assertEqual(rc, 0)

    def test_config_flag_both_positions(self):
        """--config must work both before and after the subcommand."""
        from python_agent_harness.cli import build_parser

        parser = build_parser()
        before = parser.parse_args(["--config", "/x.json", "run", "/tmp"])
        self.assertEqual(before.config, "/x.json")
        after = parser.parse_args(["run", "/tmp", "--config", "/x.json"])
        self.assertEqual(after.config, "/x.json")
        plain = parser.parse_args(["sessions"])
        self.assertIsNone(plain.config)


if __name__ == "__main__":
    unittest.main()
