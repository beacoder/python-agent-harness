"""Configuration-file tests (LLM settings via TOML, no env vars required)."""

import os
import tempfile
import unittest
from pathlib import Path

from python_agent_harness import config

ENV_KEYS = [
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "PYTHON_AGENT_HARNESS_CONFIG",
    "OPENAI_SUBAGENT_BASE_URL",
    "OPENAI_SUBAGENT_API_KEY",
    "OPENAI_SUBAGENT_MODEL",
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
        self.assertIsNone(settings["reasoning_effort"])
        self.assertIs(settings["stream"], True)  # streaming is the default

    def test_file_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"llm": {'
                '"base_url": "https://api.example.com/v1", '
                '"api_key": "sk-test", '
                '"model": "custom-model", '
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
            self.assertEqual(settings["temperature"], 0.2)
            self.assertEqual(settings["max_tokens"], 2048)
            self.assertEqual(settings["timeout"], 30.0)
            self.assertEqual(settings["reasoning_effort"], "high")

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
            self.assertIs(settings["stream"], True)

    def test_file_can_disable_streaming(self):
        """stream: false in the config file switches to non-streaming."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {"stream": false}}', encoding="utf-8")
            settings = config.load_llm_config(p)
            self.assertIs(settings["stream"], False)
            self.assertEqual(settings["model"], config.DEFAULT_LLM["model"])

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

    def test_config_path_defaults_to_global(self):
        """No explicit path and no env var -> the default config file."""
        self.assertEqual(config._config_path(), config.CONFIG_FILE)

    def test_config_path_expands_user(self):
        self.assertEqual(
            config._config_path("~/custom.json"),
            Path("~/custom.json").expanduser(),
        )

    def test_llm_must_be_object(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": "not-an-object"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load_llm_config(p)

    def test_paths_bad_json_returns_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text("not {valid json", encoding="utf-8")
            self.assertEqual(config.load_paths_config(p), config.DEFAULT_PATHS)

    def test_paths_not_object_returns_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"paths": "nope"}', encoding="utf-8")
            self.assertEqual(config.load_paths_config(p), config.DEFAULT_PATHS)

    def test_paths_expanded_to_absolute(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"paths": {"context_path": "~/ctx", "skill_path": "rel/skills"}}',
                encoding="utf-8",
            )
            settings = config.load_paths_config(p)
            self.assertEqual(
                settings["context_path"],
                os.path.abspath(os.path.expanduser("~/ctx")),
            )
            self.assertEqual(settings["skill_path"], os.path.abspath("rel/skills"))

    def test_paths_empty_values_stay_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"paths": {"context_path": "   ", "skill_path": null}}', encoding="utf-8")
            settings = config.load_paths_config(p)
            self.assertIsNone(settings["context_path"])
            self.assertIsNone(settings["skill_path"])

    def test_template_contains_llm(self):
        self.assertIn('"llm"', config.CONFIG_TEMPLATE)
        self.assertIn("base_url", config.CONFIG_TEMPLATE)
        self.assertIn("reasoning_effort", config.CONFIG_TEMPLATE)
        self.assertIn('"stream"', config.CONFIG_TEMPLATE)
        self.assertIn('"subagent_llm"', config.CONFIG_TEMPLATE)
        self.assertIn('"context_window"', config.CONFIG_TEMPLATE)


class TestContextWindowConfig(unittest.TestCase):
    """Context-window resolution: ``llm.context_window`` in the config
    file, then built-in CONTEXT_WINDOWS table, then default."""

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

    def test_llm_context_window_loaded(self):
        """llm.context_window in the config file is read by load_llm_config."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {"context_window": 256000}}', encoding="utf-8")
            settings = config.load_llm_config(p)
            self.assertEqual(settings["context_window"], 256000)

    def test_llm_context_window_none_by_default(self):
        """When unset, context_window defaults to None (resolve at runtime)."""
        settings = config.load_llm_config("/no/such/file.json")
        self.assertIsNone(settings["context_window"])

    def test_context_window_bool_rejected(self):
        """True is an int subclass but not a valid context window."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {"context_window": true}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load_llm_config(p)

    def test_context_window_zero_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {"context_window": 0}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load_llm_config(p)

    def test_context_window_negative_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {"context_window": -1}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load_llm_config(p)

    def test_context_window_string_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {"context_window": "big"}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load_llm_config(p)

    def test_get_context_window_llm_setting_wins(self):
        """llm.context_window beats the built-in table."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {"context_window": 999999}}', encoding="utf-8")
            self.assertEqual(config.get_context_window_for_model("gpt-5-mini", p), 999999)

    def test_get_context_window_falls_to_table(self):
        """No llm.context_window -> built-in CONTEXT_WINDOWS table."""
        self.assertEqual(
            config.get_context_window_for_model("deepseek-v4", "/no/such/file.json"),
            1_000_000,
        )
        self.assertEqual(
            config.get_context_window_for_model("gpt-5-mini", "/no/such/file.json"),
            128_000,
        )

    def test_get_context_window_falls_to_default(self):
        """Unknown model with no config -> DEFAULT_CONTEXT_WINDOW."""
        self.assertEqual(
            config.get_context_window_for_model("totally-unknown", "/no/such/file.json"),
            config.DEFAULT_CONTEXT_WINDOW,
        )

    def test_get_context_window_case_insensitive(self):
        """Built-in table matching is case-insensitive."""
        self.assertEqual(
            config.get_context_window_for_model("DeepSeek-V4", "/no/such/file.json"),
            1_000_000,
        )

    def test_get_context_window_provider_prefixed_name(self):
        """Provider-prefixed model names (e.g. "ZhipuAI/GLM-5.2") match
        their table entry via case-insensitive substring match."""
        self.assertEqual(
            config.get_context_window_for_model("ZhipuAI/GLM-5.2", "/no/such/config.json"),
            1_000_000,
        )
        self.assertEqual(
            config.get_context_window_for_model("zhipuai/glm-5.2", "/no/such/config.json"),
            1_000_000,
        )
        self.assertEqual(
            config.get_context_window_for_model("glm-5.2", "/no/such/config.json"),
            1_000_000,
        )
        # a different minor version must NOT hit the glm-5.2 entry
        self.assertEqual(
            config.get_context_window_for_model("ZhipuAI/GLM-5.3", "/no/such/config.json"),
            config.DEFAULT_CONTEXT_WINDOW,
        )
        self.assertEqual(
            config.get_context_window_for_model(
                "DeepSeek/deepseek-v4-flash", "/no/such/config.json"
            ),
            1_000_000,
        )

    def test_resolve_context_window_from_settings(self):
        """resolve_context_window uses settings['context_window'] first."""
        settings = {"model": "gpt-5-mini", "context_window": 500_000}
        self.assertEqual(config.resolve_context_window(settings), 500_000)

    def test_resolve_context_window_falls_to_table(self):
        """resolve_context_window falls to built-in table when context_window is None."""
        settings = {"model": "deepseek-v4", "context_window": None}
        self.assertEqual(config.resolve_context_window(settings), 1_000_000)

    def test_resolve_context_window_falls_to_default(self):
        """resolve_context_window uses DEFAULT_CONTEXT_WINDOW for unknown models."""
        settings = {"model": "unknown", "context_window": None}
        self.assertEqual(config.resolve_context_window(settings), config.DEFAULT_CONTEXT_WINDOW)

    def test_context_window_inherits_in_subagent(self):
        """context_window set in llm inherits to subagent_llm when unset."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"llm": {"model": "m", "context_window": 500000}}',
                encoding="utf-8",
            )
            main = config.load_llm_config(p)
            sub = config.load_subagent_llm_config(p, main=main)
        self.assertEqual(sub["context_window"], 500000)

    def test_context_window_subagent_override(self):
        """subagent_llm.context_window overrides the inherited value."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"llm": {"model": "m", "context_window": 500000},'
                ' "subagent_llm": {"context_window": 200000}}',
                encoding="utf-8",
            )
            main = config.load_llm_config(p)
            sub = config.load_subagent_llm_config(p, main=main)
        self.assertEqual(sub["context_window"], 200000)

    def test_context_window_profile_override(self):
        """A models profile's context_window overrides the llm value
        when referenced via subagent_llm.profile."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"llm": {"model": "m", "context_window": 500000},'
                ' "models": {"cheap": {"model": "cheap-m", "context_window": 300000}},'
                ' "subagent_llm": {"profile": "cheap"}}',
                encoding="utf-8",
            )
            main = config.load_llm_config(p)
            sub = config.load_subagent_llm_config(p, main=main)
        self.assertEqual(sub["context_window"], 300000)

    def test_old_context_windows_section_emits_deprecation(self):
        """A config file with the old top-level context_windows section
        triggers a DeprecationWarning on load_llm_config."""
        import warnings

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"llm": {"model": "m"}, "context_windows": {"deepseek-v4": 1000000}}',
                encoding="utf-8",
            )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                config.load_llm_config(p)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertEqual(len(dep_warnings), 1)
            self.assertIn("context_windows", str(dep_warnings[0].message))
            self.assertIn("deprecated", str(dep_warnings[0].message))

    def test_no_deprecation_without_old_section(self):
        """No warning when the config file has no context_windows section."""
        import warnings

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {"model": "m", "context_window": 500000}}', encoding="utf-8")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                config.load_llm_config(p)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertEqual(len(dep_warnings), 0)


class TestSubagentLlmConfig(unittest.TestCase):
    """Sub-agent LLM settings: unset keys inherit the main settings
    (mirrors gptel-agent-harness-subagent-model)."""

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

    MAIN = {
        "base_url": "https://api.main.example/v1",
        "api_key": "sk-main",
        "model": "big-model",
        "temperature": 0.0,
        "max_tokens": None,
        "timeout": 600.0,
        "reasoning_effort": "high",
        "stream": True,
    }

    def test_inherits_main_when_unset(self):
        """No subagent_llm in the file: the resolved settings are
        exactly the main ones."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {"model": "file-model"}}', encoding="utf-8")
            main = config.load_llm_config(p)
            sub = config.load_subagent_llm_config(p, main=main)
        self.assertEqual(sub, main)

    def test_file_override_inherits_rest(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"llm": {"model": "main-model", "base_url": "https://a/v1",'
                ' "timeout": 30.0},'
                ' "subagent_llm": {"model": "cheap-model",'
                ' "base_url": "https://b/v1", "api_key": "sk-sub"}}',
                encoding="utf-8",
            )
            main = config.load_llm_config(p)
            sub = config.load_subagent_llm_config(p, main=main)
        self.assertEqual(sub["model"], "cheap-model")
        self.assertEqual(sub["base_url"], "https://b/v1")
        self.assertEqual(sub["api_key"], "sk-sub")
        # unset keys inherit the main settings
        self.assertEqual(sub["timeout"], 30.0)
        self.assertEqual(sub["temperature"], main["temperature"])
        self.assertEqual(sub["stream"], main["stream"])

    def test_env_wins_over_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"subagent_llm": {"model": "file-sub-model", "base_url": "https://file/v1"}}',
                encoding="utf-8",
            )
            main = config.load_llm_config(p)
            os.environ["OPENAI_SUBAGENT_MODEL"] = "env-sub-model"
            os.environ["OPENAI_SUBAGENT_BASE_URL"] = "https://env/v1"
            sub = config.load_subagent_llm_config(p, main=main)
        self.assertEqual(sub["model"], "env-sub-model")
        self.assertEqual(sub["base_url"], "https://env/v1")

    def test_no_main_uses_code_defaults(self):
        """Without explicit main settings, the sub-agent resolution
        starts from the code defaults (like load_llm_config)."""
        sub = config.load_subagent_llm_config("/no/such/file.json")
        self.assertEqual(sub["model"], config.DEFAULT_LLM["model"])
        self.assertEqual(sub["base_url"], config.DEFAULT_LLM["base_url"])

    def test_bad_json_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text("not {valid json", encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load_subagent_llm_config(p, main=dict(self.MAIN))

    def test_subagent_llm_must_be_object(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"subagent_llm": "nope"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load_subagent_llm_config(p, main=dict(self.MAIN))

    def test_null_values_inherit(self):
        """Explicit null values in the file mean 'inherit', not 'reset'."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"subagent_llm": {"model": null, "api_key": null}}',
                encoding="utf-8",
            )
            sub = config.load_subagent_llm_config(p, main=dict(self.MAIN))
        self.assertEqual(sub["model"], self.MAIN["model"])
        self.assertEqual(sub["api_key"], self.MAIN["api_key"])

    def test_profile_applies_settings_from_models(self):
        """subagent_llm.profile reuses a named models profile: its
        settings override the main ones (and explicit subagent_llm
        keys); keys it leaves unset still inherit main."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"llm": {"model": "big-model", "base_url": "https://main/v1",'
                ' "reasoning_effort": "high", "temperature": 0.0},'
                ' "models": {"cheap": {"model": "cheap-model",'
                ' "base_url": "https://cheap/v1", "temperature": 0.7}},'
                ' "subagent_llm": {"profile": "cheap"}}',
                encoding="utf-8",
            )
            main = config.load_llm_config(p)
            sub = config.load_subagent_llm_config(p, main=main)
        self.assertEqual(sub["model"], "cheap-model")
        self.assertEqual(sub["base_url"], "https://cheap/v1")
        self.assertEqual(sub["temperature"], 0.7)
        # keys the profile does not set inherit main
        self.assertEqual(sub["reasoning_effort"], "high")

    def test_profile_wins_over_explicit_subagent_keys(self):
        """A referenced profile's settings take precedence over explicit
        subagent_llm keys (same precedence as llm vs models)."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"llm": {"model": "big-model", "base_url": "https://main/v1"},'
                ' "models": {"cheap": {"model": "cheap-model",'
                ' "base_url": "https://cheap/v1", "temperature": 0.7}},'
                ' "subagent_llm": {"profile": "cheap", "model": "explicit-model",'
                ' "temperature": 0.2}}',
                encoding="utf-8",
            )
            main = config.load_llm_config(p)
            sub = config.load_subagent_llm_config(p, main=main)
        self.assertEqual(sub["model"], "cheap-model")
        self.assertEqual(sub["base_url"], "https://cheap/v1")
        self.assertEqual(sub["temperature"], 0.7)

    def test_profile_unknown_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"models": {"a": {"model": "m"}}, "subagent_llm": {"profile": "nope"}}',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                config.load_subagent_llm_config(p, main=dict(self.MAIN))

    def test_profile_env_still_wins(self):
        """OPENAI_SUBAGENT_* env vars beat a referenced profile."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"models": {"cheap": {"model": "cheap-model",'
                ' "base_url": "https://cheap/v1"}},'
                ' "subagent_llm": {"profile": "cheap"}}',
                encoding="utf-8",
            )
            main = config.load_llm_config(p)
            os.environ["OPENAI_SUBAGENT_MODEL"] = "env-sub-model"
            sub = config.load_subagent_llm_config(p, main=main)
        self.assertEqual(sub["model"], "env-sub-model")
        self.assertEqual(sub["base_url"], "https://cheap/v1")


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

    def test_config_show_subagent_llm_line(self):
        """config output reports the effective sub-agent LLM: a summary
        line when it differs from the main LLM, an inherit note when it
        matches."""
        import io
        from contextlib import redirect_stdout

        from python_agent_harness.cli import main

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(
                '{"llm": {"model": "main-model"},'
                ' "subagent_llm": {"model": "cheap-model",'
                ' "api_key": "sk-sub"}}',
                encoding="utf-8",
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["config", "--path", str(p)])
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("subagent_llm: model=cheap-model", out)
            self.assertIn("subagent_llm:", out)
            self.assertIn("api_key=****", out)

            # unset subagent_llm -> "(inherits main)"
            p.write_text('{"llm": {"model": "main-model"}}', encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["config", "--path", str(p)])
            self.assertEqual(rc, 0)
            self.assertIn("subagent_llm: (inherits main)", buf.getvalue())

    def test_config_init_refuses_overwrite(self):
        from python_agent_harness.cli import main

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text('{"llm": {}}', encoding="utf-8")
            rc = main(["config", "--init", "--path", str(p)])
            self.assertEqual(rc, 1)
            rc = main(["config", "--init", "--force", "--path", str(p)])
            self.assertEqual(rc, 0)

    def test_config_show_missing_file_message(self):
        """config with a nonexistent path prints a hint, falls back to
        defaults, and still exits 0."""
        import io
        from contextlib import redirect_stdout

        from python_agent_harness.cli import main

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "missing.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["config", "--path", str(p)])
            self.assertEqual(rc, 0)
            self.assertIn("file does not exist yet", buf.getvalue())
            self.assertIn("gpt-5-mini", buf.getvalue())  # default model shown

    def test_config_flag_both_positions(self):
        """--config must work both before and after the subcommand."""
        from python_agent_harness.cli import build_parser

        parser = build_parser()
        before = parser.parse_args(["--config", "/x.json", "run", "/tmp"])
        self.assertEqual(before.config, "/x.json")
        after = parser.parse_args(["run", "/tmp", "--config", "/x.json"])
        self.assertEqual(after.config, "/x.json")
        plain = parser.parse_args(["config"])
        self.assertIsNone(plain.config)

    def test_no_stream_flag(self):
        """--no-stream must be accepted on run and default to off."""
        from python_agent_harness.cli import build_parser

        parser = build_parser()
        off = parser.parse_args(["run", "/tmp", "--no-stream"])
        self.assertTrue(off.no_stream)
        on = parser.parse_args(["run", "/tmp"])
        self.assertFalse(on.no_stream)


if __name__ == "__main__":
    unittest.main()
