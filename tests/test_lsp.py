"""Tests for the built-in LSP client and the agent-facing LSP tool."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from python_agent_harness.lsp import LSPClient, LSPError, shutdown_all
from python_agent_harness.lsp import manager as lsp_manager
from python_agent_harness.tools import ToolContext
from python_agent_harness.tools import lsp as tools_lsp
from python_agent_harness.tools.lsp import LSP


def _make_client(**kwargs) -> LSPClient:
    defaults = dict(command=["noop"], root="/tmp", language_id="python")
    defaults.update(kwargs)
    return LSPClient(**defaults)


@contextlib.contextmanager
def _config_file(servers: dict):
    """Write a temp config file with the given lsp.servers and yield its path."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"lsp": {"servers": servers}}, f)
        path = f.name
    try:
        yield path
    finally:
        os.unlink(path)


def _fake_server(responses: dict[str, object], encoding: str = "utf-16") -> LSPClient:
    """A client whose start/request are stubbed; requests hit *responses*."""

    class _FakeServer(LSPClient):
        def __init__(self) -> None:
            super().__init__(command=["fake"], root="/tmp", language_id="python")
            self.requests: list[tuple[str, object]] = []
            self.position_encoding = encoding
            self.proc = SimpleNamespace(
                stdin=SimpleNamespace(write=lambda b: None, flush=lambda: None),
                poll=lambda: None,
            )

        def start(self) -> None:
            self._capabilities = {"positionEncoding": encoding}

        def request(self, method, params=None):
            self.requests.append((method, params))
            result = responses.get(method)
            if isinstance(result, LSPError):
                raise result
            return result

    return _FakeServer()


class TestURIToPath(unittest.TestCase):
    def test_plain_uri_roundtrip(self):
        from python_agent_harness.tools.lsp import _uri_to_path

        p = Path(tempfile.gettempdir()) / "project" / "main.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        self.assertEqual(_uri_to_path(p.as_uri()), str(p))

    def test_uri_with_spaces(self):
        from python_agent_harness.tools.lsp import _uri_to_path

        p = Path(tempfile.gettempdir()) / "my project" / "main file.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        self.assertEqual(_uri_to_path(p.as_uri()), str(p))

    def test_non_file_uri_returned_as_is(self):
        from python_agent_harness.tools.lsp import _uri_to_path

        self.assertEqual(_uri_to_path("untitled:Untitled-1"), "untitled:Untitled-1")

    @unittest.skipUnless(os.name == "nt", "windows-specific drive-letter handling")
    def test_windows_drive_uri(self):
        from python_agent_harness.tools.lsp import _uri_to_path

        self.assertEqual(_uri_to_path("file:///C:/src/main.py"), r"C:\src\main.py")


class TestJsonable(unittest.TestCase):
    def test_recursive_structures(self):
        from python_agent_harness.tools.lsp import _jsonable

        nested = {"a": [1, {"b": (2, 3)}]}
        self.assertEqual(_jsonable(nested), {"a": [1, {"b": (2, 3)}]})


class TestToolValidation(unittest.TestCase):
    def _write_file(self, tmpdir: str, content: str) -> Path:
        f = Path(tmpdir) / "sample.py"
        f.write_text(content, encoding="utf-8")
        return f

    def test_unknown_operation_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._write_file(tmpdir, "x = 1\n")
            result = LSP().run(
                {"operation": "teleport", "file_path": str(f), "line": 1, "character": 1},
                ToolContext(),
            )
            self.assertTrue(result.startswith("Error: unsupported LSP operation"))

    def test_missing_file_path_rejected(self):
        result = LSP().run({"operation": "hover", "line": 1, "character": 1}, ToolContext())
        self.assertEqual(result, "Error: file_path must not be empty")

    def test_nonexistent_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = str(Path(tmpdir) / "nope.py")
            result = LSP().run(
                {"operation": "hover", "file_path": missing, "line": 1, "character": 1},
                ToolContext(),
            )
            self.assertIn("File not found", result)

    def test_zero_line_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._write_file(tmpdir, "x = 1\n")
            result = LSP().run(
                {"operation": "hover", "file_path": str(f), "line": 0, "character": 1},
                ToolContext(),
            )
            self.assertIn("must be >= 1", result)

    def test_zero_character_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._write_file(tmpdir, "x = 1\n")
            result = LSP().run(
                {"operation": "hover", "file_path": str(f), "line": 1, "character": 0},
                ToolContext(),
            )
            self.assertIn("must be >= 1", result)

    def test_line_beyond_eof_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._write_file(tmpdir, "x = 1\n")
            result = LSP().run(
                {"operation": "hover", "file_path": str(f), "line": 99, "character": 1},
                ToolContext(),
            )
            self.assertIn("beyond end of file", result)

    def test_relative_path_resolved_against_ctx_cwd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_file(tmpdir, "x = 1\n")
            tool = LSP()
            fake = SimpleNamespace(project_dir=tmpdir, config_path=None)
            with mock.patch.object(tools_lsp, "get_client", side_effect=LSPError("stop-here")):
                result = tool.run(
                    {"operation": "hover", "file_path": "sample.py", "line": 1, "character": 1},
                    ToolContext(fake),
                )
            self.assertIn("stop-here", result)

    def test_non_integer_line_returns_clean_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._write_file(tmpdir, "x = 1\n")
            result = LSP().run(
                {"operation": "hover", "file_path": str(f), "line": "abc", "character": 1},
                ToolContext(),
            )
            self.assertEqual(result, "Error: line and character must be integers")

    def test_null_character_returns_clean_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._write_file(tmpdir, "x = 1\n")
            result = LSP().run(
                {"operation": "hover", "file_path": str(f), "line": 1, "character": None},
                ToolContext(),
            )
            self.assertEqual(result, "Error: line and character must be integers")

    def test_workspaceSymbol_ignores_invalid_position(self):
        # workspaceSymbol never uses line/character: a bad/out-of-range
        # position must NOT be rejected (it is skipped entirely).
        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._write_file(tmpdir, "x = 1\n")
            client = _fake_server({"workspace/symbol": [{"name": "x"}]})
            with mock.patch.object(tools_lsp, "get_client", return_value=(client, "k")):
                result = LSP().run(
                    {
                        "operation": "workspaceSymbol",
                        "file_path": str(f),
                        "line": 0,  # invalid for position ops, ignored here
                        "character": 999,  # beyond EOF, ignored here
                        "query": "x",
                    },
                    ToolContext(SimpleNamespace(project_dir=tmpdir, config_path=None)),
                )
            self.assertIn("x", result)
            method, params = client.requests[-1]
            self.assertEqual(method, "workspace/symbol")
            self.assertEqual(params, {"query": "x"})

    def test_workspaceSymbol_without_position_args(self):
        # line/character are no longer required for workspaceSymbol.
        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._write_file(tmpdir, "x = 1\n")
            client = _fake_server({"workspace/symbol": [{"name": "x"}]})
            with mock.patch.object(tools_lsp, "get_client", return_value=(client, "k")):
                result = LSP().run(
                    {"operation": "workspaceSymbol", "file_path": str(f), "query": "x"},
                    ToolContext(SimpleNamespace(project_dir=tmpdir, config_path=None)),
                )
            self.assertIn("x", result)

    def test_config_path_forwarded_to_get_client(self):
        # The tool must pass the session's config_path through to get_client
        # so a session started with --config reads the same lsp.servers.
        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._write_file(tmpdir, "x = 1\n")
            captured: dict = {}

            def fake_get_client(path, project_dir, config_path=None):
                captured["config_path"] = config_path
                raise LSPError("stop-here")

            fake = SimpleNamespace(project_dir=tmpdir, config_path="/tmp/custom-config.json")
            with mock.patch.object(tools_lsp, "get_client", side_effect=fake_get_client):
                LSP().run(
                    {"operation": "hover", "file_path": str(f), "line": 1, "character": 1},
                    ToolContext(fake),
                )
            self.assertEqual(captured["config_path"], "/tmp/custom-config.json")

    def test_malformed_config_valueerror_returns_clean_error(self):
        # A malformed lsp.servers surfacing on the lazy get_client call
        # (ValueError) must degrade to a clean tool error, not escape.
        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._write_file(tmpdir, "x = 1\n")
            fake = SimpleNamespace(project_dir=tmpdir, config_path=None)
            with mock.patch.object(
                tools_lsp, "get_client", side_effect=ValueError("lsp.servers.zig bad")
            ):
                result = LSP().run(
                    {"operation": "hover", "file_path": str(f), "line": 1, "character": 1},
                    ToolContext(fake),
                )
            self.assertTrue(result.startswith("Error:"))
            self.assertIn("lsp.servers.zig bad", result)


class TestToolOperations(unittest.TestCase):
    """Position math + request routing with a stubbed LSP client."""

    def _run(self, tmpdir: str, args: dict, responses: dict) -> tuple[str, LSPClient]:
        f = Path(tmpdir) / "sample.py"
        content = "def hello():\n    pass\n"
        f.write_text(content, encoding="utf-8")
        args = dict(args)
        args.setdefault("file_path", str(f))
        client = _fake_server(responses)
        with mock.patch.object(tools_lsp, "get_client", return_value=(client, "k")):
            result = LSP().run(
                args, ToolContext(SimpleNamespace(project_dir=tmpdir, config_path=None))
            )
        return result, client

    def test_hover_converts_1based_to_0based(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, client = self._run(
                tmpdir,
                {"operation": "hover", "line": 1, "character": 1},
                {"textDocument/hover": {"contents": "def hello()"}},
            )
            self.assertIn('"contents"', result)
            method, params = client.requests[-1]
            self.assertEqual(method, "textDocument/hover")
            self.assertEqual(params["position"], {"line": 0, "character": 0})

    def test_utf8_encoding_conversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text("# 中文注释\ndef f():\n    pass\n", encoding="utf-8")
            client = _fake_server({}, encoding="utf-8")
            with mock.patch.object(tools_lsp, "get_client", return_value=(client, "k")):
                LSP().run(
                    {"operation": "hover", "file_path": str(f), "line": 1, "character": 10},
                    ToolContext(SimpleNamespace(project_dir=tmpdir, config_path=None)),
                )
            # 10-1=9 python chars into the line; the comment is 5 chars + space,
            # so we reach into the CJK text: each CJK char is 3 utf-8 bytes.
            # py chars: "# 中文注释" = 1 + 1 + 4*1 = 6 chars => position 9 goes past
            # the 6-char comment to "释" area; compute precisely below.
            _, params = client.requests[-1]
            prefix = "# 中文注释"[:8]  # min(9, len(line))
            self.assertEqual(
                params["position"], {"line": 0, "character": len(prefix.encode("utf-8"))}
            )

    def test_utf32_encoding_conversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text("# 中文\ndef f():\n    pass\n", encoding="utf-8")
            client = _fake_server({}, encoding="utf-32")
            with mock.patch.object(tools_lsp, "get_client", return_value=(client, "k")):
                LSP().run(
                    {"operation": "hover", "file_path": str(f), "line": 1, "character": 4},
                    ToolContext(SimpleNamespace(project_dir=tmpdir, config_path=None)),
                )
            _, params = client.requests[-1]
            self.assertEqual(params["position"], {"line": 0, "character": 3})

    def test_utf16_encoding_conversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text("# 中文\ndef f():\n    pass\n", encoding="utf-8")
            client = _fake_server({}, encoding="utf-16")
            with mock.patch.object(tools_lsp, "get_client", return_value=(client, "k")):
                LSP().run(
                    {"operation": "hover", "file_path": str(f), "line": 1, "character": 4},
                    ToolContext(SimpleNamespace(project_dir=tmpdir, config_path=None)),
                )
            _, params = client.requests[-1]
            # "# 中" in utf-16 code units = 3
            self.assertEqual(params["position"], {"line": 0, "character": 3})

    def test_findReferences_includes_declaration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, client = self._run(
                tmpdir,
                {"operation": "findReferences", "line": 1, "character": 6},
                {"textDocument/references": [{"uri": "file:///x.py"}]},
            )
            self.assertIn("file:///x.py", result)
            method, params = client.requests[-1]
            self.assertEqual(method, "textDocument/references")
            self.assertTrue(params["context"]["includeDeclaration"])

    def test_workspaceSymbol_sends_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text("def hello():\n    pass\n", encoding="utf-8")
            client = _fake_server({"workspace/symbol": [{"name": "hello"}]})
            with mock.patch.object(tools_lsp, "get_client", return_value=(client, "k")):
                result = LSP().run(
                    {
                        "operation": "workspaceSymbol",
                        "file_path": str(f),
                        "line": 1,
                        "character": 1,
                        "query": "hello",
                    },
                    ToolContext(SimpleNamespace(project_dir=tmpdir, config_path=None)),
                )
            self.assertIn("hello", result)
            method, params = client.requests[-1]
            self.assertEqual(method, "workspace/symbol")
            self.assertEqual(params, {"query": "hello"})

    def test_incomingCalls_two_step_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            item = {"name": "hello", "uri": "file:///x.py", "range": {}}
            responses = {
                "textDocument/prepareCallHierarchy": [item],
                "callHierarchy/incomingCalls": [{"from": item}],
            }
            result, client = self._run(
                tmpdir,
                {"operation": "incomingCalls", "line": 1, "character": 6},
                responses,
            )
            self.assertIn("incomingCalls" in result.__str__() or "hello" in result, [True])
            methods = [m for m, _ in client.requests]
            self.assertEqual(
                methods, ["textDocument/prepareCallHierarchy", "callHierarchy/incomingCalls"]
            )
            _, params = client.requests[-1]
            self.assertEqual(params, {"item": item})

    def test_outgoingCalls_two_step_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            item = {"name": "hello", "uri": "file:///x.py", "range": {}}
            responses = {
                "textDocument/prepareCallHierarchy": [item],
                "callHierarchy/outgoingCalls": [{"to": item}],
            }
            _, client = self._run(
                tmpdir,
                {"operation": "outgoingCalls", "line": 1, "character": 6},
                responses,
            )
            methods = [m for m, _ in client.requests]
            self.assertEqual(methods[-1], "callHierarchy/outgoingCalls")

    def test_incomingCalls_no_prepared_item_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run(
                tmpdir,
                {"operation": "incomingCalls", "line": 1, "character": 1},
                {"textDocument/prepareCallHierarchy": []},
            )
            self.assertIn("No call hierarchy item found", result)

    def test_incomingCalls_non_list_prepared_handled(self):
        # A misbehaving server returns a truthy non-list; the tool must not
        # raise past its LSPError guard, just report no item found.
        with tempfile.TemporaryDirectory() as tmpdir:
            result, client = self._run(
                tmpdir,
                {"operation": "incomingCalls", "line": 1, "character": 1},
                {"textDocument/prepareCallHierarchy": True},
            )
            self.assertIn("No call hierarchy item found", result)
            # only the prepare step ran; no incomingCalls follow-up
            self.assertEqual([m for m, _ in client.requests], ["textDocument/prepareCallHierarchy"])

    def test_document_closed_after_use(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            closed: list[str] = []

            class _CloseTracking(_fake_server({"textDocument/hover": {"contents": "x"}}).__class__):
                def close_document(self, uri):
                    closed.append(uri)

            client = _CloseTracking()
            f = Path(tmpdir) / "sample.py"
            f.write_text("def hello():\n    pass\n", encoding="utf-8")
            with mock.patch.object(tools_lsp, "get_client", return_value=(client, "k")):
                LSP().run(
                    {"operation": "hover", "file_path": str(f), "line": 1, "character": 1},
                    ToolContext(SimpleNamespace(project_dir=tmpdir, config_path=None)),
                )
            self.assertEqual(len(closed), 1)
            self.assertEqual(closed[0], Path(f).resolve().as_uri())

    def test_document_closed_even_on_lsp_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            closed: list[str] = []

            class _CloseTracking(_fake_server({"textDocument/hover": LSPError("boom")}).__class__):
                def close_document(self, uri):
                    closed.append(uri)

            client = _CloseTracking()
            f = Path(tmpdir) / "sample.py"
            f.write_text("def hello():\n    pass\n", encoding="utf-8")
            with mock.patch.object(tools_lsp, "get_client", return_value=(client, "k")):
                result = LSP().run(
                    {"operation": "hover", "file_path": str(f), "line": 1, "character": 1},
                    ToolContext(SimpleNamespace(project_dir=tmpdir, config_path=None)),
                )
            self.assertIn("boom", result)
            self.assertEqual(len(closed), 1)  # closed despite the request raising

    def test_empty_result_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run(
                tmpdir,
                {"operation": "hover", "line": 1, "character": 1},
                {"textDocument/hover": None},
            )
            self.assertIn("No results found", result)

    def test_empty_list_result_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run(
                tmpdir,
                {"operation": "goToDefinition", "line": 1, "character": 1},
                {"textDocument/definition": []},
            )
            self.assertIn("No results found", result)

    def test_lsp_error_surfaced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run(
                tmpdir,
                {"operation": "hover", "line": 1, "character": 1},
                {"textDocument/hover": LSPError("boom")},
            )
            self.assertIn("Error:", result)
            self.assertIn("boom", result)

    def test_document_didOpen_called_with_file_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text("def hello():\n    pass\n", encoding="utf-8")
            opened: list[tuple[str, str]] = []

            class _OpenTracking(_fake_server({}).__class__):
                def open_document(self, uri, text):
                    opened.append((uri, text))

            client = _OpenTracking()
            with mock.patch.object(tools_lsp, "get_client", return_value=(client, "k")):
                LSP().run(
                    {"operation": "hover", "file_path": str(f), "line": 1, "character": 1},
                    ToolContext(SimpleNamespace(project_dir=tmpdir, config_path=None)),
                )
            self.assertEqual(len(opened), 1)
            self.assertEqual(opened[0][1], "def hello():\n    pass\n")


class TestManager(unittest.TestCase):
    def setUp(self):
        shutdown_all()

    def tearDown(self):
        shutdown_all()

    def test_load_server_config_defaults(self):
        # No config file -> only the built-in DEFAULT_SERVERS.
        config = lsp_manager._load_server_config("/no/such/config.json")
        self.assertIn(".py", config)
        self.assertEqual(config[".py"], (["pyright-langserver", "--stdio"], "python"))

    def test_load_server_config_override(self):
        with _config_file({".py": {"command": ["fake-ls"], "language_id": "py"}}) as p:
            config = lsp_manager._load_server_config(p)
            self.assertEqual(config[".py"], (["fake-ls"], "py"))
            # built-ins for other extensions remain
            self.assertIn(".ts", config)

    def test_load_server_config_new_extension_added(self):
        with _config_file({".zig": {"command": ["zls"]}}) as p:
            config = lsp_manager._load_server_config(p)
            # language_id defaults to the extension without its dot
            self.assertEqual(config[".zig"], (["zls"], "zig"))

    def test_load_server_config_bad_json_raises(self):
        # A malformed config file surfaces as an error (via _read_config).
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{not json")
            p = f.name
        try:
            with self.assertRaises(ValueError):
                lsp_manager._load_server_config(p)
        finally:
            os.unlink(p)

    def test_load_server_config_entry_without_command_list_raises(self):
        with (
            _config_file({".zz": {"command": "notalist"}}) as p,
            self.assertRaises(ValueError),
        ):
            lsp_manager._load_server_config(p)

    def test_load_server_config_non_object_servers_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"lsp": {"servers": [1, 2]}}, f)
            p = f.name
        try:
            with self.assertRaises(ValueError):
                lsp_manager._load_server_config(p)
        finally:
            os.unlink(p)

    def test_server_for_unknown_extension(self):
        self.assertIsNone(lsp_manager._server_for("/tmp/x.unknownext", "/no/such/config.json"))

    def test_server_for_missing_binary(self):
        # missing binary => None. Force the condition to be deterministic
        # via a config-file override pointing at a nonexistent binary.
        with _config_file(
            {".py": {"command": ["definitely-not-a-real-binary-xyz"], "language_id": "python"}}
        ) as p:
            self.assertIsNone(lsp_manager._server_for("/tmp/x.py", p))

    def test_find_root_prefers_marker_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "proj"
            sub = root / "pkg" / "sub"
            sub.mkdir(parents=True)
            (root / "pyproject.toml").write_text("", encoding="utf-8")
            (root / ".git").mkdir()
            # Use resolve() consistently on both sides to handle macOS /var → /private/var
            found = lsp_manager._find_root(str(sub / "a.py"), str(tmpdir))
            expected = str(root.resolve())
            self.assertEqual(found, expected)

    def test_find_root_falls_back_to_project_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "pkg"
            sub.mkdir()
            found = lsp_manager._find_root(str(sub / "a.py"), tmpdir)
            expected = str(Path(tmpdir).resolve())
            self.assertEqual(found, expected)

    def test_get_client_raises_for_unhandled_type(self):
        with self.assertRaises(LSPError):
            lsp_manager.get_client("/tmp/nope.unknownext", "/tmp", "/no/such/config.json")

    def test_get_client_reuses_live_client(self):
        calls = []

        class _Stub(LSPClient):
            def __init__(self, command, root, language_id, timeout=30.0):
                super().__init__(command, root, language_id, timeout)
                calls.append(root)

            def start(self):
                pass

            @property
            def alive(self) -> bool:
                return True

        f = Path("/tmp/x.py")
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            _config_file({".py": {"command": ["fake-ls"], "language_id": "python"}}) as cfg,
        ):
            (Path(tmpdir) / ".git").mkdir()
            with (
                mock.patch.object(lsp_manager, "shutil") as fake_shutil,
                mock.patch.object(lsp_manager, "LSPClient", _Stub),
            ):
                fake_shutil.which.return_value = "/usr/bin/fake-ls"
                c1, _ = lsp_manager.get_client(str(f), tmpdir, cfg)
                c2, _ = lsp_manager.get_client(str(f), tmpdir, cfg)
                self.assertIs(c1, c2)
                self.assertEqual(len(calls), 1)


class TestClientWireProtocol(unittest.TestCase):
    """Exercise the real _send/_read_loop with pipes instead of processes."""

    def _make_piped_client(self) -> tuple[LSPClient, object, object]:
        client = _make_client()
        client.start_if_needed = lambda: None
        client._closed = False

        server_stdin_read, client_stdin_write = os.pipe()
        client_stdout_read, server_stdout_write = os.pipe()
        client.proc = SimpleNamespace(
            stdin=os.fdopen(client_stdin_write, "wb", buffering=0),
            stdout=os.fdopen(client_stdout_read, "rb", buffering=0),
            poll=lambda: None,
        )
        reader = threading.Thread(target=client._read_loop, daemon=True)
        reader.start()
        return client, server_stdin_read, server_stdout_write

    def test_send_writes_content_length_framing(self):
        client, stdin_read, _ = self._make_piped_client()
        client._send({"jsonrpc": "2.0", "method": "ping"})
        data = os.read(stdin_read, 65536)
        header, _, payload = data.partition(b"\r\n\r\n")
        self.assertTrue(header.startswith(b"Content-Length: "))
        length = int(header.split(b":")[1].strip())
        self.assertEqual(len(payload), length)
        self.assertEqual(json.loads(payload)["method"], "ping")

    def test_send_raises_when_server_dead(self):
        client, stdin_read, _ = self._make_piped_client()
        client.proc.poll = lambda: 1
        with self.assertRaises(LSPError):
            client._send({"jsonrpc": "2.0", "method": "ping"})
        os.close(stdin_read)

    def test_roundtrip_request_response(self):
        client, stdin_read, stdout_write = self._make_piped_client()
        results: list[object] = []

        def ask():
            try:
                results.append(client.request("textDocument/hover", {"x": 1}))
            except LSPError as e:
                results.append(e)

        t = threading.Thread(target=ask)
        t.start()
        # read the request
        data = b""
        while b"\r\n\r\n" not in data:
            data += os.read(stdin_read, 65536)
        header, _, payload = data.partition(b"\r\n\r\n")
        req = json.loads(payload)
        resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"contents": "hi"}}
        body = json.dumps(resp).encode()
        os.write(stdout_write, b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        t.join(timeout=5)
        os.close(stdin_read)
        self.assertEqual(results, [{"contents": "hi"}])

    def test_request_timeout_raises_lsp_error(self):
        client = _make_client(timeout=0.05)
        client.start_if_needed = lambda: None
        with mock.patch.object(client, "_send"), self.assertRaises(LSPError):
            client.request("slow/method")

    def test_request_error_response_raises(self):
        client, stdin_read, stdout_write = self._make_piped_client()
        results: list[object] = []

        def ask():
            try:
                client.request("bad/method")
            except LSPError as e:
                results.append(str(e))

        t = threading.Thread(target=ask)
        t.start()
        data = b""
        while b"\r\n\r\n" not in data:
            data += os.read(stdin_read, 65536)
        _, _, payload = data.partition(b"\r\n\r\n")
        req = json.loads(payload)
        resp = {"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32601, "message": "nope"}}
        body = json.dumps(resp).encode()
        os.write(stdout_write, b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        t.join(timeout=5)
        os.close(stdin_read)
        self.assertEqual(results, ["LSP bad/method failed (-32601): nope"])

    def test_handle_message_diagnostics_stored(self):
        client = _make_client()
        uri = "file:///x.py"
        client._handle_message(
            {
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "diagnostics": [{"message": "m"}]},
            }
        )
        self.assertEqual(client.diagnostics(uri), [{"message": "m"}])

    def test_handle_message_responds_to_workspace_configuration(self):
        client, stdin_read, _ = self._make_piped_client()
        client._handle_message(
            {"id": 7, "method": "workspace/configuration", "params": [{"a": 1}, {"b": 2}]}
        )
        data = os.read(stdin_read, 65536)
        _, _, payload = data.partition(b"\r\n\r\n")
        resp = json.loads(payload)
        self.assertEqual(resp["id"], 7)
        self.assertEqual(resp["result"], [None, None])

    def test_handle_message_apply_edit_declined(self):
        client, stdin_read, _ = self._make_piped_client()
        client._handle_message({"id": 8, "method": "workspace/applyEdit", "params": {}})
        data = os.read(stdin_read, 65536)
        _, _, payload = data.partition(b"\r\n\r\n")
        resp = json.loads(payload)
        self.assertEqual(resp["result"], {"applied": False})

    def test_open_document_did_open_then_noop_on_same_text(self):
        sent: list[dict] = []
        client = _make_client()
        client.start_if_needed = lambda: None
        client.notify = lambda m, p=None: sent.append((m, p))
        uri = "file:///x.py"
        client.open_document(uri, "text")
        client.open_document(uri, "text")  # unchanged -> no extra notification
        methods = [m for m, _ in sent]
        self.assertEqual(methods, ["textDocument/didOpen"])

    def test_open_document_sends_didChange_on_edit(self):
        sent: list[dict] = []
        client = _make_client()
        client.start_if_needed = lambda: None
        client.notify = lambda m, p=None: sent.append((m, p))
        uri = "file:///x.py"
        client.open_document(uri, "v1")
        client.open_document(uri, "v2")
        methods = [m for m, _ in sent]
        self.assertEqual(methods, ["textDocument/didOpen", "textDocument/didChange"])
        self.assertEqual(sent[-1][1]["contentChanges"], [{"text": "v2"}])

    def test_close_document_only_when_open(self):
        sent: list[dict] = []
        client = _make_client()
        client.start_if_needed = lambda: None
        client.notify = lambda m, p=None: sent.append((m, p))
        client.proc = SimpleNamespace(
            stdin=SimpleNamespace(write=lambda b: None, flush=lambda: None),
            poll=lambda: None,
        )
        client.close_document("file:///never-opened.py")
        self.assertEqual(sent, [])
        client.open_document("file:///x.py", "t")
        client.close_document("file:///x.py")
        self.assertEqual([m for m, _ in sent][-1], "textDocument/didClose")

    def test_pending_waiters_drained_on_reader_exit(self):
        client = _make_client(timeout=5)
        client.start_if_needed = lambda: None
        waiter: queue.Queue = queue.Queue(maxsize=1)
        client._pending[99] = waiter
        # No stdout stream => the read loop returns immediately and drains.
        client.proc = SimpleNamespace(stdout=None)
        client._closed = False
        client._read_loop()
        self.assertEqual(waiter.get(timeout=5), ("error", {"message": "LSP server exited"}))

    def test_close_is_idempotent(self):
        client = _make_client()
        client.proc = None
        client.close()
        client.close()
        self.assertTrue(client._closed)

    def test_close_closes_proc_streams(self):
        # close() must release the pipe file objects so their fds are not
        # leaked (subprocess does not close them on terminate/kill).
        closed_streams: list[str] = []

        class _Stream:
            def __init__(self, tag):
                self.tag = tag

            def close(self):
                closed_streams.append(self.tag)

        client = _make_client()
        client.proc = SimpleNamespace(
            stdin=_Stream("stdin"),
            stdout=_Stream("stdout"),
            poll=lambda: 0,  # already exited: skip terminate/kill
        )
        client.close()
        self.assertEqual(sorted(closed_streams), ["stdin", "stdout"])

    def test_close_tolerates_stream_close_failure(self):
        # a stream whose close() raises must not break close() (best-effort).
        class _BadStream:
            def close(self):
                raise OSError("boom")

        client = _make_client()
        client.proc = SimpleNamespace(
            stdin=_BadStream(),
            stdout=_BadStream(),
            poll=lambda: 0,
        )
        client.close()  # must not raise
        self.assertTrue(client._closed)


class TestRegistryIntegration(unittest.TestCase):
    def test_lsp_tool_registered_in_default_registry(self):
        from python_agent_harness.tools import default_registry

        reg = default_registry()
        tool = reg.get("LSP")
        self.assertIsNotNone(tool)
        self.assertTrue(tool.is_readonly)
        self.assertEqual(tool.name, "LSP")

    def test_lsp_in_default_tools_config(self):
        from python_agent_harness.config import DEFAULT_TOOLS
        from python_agent_harness.tools import default_registry

        reg_names = set(default_registry()._tools.keys())
        for name in DEFAULT_TOOLS:
            self.assertIn(name, reg_names, f"DEFAULT_TOOLS entry {name!r} missing from registry")

    def test_lsp_spec_exposed(self):
        from python_agent_harness.tools import default_registry

        specs = {s.name: s for s in default_registry().specs(["LSP"])}
        self.assertIn("LSP", specs)
        self.assertIn("operation", json.loads(json.dumps(specs["LSP"].parameters))["properties"])

    def test_workspaceSymbol_with_empty_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text("def hello():\n    pass\n", encoding="utf-8")
            client = _fake_server({"workspace/symbol": [{"name": "hello"}]})
            with mock.patch.object(tools_lsp, "get_client", return_value=(client, "k")):
                result = LSP().run(
                    {
                        "operation": "workspaceSymbol",
                        "file_path": str(f),
                        "line": 1,
                        "character": 1,
                        "query": "",
                    },
                    ToolContext(SimpleNamespace(project_dir=tmpdir, config_path=None)),
                )
            self.assertIn("hello", result)

    def test_goToImplementation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text(
                "class Base:\n    pass\nclass Derived(Base):\n    pass\n", encoding="utf-8"
            )
            client = _fake_server(
                {"textDocument/implementation": [{"uri": "file:///x.py", "range": {}}]}
            )
            with mock.patch.object(tools_lsp, "get_client", return_value=(client, "k")):
                result = LSP().run(
                    {
                        "operation": "goToImplementation",
                        "file_path": str(f),
                        "line": 2,
                        "character": 1,
                    },
                    ToolContext(SimpleNamespace(project_dir=tmpdir, config_path=None)),
                )
            self.assertIn("file:///x.py", result)

    def test_prepareCallHierarchy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "sample.py"
            f.write_text("def hello():\n    pass\n", encoding="utf-8")
            item = {"name": "hello", "uri": "file:///x.py", "range": {}}
            client = _fake_server({"textDocument/prepareCallHierarchy": [item]})
            with mock.patch.object(tools_lsp, "get_client", return_value=(client, "k")):
                result = LSP().run(
                    {
                        "operation": "prepareCallHierarchy",
                        "file_path": str(f),
                        "line": 1,
                        "character": 1,
                    },
                    ToolContext(SimpleNamespace(project_dir=tmpdir, config_path=None)),
                )
            self.assertIn("hello", result)


class TestLSPConfigDataclass(unittest.TestCase):
    """Unit tests for the lsp/config.py dataclasses (mirrors mcp/config)."""

    def test_from_dict_empty(self):
        from python_agent_harness.lsp.config import LSPConfig

        self.assertEqual(LSPConfig.from_dict(None).servers, {})
        self.assertEqual(LSPConfig.from_dict({}).servers, {})

    def test_from_dict_fills_ext_and_language_id(self):
        from python_agent_harness.lsp.config import LSPConfig

        cfg = LSPConfig.from_dict({".zig": {"command": ["zls"]}})
        srv = cfg.servers[".zig"]
        self.assertEqual(srv.command, ["zls"])
        self.assertEqual(srv.language_id, "zig")  # defaulted from ext
        self.assertEqual(srv.ext, ".zig")

    def test_from_dict_explicit_language_id(self):
        from python_agent_harness.lsp.config import LSPConfig

        cfg = LSPConfig.from_dict({".cpp": {"command": ["clangd"], "language_id": "cpp"}})
        self.assertEqual(cfg.servers[".cpp"].language_id, "cpp")

    def test_from_dict_skips_comment_key(self):
        from python_agent_harness.lsp.config import LSPConfig

        cfg = LSPConfig.from_dict({"_comment": "x", ".zig": {"command": ["zls"]}})
        self.assertEqual(set(cfg.servers), {".zig"})

    def test_from_dict_missing_command_raises(self):
        from python_agent_harness.lsp.config import LSPConfig

        with self.assertRaises(ValueError):
            LSPConfig.from_dict({".zig": {"language_id": "zig"}})

    def test_from_dict_non_object_entry_raises(self):
        from python_agent_harness.lsp.config import LSPConfig

        with self.assertRaises(ValueError):
            LSPConfig.from_dict({".zig": ["zls"]})

    def test_compact_construction_fills_defaults(self):
        # Building LSPConfig directly (the dict key is authoritative).
        from python_agent_harness.lsp.config import LSPConfig, LSPServerConfig

        cfg = LSPConfig(servers={".cpp": LSPServerConfig(command=["clangd"])})
        self.assertEqual(cfg.servers[".cpp"].ext, ".cpp")
        self.assertEqual(cfg.servers[".cpp"].language_id, "cpp")

    def test_validate_rejects_empty_command(self):
        from python_agent_harness.lsp.config import LSPServerConfig

        with self.assertRaises(ValueError):
            LSPServerConfig(command=[], ext=".zig").validate()


if __name__ == "__main__":
    unittest.main()
