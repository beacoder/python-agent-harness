"""Tests for the built-in LSP client and the agent-facing LSP tool."""

from __future__ import annotations

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

        p = Path("/tmp/project/main.py")
        self.assertEqual(_uri_to_path(p.as_uri()), str(p))

    def test_uri_with_spaces(self):
        from python_agent_harness.tools.lsp import _uri_to_path

        p = Path("/tmp/my project/main file.py")
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
            fake = SimpleNamespace(project_dir=tmpdir)
            with mock.patch.object(tools_lsp, "get_client", side_effect=LSPError("stop-here")):
                result = tool.run(
                    {"operation": "hover", "file_path": "sample.py", "line": 1, "character": 1},
                    ToolContext(fake),
                )
            self.assertIn("stop-here", result)


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
            result = LSP().run(args, ToolContext(SimpleNamespace(project_dir=tmpdir)))
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
                    ToolContext(SimpleNamespace(project_dir=tmpdir)),
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
                    ToolContext(SimpleNamespace(project_dir=tmpdir)),
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
                    ToolContext(SimpleNamespace(project_dir=tmpdir)),
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
                    ToolContext(SimpleNamespace(project_dir=tmpdir)),
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
                    ToolContext(SimpleNamespace(project_dir=tmpdir)),
                )
            self.assertEqual(len(opened), 1)
            self.assertEqual(opened[0][1], "def hello():\n    pass\n")


class TestManager(unittest.TestCase):
    def setUp(self):
        shutdown_all()

    def tearDown(self):
        shutdown_all()

    def test_load_server_config_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYTHON_AGENT_HARNESS_LSP_SERVERS", None)
            config = lsp_manager._load_server_config()
            self.assertIn(".py", config)
            self.assertEqual(config[".py"], (["pyright-langserver", "--stdio"], "python"))

    def test_load_server_config_override(self):
        env = {
            "PYTHON_AGENT_HARNESS_LSP_SERVERS": json.dumps(
                {".py": {"command": ["fake-ls"], "language_id": "py"}}
            )
        }
        with mock.patch.dict(os.environ, env):
            config = lsp_manager._load_server_config()
            self.assertEqual(config[".py"], (["fake-ls"], "py"))

    def test_load_server_config_bad_json_falls_back(self):
        with mock.patch.dict(os.environ, {"PYTHON_AGENT_HARNESS_LSP_SERVERS": "{not json"}):
            config = lsp_manager._load_server_config()
            self.assertIn(".py", config)

    def test_load_server_config_non_dict_falls_back(self):
        with mock.patch.dict(os.environ, {"PYTHON_AGENT_HARNESS_LSP_SERVERS": "[1,2]"}):
            config = lsp_manager._load_server_config()
            self.assertIn(".py", config)

    def test_load_server_config_entry_without_command_list_ignored(self):
        env = {"PYTHON_AGENT_HARNESS_LSP_SERVERS": json.dumps({".zz": {"command": "notalist"}})}
        with mock.patch.dict(os.environ, env):
            config = lsp_manager._load_server_config()
            self.assertNotIn(".zz", config)

    def test_server_for_unknown_extension(self):
        self.assertIsNone(lsp_manager._server_for("/tmp/x.unknownext"))

    def test_server_for_missing_binary(self):
        # .py is configured by default but pyright may not be installed;
        # missing binary => None. Force the condition to be deterministic.
        env = {
            "PYTHON_AGENT_HARNESS_LSP_SERVERS": json.dumps(
                {".py": {"command": ["definitely-not-a-real-binary-xyz"], "language_id": "python"}}
            )
        }
        with mock.patch.dict(os.environ, env):
            self.assertIsNone(lsp_manager._server_for("/tmp/x.py"))

    def test_find_root_prefers_marker_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "proj"
            sub = root / "pkg" / "sub"
            sub.mkdir(parents=True)
            (root / "pyproject.toml").write_text("", encoding="utf-8")
            (root / ".git").mkdir()
            self.assertEqual(lsp_manager._find_root(str(sub / "a.py"), str(tmpdir)), str(root))

    def test_find_root_falls_back_to_project_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "pkg"
            sub.mkdir()
            self.assertEqual(
                lsp_manager._find_root(str(sub / "a.py"), tmpdir), str(Path(tmpdir).resolve())
            )

    def test_get_client_raises_for_unhandled_type(self):
        with self.assertRaises(LSPError):
            lsp_manager.get_client("/tmp/nope.unknownext", "/tmp")

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
        env = {
            "PYTHON_AGENT_HARNESS_LSP_SERVERS": json.dumps(
                {".py": {"command": ["fake-ls"], "language_id": "python"}}
            )
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".git").mkdir()
            with (
                mock.patch.dict(os.environ, env),
                mock.patch.object(lsp_manager, "shutil") as fake_shutil,
                mock.patch.object(lsp_manager, "LSPClient", _Stub),
            ):
                fake_shutil.which.return_value = "/usr/bin/fake-ls"
                c1, _ = lsp_manager.get_client(str(f), tmpdir)
                c2, _ = lsp_manager.get_client(str(f), tmpdir)
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


class TestRegistryIntegration(unittest.TestCase):
    def test_lsp_tool_registered_in_default_registry(self):
        from python_agent_harness.tools import default_registry

        reg = default_registry()
        tool = reg.get("lsp")
        self.assertIsNotNone(tool)
        self.assertTrue(tool.is_readonly)
        self.assertEqual(tool.name, "lsp")

    def test_lsp_spec_exposed(self):
        from python_agent_harness.tools import default_registry

        specs = {s.name: s for s in default_registry().specs(["lsp"])}
        self.assertIn("lsp", specs)
        self.assertIn("operation", json.loads(json.dumps(specs["lsp"].parameters))["properties"])


if __name__ == "__main__":
    unittest.main()
