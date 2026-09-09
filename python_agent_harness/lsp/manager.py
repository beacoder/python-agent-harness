"""Workspace/server management for the built-in LSP tool."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import threading
from pathlib import Path

from .client import LSPClient, LSPError

# command, language id. Users can override/add servers with
# PYTHON_AGENT_HARNESS_LSP_SERVERS as a JSON object keyed by language/extension.
DEFAULT_SERVERS: dict[str, tuple[list[str], str]] = {
    ".py": (["pyright-langserver", "--stdio"], "python"),
    ".pyi": (["pyright-langserver", "--stdio"], "python"),
    ".js": (["typescript-language-server", "--stdio"], "javascript"),
    ".jsx": (["typescript-language-server", "--stdio"], "javascriptreact"),
    ".ts": (["typescript-language-server", "--stdio"], "typescript"),
    ".tsx": (["typescript-language-server", "--stdio"], "typescriptreact"),
    ".c": (["clangd"], "c"),
    ".h": (["clangd"], "c"),
    ".cc": (["clangd"], "cpp"),
    ".cpp": (["clangd"], "cpp"),
    ".cxx": (["clangd"], "cpp"),
    ".hpp": (["clangd"], "cpp"),
    ".rs": (["rust-analyzer"], "rust"),
    ".go": (["gopls"], "go"),
}

_SERVERS: dict[str, LSPClient] = {}
_LOCK = threading.RLock()


def _load_server_config() -> dict[str, tuple[list[str], str]]:
    result = dict(DEFAULT_SERVERS)
    raw = os.environ.get("PYTHON_AGENT_HARNESS_LSP_SERVERS")
    if not raw:
        return result
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return result
    if not isinstance(data, dict):
        return result
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(value.get("command"), list):
            command = [str(x) for x in value["command"]]
            language_id = str(value.get("language_id", key.lstrip(".")))
            result[str(key)] = (command, language_id)
    return result


def _find_root(path: str, project_dir: str) -> str:
    p = Path(path).resolve()
    current = p.parent if p.is_file() else p
    project = Path(project_dir).resolve()
    markers = {
        ".git",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "compile_commands.json",
    }
    while True:
        if any((current / marker).exists() for marker in markers):
            return str(current)
        if current == project or current.parent == current:
            return str(project)
        current = current.parent


def _server_for(path: str) -> tuple[str, list[str], str] | None:
    ext = Path(path).suffix.lower()
    config = _load_server_config()
    spec = config.get(ext)
    if spec is None:
        return None
    command, language_id = spec
    if not command or shutil.which(command[0]) is None:
        return None
    return ext, command, language_id


def get_client(path: str, project_dir: str) -> tuple[LSPClient, str]:
    spec = _server_for(path)
    if spec is None:
        raise LSPError(
            f"No LSP server configured or installed for {Path(path).suffix or 'this'} file type."
        )
    ext, command, language_id = spec
    root = _find_root(path, project_dir)
    key = os.path.normcase(os.path.realpath(root)) + "\0" + "\0".join(command)
    with _LOCK:
        client = _SERVERS.get(key)
        if client is None or not client.alive:
            if client is not None:
                client.close()
            client = LSPClient(command, root, language_id)
            client.start()
            _SERVERS[key] = client
        return client, key


def shutdown_all() -> None:
    with _LOCK:
        clients = list(_SERVERS.values())
        _SERVERS.clear()
    for client in clients:
        client.close()


atexit.register(shutdown_all)
