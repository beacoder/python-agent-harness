"""Workspace/server management for the built-in LSP tool."""

from __future__ import annotations

import atexit
import os
import shutil
import threading
from pathlib import Path

from .. import config
from .client import LSPClient, LSPError

# command, language id. Users can override/add servers via the config
# file's ``lsp.servers`` object (see config.load_lsp_config), keyed by
# file extension; those entries layer on top of this table.
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


def _load_server_config(
    config_path: str | os.PathLike | None = None,
) -> dict[str, tuple[list[str], str]]:
    """Merge the built-in DEFAULT_SERVERS with the config file's overrides.

    Config-file entries (``lsp.servers``, parsed into an ``LSPConfig``)
    win over the built-ins for the same extension. A missing/empty
    section leaves the built-ins intact.
    """
    result = dict(DEFAULT_SERVERS)
    for ext, server in config.load_lsp_config(config_path).servers.items():
        result[ext] = (server.command, server.language_id)
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


def _server_for(
    path: str, config_path: str | os.PathLike | None = None
) -> tuple[str, list[str], str] | None:
    ext = Path(path).suffix.lower()
    config_table = _load_server_config(config_path)
    spec = config_table.get(ext)
    if spec is None:
        return None
    command, language_id = spec
    if not command or shutil.which(command[0]) is None:
        return None
    return ext, command, language_id


def get_client(
    path: str, project_dir: str, config_path: str | os.PathLike | None = None
) -> tuple[LSPClient, str]:
    spec = _server_for(path, config_path)
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
