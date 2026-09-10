"""LSP server configuration for python-agent-harness.

This module defines plain configuration data classes for the built-in
LSP tool, mirroring ``mcp/config.py``.  It imports nothing from the rest
of the package (only stdlib), so it stays free of import cycles and can
be read without pulling in the LSP client/manager machinery.

Configuration model:

- ``LSPServerConfig`` describes one server: the ``command`` (server
  argv) plus a ``language_id`` (the LSP ``languageId``, e.g. ``python``,
  ``cpp``).
- ``LSPConfig`` maps a file **extension** (e.g. ``.py``, ``.cpp``) to an
  ``LSPServerConfig``.  These entries layer on top of the built-in
  ``DEFAULT_SERVERS`` table in ``lsp/manager.py``: an entry for an
  existing extension replaces its default.

The ``language_id`` defaults to the extension without its leading dot,
matching the manager's historic behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LSPServerConfig:
    """Configuration for one LSP server, keyed by file extension.

    ``ext`` is optional: when the server lives in an ``LSPConfig`` dict
    the dict key is authoritative (it fills ``ext`` on construction) and
    also seeds ``language_id`` when unset, so the compact form works::

        LSPConfig(servers={".cpp": LSPServerConfig(command=["clangd"])})
    """

    command: list[str] = field(default_factory=list)
    language_id: str = ""
    ext: str = ""

    def validate(self) -> None:
        label = self.ext or "(unnamed)"
        if not self.command:
            raise ValueError(f"LSP server {label!r}: requires a non-empty `command` list")


@dataclass
class LSPConfig:
    """The set of per-extension LSP server overrides.

    Usage (the compact form — the dict key IS the file extension)::

        config = LSPConfig(
            servers={
                ".cpp": LSPServerConfig(
                    command=["clangd", "--background-index"],
                    language_id="cpp",
                ),
            }
        )
    """

    servers: dict[str, LSPServerConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The dict key is the authoritative extension: fill in any config
        # whose ext was left unset (compact construction), and default the
        # language_id to the extension without its leading dot.
        for key, server in self.servers.items():
            if not server.ext:
                server.ext = key
            if not server.language_id:
                server.language_id = str(key).lstrip(".")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> LSPConfig:
        """Build from a plain mapping (e.g. the config file's ``lsp.servers``).

        Keys are file extensions; each value is an object with a
        ``command`` list and an optional ``language_id`` (defaults to the
        extension without its dot).  A ``_comment`` key (or any key
        starting with ``_``) is ignored so the config file can be
        annotated.

        Raises ValueError on malformed entries (missing/empty command,
        non-object entry) so config errors surface at session start, not
        mid-run.
        """
        config = cls()
        for ext, raw in (data or {}).items():
            if str(ext).startswith("_"):  # allow a "_comment" key
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"LSP server {ext!r}: expected an object")
            command = raw.get("command")
            if not isinstance(command, list) or not command:
                raise ValueError(f"LSP server {ext!r}: requires a non-empty `command` list")
            server = LSPServerConfig(
                command=[str(x) for x in command],
                language_id=str(raw.get("language_id", str(ext).lstrip("."))),
                ext=str(ext),
            )
            server.validate()
            config.servers[str(ext)] = server
        return config
