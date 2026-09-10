"""Agent-facing LSP code-intelligence tool.

A small OpenCode-style surface over the built-in LSP client. The model sees
1-based file/line/character coordinates; the client translates them to LSP's
0-based representation.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..lsp import LSPError, get_client
from .base import Tool, ToolContext

OPERATIONS = (
    "goToDefinition",
    "findReferences",
    "hover",
    "documentSymbol",
    "workspaceSymbol",
    "goToImplementation",
    "prepareCallHierarchy",
    "incomingCalls",
    "outgoingCalls",
)

DESCRIPTION = """Interact with Language Server Protocol (LSP) servers to get code intelligence features.

Supported operations:
- goToDefinition: Find where a symbol is defined
- findReferences: Find all references to a symbol
- hover: Get hover information (documentation, type info) for a symbol
- documentSymbol: Get all symbols (functions, classes, variables) in a document
- workspaceSymbol: List project-wide symbols matching a query string
- goToImplementation: Find implementations of an interface or abstract method
- prepareCallHierarchy: Get call hierarchy item at a position (functions/methods)
- incomingCalls: Find all functions/methods that call the function at a position
- outgoingCalls: Find all functions/methods called by the function at a position

All position-based operations use 1-based line and character numbers, as shown in editors.
workspaceSymbol uses file_path only to select the workspace/LSP server and query is optional.
"""

PARAMETERS = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": list(OPERATIONS),
            "description": "The LSP operation to perform",
        },
        "file_path": {
            "type": "string",
            "description": "The absolute or relative path to the file",
        },
        "line": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "The line number (1-based, as shown in editors). "
                "Required for all operations except workspaceSymbol."
            ),
        },
        "character": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "The character offset (1-based, as shown in editors). "
                "Required for all operations except workspaceSymbol."
            ),
        },
        "query": {
            "type": "string",
            "description": "Search query for workspaceSymbol. Empty string requests all symbols.",
        },
    },
    "required": ["operation", "file_path"],
}

# Operations that require a position (line/character).
_POSITION_OPS = frozenset(o for o in OPERATIONS if o != "workspaceSymbol")


def _uri_to_path(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        path = unquote(parsed.path)
        if os.name == "nt" and path.startswith("/") and len(path) >= 3 and path[2] == ":":
            path = path[1:]
        return os.path.normpath(path)
    return uri


def _jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _resolve_path(raw_path: str, cwd: str) -> str:
    if not os.path.isabs(raw_path):
        return os.path.realpath(os.path.abspath(os.path.join(cwd, raw_path)))
    return os.path.realpath(raw_path)


def _to_lsp_position(lines: list[str], line: int, character: int, encoding: str) -> dict[str, int]:
    source_line = lines[line - 1].rstrip("\r\n")
    py_index = min(character - 1, len(source_line))
    if encoding == "utf-8":
        lsp_character = len(source_line[:py_index].encode("utf-8"))
    elif encoding == "utf-32":
        lsp_character = py_index
    else:
        lsp_character = len(source_line[:py_index].encode("utf-16-le")) // 2
    return {"line": line - 1, "character": lsp_character}


def _format_result(result: Any, operation: str) -> str:
    if result is None:
        result = []
    if isinstance(result, (dict, list)) and not result:
        return f"No results found for {operation}"
    return json.dumps(_jsonable(result), ensure_ascii=False, indent=2)


# --- Operation handlers -------------------------------------------------
# Each handler receives (client, uri, position, args) and returns the raw
# LSP response (or a str error message for early-exit cases like empty
# call-hierarchy preparation).


def _op_definition(client, uri, position, args) -> Any:
    return client.request(
        "textDocument/definition",
        {"textDocument": {"uri": uri}, "position": position},
    )


def _op_references(client, uri, position, args) -> Any:
    return client.request(
        "textDocument/references",
        {
            "textDocument": {"uri": uri},
            "position": position,
            "context": {"includeDeclaration": True},
        },
    )


def _op_hover(client, uri, position, args) -> Any:
    return client.request(
        "textDocument/hover", {"textDocument": {"uri": uri}, "position": position}
    )


def _op_document_symbol(client, uri, position, args) -> Any:
    return client.request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})


def _op_workspace_symbol(client, uri, position, args) -> Any:
    return client.request("workspace/symbol", {"query": str(args.get("query", ""))})


def _op_implementation(client, uri, position, args) -> Any:
    return client.request(
        "textDocument/implementation",
        {"textDocument": {"uri": uri}, "position": position},
    )


def _op_prepare_call_hierarchy(client, uri, position, args) -> Any:
    return client.request(
        "textDocument/prepareCallHierarchy",
        {"textDocument": {"uri": uri}, "position": position},
    )


def _op_call_hierarchy(direction: str) -> Callable[..., Any]:
    def handler(client, uri, position, args) -> Any:
        prepared = client.request(
            "textDocument/prepareCallHierarchy",
            {"textDocument": {"uri": uri}, "position": position},
        )
        if not isinstance(prepared, list) or not prepared:
            return "No call hierarchy item found at this position"
        item = prepared[0]
        method = (
            "callHierarchy/incomingCalls"
            if direction == "incoming"
            else "callHierarchy/outgoingCalls"
        )
        return client.request(method, {"item": item})

    return handler


_DISPATCH: dict[str, Callable[..., Any]] = {
    "goToDefinition": _op_definition,
    "findReferences": _op_references,
    "hover": _op_hover,
    "documentSymbol": _op_document_symbol,
    "workspaceSymbol": _op_workspace_symbol,
    "goToImplementation": _op_implementation,
    "prepareCallHierarchy": _op_prepare_call_hierarchy,
    "incomingCalls": _op_call_hierarchy("incoming"),
    "outgoingCalls": _op_call_hierarchy("outgoing"),
}


class LSP(Tool):
    name = "LSP"
    description = DESCRIPTION
    parameters = PARAMETERS
    is_readonly = True

    def run(self, args: dict, ctx: ToolContext) -> str:
        operation = args.get("operation")
        if operation not in OPERATIONS:
            return f"Error: unsupported LSP operation {operation!r}"

        raw_path = str(args.get("file_path", ""))
        if not raw_path:
            return "Error: file_path must not be empty"
        path = _resolve_path(raw_path, ctx.cwd)
        if not os.path.isfile(path):
            return f"Error: File not found: {raw_path}"

        needs_position = operation in _POSITION_OPS

        try:
            line = int(args.get("line", 1))
            character = int(args.get("character", 1))
        except (TypeError, ValueError):
            return "Error: line and character must be integers"
        if needs_position and (line < 1 or character < 1):
            return "Error: line and character must be >= 1"

        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError) as e:
            return f"Error: failed to read {raw_path}: {e}"
        lines = text.splitlines(keepends=True)
        if needs_position and line > len(lines):
            return f"Error: line {line} is beyond end of file ({len(lines)} lines)"

        try:
            client, _ = get_client(path, ctx.cwd, ctx.config_path)
            uri = Path(path).as_uri()
            client.open_document(uri, text)
            try:
                position: dict[str, int] | None = None
                if needs_position:
                    position = _to_lsp_position(lines, line, character, client.position_encoding)

                handler = _DISPATCH[operation]
                result = handler(client, uri, position, args)
            finally:
                with contextlib.suppress(Exception):
                    client.close_document(uri)
        except (LSPError, ValueError) as e:
            return f"Error: {e}"

        if isinstance(result, str):
            return result
        return _format_result(result, operation)
