"""Agent-facing LSP code-intelligence tool.

A small OpenCode-style surface over the built-in LSP client. The model sees
1-based file/line/character coordinates; the client translates them to LSP's
0-based representation.
"""

from __future__ import annotations

import json
import os
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
            "description": "The line number (1-based, as shown in editors)",
        },
        "character": {
            "type": "integer",
            "minimum": 1,
            "description": "The character offset (1-based, as shown in editors)",
        },
        "query": {
            "type": "string",
            "description": "Search query for workspaceSymbol. Empty string requests all symbols.",
        },
    },
    "required": ["operation", "file_path", "line", "character"],
}


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
        path = (
            os.path.realpath(os.path.abspath(os.path.join(ctx.cwd, raw_path)))
            if not os.path.isabs(raw_path)
            else os.path.realpath(raw_path)
        )
        if not os.path.isfile(path):
            return f"Error: File not found: {raw_path}"

        # Keep the agent-facing contract 1-based, while the LSP protocol is 0-based.
        line = int(args.get("line", 1))
        character = int(args.get("character", 1))
        if line < 1 or character < 1:
            return "Error: line and character must be >= 1"

        try:
            client, _ = get_client(path, ctx.cwd)
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            uri = Path(path).as_uri()
            client.open_document(uri, text)

            # LSP positions are 0-based. Character conversion to the negotiated
            # encoding is done here using the source line.
            lines = text.splitlines(keepends=True)
            if line > len(lines):
                return f"Error: line {line} is beyond end of file ({len(lines)} lines)"
            source_line = lines[line - 1].rstrip("\r\n")
            py_index = min(character - 1, len(source_line))
            if client.position_encoding == "utf-8":
                lsp_character = len(source_line[:py_index].encode("utf-8"))
            elif client.position_encoding == "utf-32":
                lsp_character = py_index
            else:
                lsp_character = len(source_line[:py_index].encode("utf-16-le")) // 2
            position = {"line": line - 1, "character": lsp_character}

            if operation == "goToDefinition":
                result = client.request(
                    "textDocument/definition", {"textDocument": {"uri": uri}, "position": position}
                )
            elif operation == "findReferences":
                result = client.request(
                    "textDocument/references",
                    {
                        "textDocument": {"uri": uri},
                        "position": position,
                        "context": {"includeDeclaration": True},
                    },
                )
            elif operation == "hover":
                result = client.request(
                    "textDocument/hover", {"textDocument": {"uri": uri}, "position": position}
                )
            elif operation == "documentSymbol":
                result = client.request(
                    "textDocument/documentSymbol", {"textDocument": {"uri": uri}}
                )
            elif operation == "workspaceSymbol":
                result = client.request("workspace/symbol", {"query": str(args.get("query", ""))})
            elif operation == "goToImplementation":
                result = client.request(
                    "textDocument/implementation",
                    {"textDocument": {"uri": uri}, "position": position},
                )
            elif operation == "prepareCallHierarchy":
                result = client.request(
                    "textDocument/prepareCallHierarchy",
                    {"textDocument": {"uri": uri}, "position": position},
                )
            else:
                prepared = client.request(
                    "textDocument/prepareCallHierarchy",
                    {"textDocument": {"uri": uri}, "position": position},
                )
                if not prepared:
                    return "No call hierarchy item found at this position"
                item = prepared[0]
                method = (
                    "callHierarchy/incomingCalls"
                    if operation == "incomingCalls"
                    else "callHierarchy/outgoingCalls"
                )
                result = client.request(method, {"item": item})
        except LSPError as e:
            return f"Error: {e}"
        except (OSError, UnicodeError) as e:
            return f"Error: failed to read {raw_path}: {e}"

        if result is None:
            result = []
        if isinstance(result, dict) and not result:
            return f"No results found for {operation}"
        if isinstance(result, list) and not result:
            return f"No results found for {operation}"
        return json.dumps(_jsonable(result), ensure_ascii=False, indent=2)
