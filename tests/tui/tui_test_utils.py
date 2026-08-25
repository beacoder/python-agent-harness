"""Shared fixtures for the TUI test modules (test_tui_*.py).

Not a test module itself (name does not start with ``test_``) so
``unittest discover`` skips it.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
from rich.console import Console

from python_agent_harness.client import Client
from python_agent_harness.models import Message, ToolCall
from python_agent_harness.session import Session
from python_agent_harness.tools import default_registry
from python_agent_harness.tui import Tui


def make_tui() -> tuple[Tui, io.StringIO]:
    c = Client(base_url="http://127.0.0.1:1/v1", api_key="x", model="fake")
    s = Session(
        project_dir="/tmp/fakeproj",
        client=c,
        model="fake",
        registry=default_registry(),
    )
    s.last_messages = [
        Message(role="user", content="hello agent"),
        Message(
            role="assistant",
            content="hi",
            tool_calls=[ToolCall(id="1", name="Read", arguments="{}")],
        ),
        Message(role="tool", content="file contents", tool_call_id="1", name="Read"),
    ]
    s.todos = [{"content": "task one", "status": "in_progress"}]
    s.context_ratio = 0.55
    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False)
    return Tui(s, console), buf
