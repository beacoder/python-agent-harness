"""TodoWrite is parent-only: a sub-agent must not see the tool spec and a
hallucinated TodoWrite call must be refused without touching the parent's
todo list.
"""

import json
import unittest

from python_agent_harness.agent import AgentLoop
from python_agent_harness.agent_session import AgentSession
from python_agent_harness.models import Message, ToolCall, Usage
from python_agent_harness.tools import default_registry


class FakeClient:
    """Scripted: first chat returns a TodoWrite tool call, then a reply."""

    def __init__(self, sub_todos):
        self.sub_todos = sub_todos
        self.n = 0
        self.sent_tools = []

    def chat(self, messages, tools=None, system=None, temperature=None,
             max_tokens=None, reasoning_effort=None, on_delta=None, stream=True,
             cancel_check=None, on_retry=None):
        self.n += 1
        self.sent_tools.append([t.name for t in tools] if tools else None)
        if self.n == 1:
            tc = ToolCall(
                id="call_1", name="TodoWrite",
                arguments=json.dumps({"todos": self.sub_todos}),
            )
            return Message(role="assistant", content="", tool_calls=[tc]), Usage(input_tokens=10)
        return Message(role="assistant", content="sub done"), Usage(input_tokens=10)

    def chat_sync(self, *a, **k):
        return Message(role="assistant", content="SYNC"), Usage()

    def close(self):
        pass


def make_session() -> AgentSession:
    return AgentSession(
        project_dir="/tmp/fakeproj",
        client=FakeClient([]),
        model="m",
        registry=default_registry(),
    )


class TestSubagentTodoIsolation(unittest.TestCase):
    def test_update_todos_writes_parent_list(self):
        s = make_session()
        s.update_todos([{"content": "parent task", "status": "in_progress"}])
        self.assertEqual(s.todos[0]["content"], "parent task")
        s.clear_todos()
        self.assertEqual(s.todos, [])

    def test_subagent_todowrite_call_refused(self):
        """Defense in depth: even a hallucinated TodoWrite call from a
        sub-agent is refused and never reaches the registry, so the
        parent's todo list is untouched."""
        sub_todos = [
            {"content": "search", "status": "in_progress"},
            {"content": "read", "status": "pending"},
        ]
        s = make_session()
        s.client = FakeClient(sub_todos)
        s.update_todos([{"content": "parent task", "status": "in_progress"}])
        loop = AgentLoop(
            s,
            messages=[Message(role="user", content="do it")],
            top_level=False,
            system="SUB",
        )
        result = loop.run()
        self.assertIn("sub done", result)
        tool_rows = [m for m in loop.messages if m.role == "tool"]
        self.assertTrue(tool_rows)
        self.assertIn("not available to sub-agents", tool_rows[0].text())
        # the parent's todo list was never modified
        self.assertEqual(s.todos[0]["content"], "parent task")

    def test_run_subagent_exception_contained(self):
        s = make_session()

        class BoomClient(FakeClient):
            def chat(self, *a, **k):
                raise RuntimeError("api down")

        s.subagent_client = BoomClient([])
        s.update_todos([{"content": "parent task", "status": "in_progress"}])
        result = s.run_subagent("subagent", "boom", "do it")
        self.assertIn("Error", result)
        self.assertEqual(s.todos[0]["content"], "parent task")  # untouched


if __name__ == "__main__":
    unittest.main()
