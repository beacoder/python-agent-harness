"""Scoped todos: a sub-agent's TodoWrite must not clobber the parent list."""

import json
import unittest

from python_agent_harness.agent_session import AgentSession
from python_agent_harness.models import Message, ToolCall, Usage
from python_agent_harness.tools import default_registry


class FakeClient:
    """Scripted: first chat returns a TodoWrite tool call, then a reply."""

    def __init__(self, sub_todos):
        self.sub_todos = sub_todos
        self.n = 0

    def chat(self, messages, tools=None, system=None, temperature=None,
             max_tokens=None, reasoning_effort=None, on_delta=None, stream=True):
        self.n += 1
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


class TestTodoScoping(unittest.TestCase):
    def test_update_todos_writes_active_scope(self):
        s = make_session()
        s.update_todos([{"content": "parent task", "status": "in_progress"}])
        self.assertEqual(s.todos[0]["content"], "parent task")
        self.assertEqual(s._todo_scopes["main"][0]["content"], "parent task")

    def test_push_pop_restores_parent(self):
        s = make_session()
        s.update_todos([{"content": "parent task", "status": "pending"}])
        s.push_todo_scope("sub:1", "explore code")
        self.assertEqual(s.todos, [])  # sub scope starts empty
        self.assertEqual(s.todo_scope_label, "explore code")
        s.update_todos([{"content": "sub task", "status": "in_progress"}])
        self.assertEqual(s.todos[0]["content"], "sub task")
        # parent list untouched in its own scope
        self.assertEqual(s._todo_scopes["main"][0]["content"], "parent task")
        s.pop_todo_scope()
        self.assertEqual(s.todos[0]["content"], "parent task")  # restored
        self.assertIsNone(s.todo_scope_label)

    def test_nested_scopes(self):
        s = make_session()
        s.update_todos([{"content": "parent", "status": "pending"}])
        s.push_todo_scope("sub:1", "outer")
        s.update_todos([{"content": "outer sub", "status": "pending"}])
        s.push_todo_scope("sub:2", "inner")
        s.update_todos([{"content": "inner sub", "status": "pending"}])
        s.pop_todo_scope()
        self.assertEqual(s.todos[0]["content"], "outer sub")
        s.pop_todo_scope()
        self.assertEqual(s.todos[0]["content"], "parent")

    def test_run_subagent_isolates_and_restores(self):
        sub_todos = [
            {"content": "search", "status": "in_progress"},
            {"content": "read", "status": "pending"},
        ]
        s = make_session()
        s.client = FakeClient(sub_todos)
        s.update_todos([{"content": "parent task", "status": "in_progress"}])
        result = s.run_subagent("subagent", "find the bug", "do it")
        self.assertIn("sub done", result)
        # parent's todo list restored after the sub-agent finished
        self.assertEqual(s.todos[0]["content"], "parent task")
        # sub-agent's list is preserved in its own scope
        self.assertEqual(s._todo_scopes["sub:1"][0]["content"], "search")
        self.assertEqual(s.todo_scope_label, None)  # back on main

    def test_run_subagent_exception_restores(self):
        s = make_session()

        class BoomClient(FakeClient):
            def chat(self, *a, **k):
                raise RuntimeError("api down")

        s.client = BoomClient([])
        s.update_todos([{"content": "parent task", "status": "in_progress"}])
        result = s.run_subagent("subagent", "boom", "do it")
        self.assertIn("Error", result)
        self.assertEqual(s.todos[0]["content"], "parent task")  # restored
        self.assertEqual(s.todo_scope_label, None)


if __name__ == "__main__":
    unittest.main()
