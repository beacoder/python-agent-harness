"""Sub-agent isolation: a sub-agent must never leak its internal
conversation, streaming text, or plan-mode reminders into the parent
session, and must never clobber the parent's history.
"""

import json
import unittest

from python_agent_harness.agent import run_agent_loop
from python_agent_harness.harness import AgentSession
from python_agent_harness.models import Message, ToolCall, Usage
from python_agent_harness.tools import default_registry


class RecClient:
    """Scripted client that records streamed deltas and sent payloads."""

    def __init__(self, script):
        self.script = list(script)
        self.n = 0
        self.streamed = []      # turn indices that streamed
        self.sent = []

    def chat(self, messages, tools=None, system=None, temperature=None,
             max_tokens=None, reasoning_effort=None, on_delta=None):
        self.n += 1
        self.sent.append([m.to_api() for m in messages])
        if on_delta:
            on_delta(f"stream-{self.n} ")
            self.streamed.append(self.n)
        item = self.script.pop(0) if self.script else "done"
        if isinstance(item, tuple):
            text, tcs = item
        else:
            text, tcs = item, None
        return Message(role="assistant", content=text, tool_calls=tcs), Usage(input_tokens=10)

    def chat_sync(self, *a, **k):
        return Message(role="assistant", content="SYNC"), Usage()

    def close(self):
        pass


AGENT_CALL = json.dumps({
    "subagent_type": "subagent",
    "description": "explore",
    "prompt": "find stuff",
})


def make_session(client) -> AgentSession:
    return AgentSession(
        project_dir="/tmp/fakeproj", client=client, model="m",
        registry=default_registry(),
    )


class TestSubagentIsolation(unittest.TestCase):
    def test_subagent_history_not_in_parent_last_messages(self):
        """The sub-agent's internal messages (its prompt/replies) must
        not end up in the parent's conversation history — only the Agent
        tool result (its return string) may appear."""
        client = RecClient([
            ("", [ToolCall(id="p1", name="Agent", arguments=AGENT_CALL)]),  # parent
            ("", [ToolCall(id="s1", name="Read", arguments="{}")]),          # sub
            "sub done",                                                      # sub
            "parent done",                                                    # parent
        ])
        s = make_session(client)
        result = run_agent_loop(
            s, messages=[Message(role="user", content="delegate")], top_level=True
        )
        texts = [m.text() for m in s.last_messages]
        self.assertIn("parent done", texts)
        # the sub-agent's internal prompt must never appear in history
        self.assertFalse(any("find stuff" in t for t in texts))
        # 'sub done' may appear ONLY as the Agent tool result row
        sub_done_rows = [m for m in s.last_messages if "sub done" in m.text()]
        self.assertTrue(sub_done_rows)
        self.assertTrue(all(m.role == "tool" for m in sub_done_rows))

    def test_subagent_does_not_stream_into_parent(self):
        """Sub-agent chat turns must not call the parent's on_delta."""
        client = RecClient([
            ("", [ToolCall(id="p1", name="Agent", arguments=AGENT_CALL)]),
            ("", [ToolCall(id="s1", name="Read", arguments="{}")]),
            "sub done",
            "parent done",
        ])
        s = make_session(client)
        s.on_delta = lambda t: None  # parent streams
        run_agent_loop(
            s, messages=[Message(role="user", content="delegate")], top_level=True
        )
        # turns 2 and 3 are the sub-agent's chats — they must NOT stream;
        # all other turns (parent, incl. nudge-redirects) may stream
        self.assertNotIn(2, client.streamed)
        self.assertNotIn(3, client.streamed)
        self.assertIn(1, client.streamed)   # parent's Agent-call turn
        self.assertIn(4, client.streamed)   # parent's reply turn

    def test_plan_reminder_injected_once(self):
        """In plan mode the read-only reminder must appear exactly once
        in the sub-agent's request (not duplicated by the runner)."""
        from python_agent_harness.models import AgentMode

        client = RecClient(["sub done"])
        s = make_session(client)
        s.plan_mode.set_mode(AgentMode.PLAN, {
            "plan": "P", "plan-mode": "PM ${planInfo}", "build-switch": "B",
        })
        result = s.run_subagent("subagent", "explore", "find stuff")
        self.assertIn("sub done", result)
        payloads = client.sent
        self.assertTrue(payloads)
        reminder = s.plan_mode.plan_reminder()
        count = sum(
            m.get("content") == reminder
            for payload in payloads for m in payload
        )
        self.assertEqual(count, 1, "reminder injected more than once")

    def test_nudge_not_used_as_title_source(self):
        """The completion-nudge message must never become the first-user
        message used for session-title generation."""
        s = make_session(RecClient([]))
        s.remember_user_text([
            Message(role="user", content="real request"),
            Message(role="assistant", content="ok"),
            Message(role="user", content="Review the original user request and the Task Completion Rules in the context. Verify whether all completion criteria are satisfied. If not, continue by making tool calls. Do not stop until the rules are fully met."),
        ])
        self.assertEqual(s.store.first_user_message(), "real request")

    def test_subagent_never_inherits_parent_prompt(self):
        """A session without a configured subagent prompt must still give
        the sub-agent ONLY its own default prompt — never the parent's
        system prompt (which carries context + completion rules)."""
        from python_agent_harness.compaction import assemble_agent_prompt

        parent = make_session(RecClient([]))
        parent.system_prompt = assemble_agent_prompt(
            "/tmp/fakeproj", "PARENT PROMPT", include_context=False
        )  # parent prompt includes the rules
        parent.subagent_system_prompt = None  # not configured
        # FakeClient with no script returns the fallback "done" — fine
        parent.run_subagent("subagent", "explore", "do it")
        from python_agent_harness.subagent import _subagent_system_prompt

        sp = _subagent_system_prompt(parent)
        self.assertIsNotNone(sp)
        self.assertIn("autonomous subagent", sp)
        self.assertNotIn("Task Completion Rules", sp)
        self.assertNotIn("PARENT PROMPT", sp)


if __name__ == "__main__":
    unittest.main()
