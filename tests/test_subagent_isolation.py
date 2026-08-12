"""Sub-agent isolation: a sub-agent must never leak its internal
conversation, streaming text, or plan-mode reminders into the parent
session, and must never clobber the parent's history.
"""

import json
import unittest

from python_agent_harness.agent import run_agent_loop
from python_agent_harness.agent_session import AgentSession
from python_agent_harness.models import Message, ToolCall, Usage
from python_agent_harness.tools import default_registry


class RecClient:
    """Scripted client that records streamed deltas and sent payloads."""

    def __init__(self, script):
        self.script = list(script)
        self.n = 0
        self.streamed = []      # turn indices that streamed
        self.sent = []
        self.sent_tools = []    # tool names sent per chat call
        self.kwargs = []        # per-call request options per chat call

    def chat(self, messages, tools=None, system=None, temperature=None,
             max_tokens=None, reasoning_effort=None, on_delta=None, stream=True,
             cancel_check=None, on_retry=None):
        self.n += 1
        self.sent.append([m.to_api() for m in messages])
        self.sent_tools.append([t.name for t in tools] if tools else None)
        self.kwargs.append({
            "temperature": temperature, "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort, "stream": stream,
        })
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
        run_agent_loop(
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

    def test_subagent_uses_separate_client_when_configured(self):
        """A session with a dedicated sub-agent client (subagent_llm in
        the config) routes sub-agent requests through it with its own
        per-request options; the parent's client only serves parent
        turns (mirrors gptel-agent-harness-subagent-model/-backend)."""
        parent_client = RecClient([
            ("", [ToolCall(id="p1", name="Agent", arguments=AGENT_CALL)]),
            "parent done",
        ])
        sub_client = RecClient(["sub done"])
        s = AgentSession(
            project_dir="/tmp/fakeproj",
            client=parent_client,
            subagent_client=sub_client,
            subagent_temperature=0.4,
            subagent_max_tokens=111,
            subagent_reasoning_effort="low",
            subagent_stream=False,
            model="m",
            temperature=0.9,
            max_tokens=222,
            reasoning_effort="high",
            stream=True,
            registry=default_registry(),
        )
        run_agent_loop(
            s, messages=[Message(role="user", content="delegate")], top_level=True
        )
        # the sub-agent's request went to the dedicated client
        self.assertEqual(len(sub_client.sent), 1)
        self.assertIn("find stuff", sub_client.sent[0][0]["content"])
        # with the sub-agent LLM's per-request options
        self.assertEqual(sub_client.kwargs[-1]["temperature"], 0.4)
        self.assertEqual(sub_client.kwargs[-1]["max_tokens"], 111)
        self.assertEqual(sub_client.kwargs[-1]["reasoning_effort"], "low")
        self.assertIs(sub_client.kwargs[-1]["stream"], False)
        # the parent's client never served the sub-agent (the exact
        # parent call count varies with completion-supervision nudges,
        # so only the payload separation is asserted)
        for payload in parent_client.sent:
            self.assertNotIn("find stuff", [m["content"] for m in payload])
        self.assertGreaterEqual(len(parent_client.sent), 2)

    def test_subagent_client_defaults_to_main(self):
        """Without a configured sub-agent LLM the session falls back to
        the main client and the main per-request options."""
        client = RecClient(["sub done"])
        s = make_session(client)
        self.assertIs(s.subagent_client, s.client)
        self.assertIs(s.subagent_temperature, s.temperature)
        self.assertIs(s.subagent_max_tokens, s.max_tokens)
        self.assertIs(s.subagent_reasoning_effort, s.reasoning_effort)
        self.assertIs(s.subagent_stream, s.stream)

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

    def test_injected_prompts_not_used_as_title_source(self):
        """Harness-injected messages (nudge, plan/build-switch reminders,
        queued mode prompts) must never become the first-user message
        used for session-title generation."""
        s = make_session(RecClient([]))
        s.remember_user_text([
            Message(role="user", content="real request"),
            Message(role="assistant", content="ok"),
            Message(
                role="user",
                content="<system-reminder>\nYour operational mode has changed "
                        "from plan to build.\n</system-reminder>",
                injected=True,
            ),
        ])
        self.assertEqual(s.store.first_user_message(), "real request")

    def test_only_injected_messages_yield_no_title_source(self):
        """When every user message is harness-injected there is no real
        first-user message to title from."""
        s = make_session(RecClient([]))
        s.remember_user_text([
            Message(role="user", content="plan reminder", injected=True),
            Message(role="assistant", content="ok"),
            Message(role="user", content="nudge", injected=True),
        ])
        self.assertIsNone(s.store.first_user_message())

    def test_subagent_does_not_touch_shared_context_accounting(self):
        """The sub-agent's rounds must not update the shared context
        ratio or calibration factor: its payload (fresh context) is
        structurally different, so its usage would skew the parent's
        compaction decisions."""
        s = make_session(RecClient([]))
        s.run_subagent("subagent", "explore", "find stuff")
        self.assertIsNone(s.context_ratio)
        self.assertEqual(s.calibrator.factor, 1.0)
        self.assertIsNone(s.calibrator.last_raw_estimate)

    def test_subagent_does_not_get_parent_only_specs(self):
        """Parent-only tools (Agent, Question, PlanExit, TodoWrite) are
        excluded from the sub-agent's request specs, while the parent
        keeps them."""
        from python_agent_harness.tools import PlanExit

        client = RecClient([
            ("", [ToolCall(id="p1", name="Agent", arguments=AGENT_CALL)]),
            "sub done",
            "parent done",
        ])
        s = make_session(client)
        s.registry.register(PlanExit())  # plan mode registers it too
        run_agent_loop(
            s, messages=[Message(role="user", content="delegate")], top_level=True
        )
        # turn 1 = parent's Agent call, turn 2 = the sub-agent, turn 3 = parent
        parent_tools, sub_tools = client.sent_tools[0], client.sent_tools[1]
        self.assertIn("Agent", parent_tools)
        self.assertIn("Question", parent_tools)
        self.assertIn("PlanExit", parent_tools)
        self.assertIn("TodoWrite", parent_tools)
        self.assertNotIn("Agent", sub_tools)
        self.assertNotIn("Question", sub_tools)
        self.assertNotIn("PlanExit", sub_tools)
        self.assertNotIn("TodoWrite", sub_tools)
        # the sub-agent keeps its working tools
        self.assertIn("Read", sub_tools)
        self.assertIn("Bash", sub_tools)

    def test_subagent_parent_only_call_refused_at_execution(self):
        """Defense in depth: even a hallucinated parent-only call (Agent,
        Question, PlanExit, TodoWrite) from a sub-agent must be refused
        at execution time, not silently run."""
        from python_agent_harness.agent import AgentLoop
        from python_agent_harness.tools import PlanExit

        client = RecClient([
            ("", [ToolCall(id="s1", name="Agent", arguments=AGENT_CALL)]),
            "sub done",
        ])
        s = make_session(client)
        s.registry.register(PlanExit())
        loop = AgentLoop(
            s,
            messages=[Message(role="user", content="find stuff")],
            top_level=False,
            system="SUB",
        )
        loop.run()
        tool_rows = [m for m in loop.messages if m.role == "tool"]
        self.assertTrue(tool_rows)
        self.assertIn("not available to sub-agents", tool_rows[0].text())
        # the refused call never executed the Agent tool, so no nested
        # sub-agent loop ran: exactly 2 chat calls (refused call turn +
        # the terminal "sub done" reply) — a real delegation would have
        # spawned an additional nested loop's chat call
        self.assertEqual(len(client.sent), 2)

    def test_tool_round_inside_subagent_loop(self):
        """A sub-agent's OWN multi-tool round runs with the same
        gptel-style semantics as the parent's: sync tools execute one
        at a time in call order, a hallucinated parent-only call is
        refused inline while its siblings still run, results arrive in
        original order, and nothing leaks into the parent's shared
        history."""
        import threading
        import time

        from python_agent_harness.agent import AgentLoop

        class BlockingSession(AgentSession):
            def __init__(self, client, duration=0.3):
                super().__init__(
                    project_dir="/tmp/fakeproj", client=client, model="m",
                    registry=default_registry(),
                )
                self.duration = duration
                self.active = 0
                self.max_active = 0
                self._lock = threading.Lock()

            def execute_tool(self, name, args, call_id=None):
                with self._lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    time.sleep(self.duration)
                    return f"result of {name}"
                finally:
                    with self._lock:
                        self.active -= 1

        client = RecClient([
            ("", [
                ToolCall(id="s1", name="Agent", arguments=AGENT_CALL),
                ToolCall(id="s2", name="Read", arguments='{"file_path": "/tmp/x.py"}'),
                ToolCall(id="s3", name="Bash", arguments='{"command": "echo hi"}'),
            ]),
            "sub done",
        ])
        s = BlockingSession(client)
        loop = AgentLoop(
            s,
            messages=[Message(role="user", content="find stuff")],
            top_level=False,
            system="SUB",
        )
        result = loop.run()
        self.assertEqual(result, "sub done")
        # the fake session treats every tool as sync, so the round is
        # fully sequential: never more than one tool at a time
        self.assertEqual(s.max_active, 1)
        # refused call first (original order), siblings after it
        tool_rows = [(m.tool_call_id, m.text()) for m in loop.messages if m.role == "tool"]
        self.assertEqual(
            [t[0] for t in tool_rows], ["s1", "s2", "s3"],
        )
        self.assertIn("not available to sub-agents", tool_rows[0][1])
        self.assertEqual(tool_rows[1][1], "result of Read")
        self.assertEqual(tool_rows[2][1], "result of Bash")
        # a sub-agent never mirrors its history onto the shared session
        self.assertEqual(s.last_messages, [])


if __name__ == "__main__":
    unittest.main()
