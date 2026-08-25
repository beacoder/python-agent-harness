"""End-to-end agent loop tests with a fake client and fake session."""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # tests/ for plan_cleanup

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)

from python_agent_harness.models import Message, ToolCall, Usage
from python_agent_harness.persistence import SessionPersistence
from python_agent_harness.session import Session
from python_agent_harness.tools import default_registry


class FakeClient:
    """Scripted chat responses: (assistant_text, tool_calls) per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.kwargs = []
        self.base_url = "https://fake.example/v1"
        self.api_key = None
        self.model = "gpt-5-mini"
        self.timeout = 600.0

    def set_timeout(self, timeout):
        self.timeout = timeout

    def chat(
        self,
        messages,
        tools=None,
        system=None,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
        on_delta=None,
        stream=True,
        cancel_check=None,
        on_retry=None,
    ):
        self.calls.append([m.to_api() for m in messages])
        self.kwargs.append(
            {
                "tools": tools,
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
                "stream": stream,
                "cancel_check": cancel_check,
            }
        )
        if not self.script:
            return Message(role="assistant", content="done"), Usage()
        item = self.script.pop(0)
        if isinstance(item, tuple):
            text, tool_calls = item
        else:
            text, tool_calls = item, None
        return Message(role="assistant", content=text, tool_calls=tool_calls), Usage(
            input_tokens=100
        )

    def chat_sync(
        self,
        messages,
        system=None,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
        cancel_check=None,
    ):
        return Message(role="assistant", content="SYNC-OK"), Usage()


class RecordingSession(Session):
    _test_session_dir: str | None = None

    def __init__(self, project_dir="/tmp/fakeproj", model_profiles=None, llm_settings=None):
        if RecordingSession._test_session_dir is None:
            import tempfile as _tf

            RecordingSession._test_session_dir = _tf.mkdtemp(prefix="pah-test-sessions-")
            import python_agent_harness.config as cfg

            cfg.SESSION_DIR = __import__("pathlib").Path(RecordingSession._test_session_dir)
        super().__init__(
            project_dir=project_dir,
            client=FakeClient([]),
            model="gpt-5-mini",
            registry=default_registry(),
            model_profiles=model_profiles or {},
            llm_settings=llm_settings,
        )
        self.executed = []
        self.store = SessionPersistence(
            project_dir=project_dir,
            model=self.model,
            backend=self.backend,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tool_names=self.store.tool_names,
        )

    def execute_tool(self, name, args, call_id=None):
        self.executed.append((name, args))
        if name == "Read":
            return "file content"
        if name == "Bash":
            return "bash output"
        return f"result of {name}"


class ParallelToolSession(RecordingSession):
    """Session whose tool calls block (DURATION seconds, or until cancel
    when DURATION is None) while tracking peak concurrency across ALL
    tools."""

    def __init__(self, duration=0.4):
        super().__init__()
        self.duration = duration
        self.active = 0
        self.max_active = 0
        self.executed_count = 0
        self._lock = threading.Lock()
        self.started = threading.Event()

    def execute_tool(self, name, args, call_id=None):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.executed_count += 1
        self.started.set()
        try:
            if self.duration is None:
                deadline = time.monotonic() + 30
                while not self.cancel_event.is_set() and time.monotonic() < deadline:
                    time.sleep(0.02)
            else:
                time.sleep(self.duration)
            if name == "Agent":
                return f"done:{args.get('description', 'task')}"
            return f"result of {name}"
        finally:
            with self._lock:
                self.active -= 1


class SerialPromptSession(RecordingSession):
    """Session whose interactive prompt handlers (Question / PlanExit
    confirm) each block DURATION seconds while tracking peak
    concurrency — to verify the session serializes them."""

    def __init__(self, duration=0.3):
        super().__init__()
        self.duration = duration
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()
        self.ask_fn = self._prompt
        self.confirm_fn = self._prompt

    def _prompt(self, prompt):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.duration)
            return "yes"
        finally:
            with self._lock:
                self.active -= 1


class StaggeredSession(RecordingSession):
    """Session with per-tool durations that records the real execution
    completion order — so a test can prove the round delivers results
    in ORIGINAL call order even when execution finishes in a different
    order."""

    def __init__(self, durations):
        super().__init__()
        self.durations = durations
        self.active = 0
        self.max_active = 0
        self.completed = []
        self._lock = threading.Lock()

    def execute_tool(self, name, args, call_id=None):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.durations[name])
            return f"result of {name}"
        finally:
            with self._lock:
                self.active -= 1
                self.completed.append(name)


class RealParallelSession(RecordingSession):
    """Session that runs REAL sub-agents (the Agent tool delegates to
    the real Session.execute_tool) while every other tool blocks
    DURATION seconds, tracking concurrency across the parent round and
    the sub-agent's own round alike (a sub-agent shares this session)."""

    def __init__(self, duration=0.4):
        super().__init__()
        self.duration = duration
        self.active = 0
        self.max_active = 0
        self.executed_names = []
        self._lock = threading.Lock()

    def execute_tool(self, name, args, call_id=None):
        if name == "Agent":
            return Session.execute_tool(self, name, args, call_id=call_id)
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.executed_names.append(name)
        try:
            time.sleep(self.duration)
            return f"result of {name}"
        finally:
            with self._lock:
                self.active -= 1


def agent_call(call_id, description, prompt="do it"):
    return ToolCall(
        id=call_id,
        name="Agent",
        arguments=json.dumps(
            {
                "subagent_type": "subagent",
                "description": description,
                "prompt": prompt,
            }
        ),
    )
