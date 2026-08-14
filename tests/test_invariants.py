"""Property / state-machine tests for the agent harness.

The harness's run loop is a finite state machine
(WAIT/TOOL/TRET/SUPERVISE with terminal DONE/ERRS/ABRT) layered on
top of a session with cancellation generations, run generations and a
plan/build mode.  These tests pin the machine's *invariants*, not its
happy paths:

- DONE/ERRS/ABRT have no outgoing execution transition (no table
  entry, so the machine can never route out of a terminal state; the
  driver stops at the first terminal state)
- a run that ends in ABRT never mutates the session's active state
  (a stale worker touches nothing; a cancelled owner only commits a
  round-complete salvage)
- a cancelled run can never append to a newer run
- plan mode cannot execute arbitrary write tools (only the plan file
  itself stays writable)
- every tool call eventually gets exactly one terminal result (no
  duplicate tool rows, no orphans, no dangling rounds)
- a mid-stream retry never duplicates assistant stream content

Each randomized property is deterministic (a seeded ``random.Random``
per case), so failures reproduce with the seed printed by
``subTest``.
"""

import contextlib
import json
import os
import random
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))  # sibling test helpers

from test_agent import ParallelToolSession, RecordingSession, agent_call  # noqa: E402

from python_agent_harness import config  # noqa: E402
from python_agent_harness.agent import AgentLoop  # noqa: E402
from python_agent_harness.agent_session import AgentSession  # noqa: E402
from python_agent_harness.client import Client  # noqa: E402
from python_agent_harness.models import Message, ToolCall, Usage  # noqa: E402
from python_agent_harness.tools import default_registry  # noqa: E402

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
TOOL_NAMES = ("Read", "Grep", "Glob", "Bash", "Write")


def valid_args(name, tag):
    """Schema-valid arguments for NAME (every required param present,
    so the call reaches execute_tool instead of failing validation)."""
    if name == "Read":
        return {"file_path": f"/tmp/pah-prop-{tag}.py"}
    if name == "Grep":
        return {"regex": "x", "path": "/tmp"}
    if name == "Glob":
        return {"pattern": "*.py"}
    if name == "Bash":
        return {"command": "echo hi"}
    return {"path": "/tmp", "filename": f"{tag}.txt", "content": "x"}


def well_formed_call(rng, round_no, i):
    """A tool call whose arguments pass schema validation."""
    name = rng.choice(TOOL_NAMES)
    return ToolCall(id=f"r{round_no}c{i}", name=name, arguments=json.dumps(valid_args(name, i)))


def random_call(rng, round_no, i):
    """A random tool call: mostly well-formed, sometimes malformed
    arguments (the loop must deliver an error result either way)."""
    call_id = f"r{round_no}c{i}"
    name = rng.choice(TOOL_NAMES)
    roll = rng.random()
    if roll < 0.15:
        arguments = "not-json{{{" if rng.random() < 0.5 else "[1, 2, 3]"
    elif roll < 0.30:
        arguments = json.dumps({"unexpected": "args"})
    else:
        arguments = json.dumps(valid_args(name, call_id))
    return ToolCall(id=call_id, name=name, arguments=arguments)


def random_script(rng, rounds=None):
    """Random scripted responses: mostly tool rounds, occasionally a
    plain text reply (which ends a non-agentic run early), always
    ending with a terminal text."""
    n = rounds if rounds is not None else rng.randint(1, 5)
    script = []
    for r in range(n):
        if rng.random() < 0.15:
            script.append(f"text {rng.randint(0, 10**6)}")
        else:
            calls = [random_call(rng, r, i) for i in range(rng.randint(1, 4))]
            script.append(("", calls))
    script.append(f"final {rng.randint(0, 10**6)}")
    return script


class ScriptedClient:
    """Scripted chat client with fault injection for property runs.

    Items are a plain text, a ``(text, tool_calls)`` tuple, an
    Exception (API error), or the string ``"CANCEL"`` (cancel while a
    request is in flight, before the response is consumed).
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.session = None

    def chat(self, messages, **kwargs):
        self.calls.append([m.to_api() for m in messages])
        if not self.script:
            return Message(role="assistant", content="done"), Usage()
        item = self.script.pop(0)
        if item == "CANCEL":
            self.session.cancel()
            item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            text, tool_calls = item
        else:
            text, tool_calls = item, None
        return Message(role="assistant", content=text, tool_calls=tool_calls), Usage(
            input_tokens=100
        )

    def chat_sync(self, messages, **kwargs):
        return Message(role="assistant", content="SYNC-OK"), Usage()


def assert_conversation_valid(testcase, messages):
    """A conversation is well-formed iff every tool call gets exactly
    one terminal result: each assistant tool-call round is followed by
    exactly the results for its call ids, in call order, with no
    duplicates, no orphans, and no dangling round."""
    open_round = None
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            testcase.assertIsNone(open_round, "new tool round opened before the previous closed")
            ids = [tc.id for tc in m.tool_calls]
            testcase.assertEqual(len(ids), len(set(ids)), "duplicate call ids within a round")
            open_round = {"ids": ids, "delivered": []}
        elif m.role == "tool":
            testcase.assertIsNotNone(open_round, "orphan tool result: no matching assistant call")
            testcase.assertIn(m.tool_call_id, open_round["ids"], "tool result for unknown call id")
            testcase.assertNotIn(
                m.tool_call_id, open_round["delivered"], "duplicate tool result (delivered twice)"
            )
            open_round["delivered"].append(m.tool_call_id)
            if len(open_round["delivered"]) == len(open_round["ids"]):
                testcase.assertEqual(
                    open_round["delivered"],
                    open_round["ids"],
                    "results delivered out of call order",
                )
                open_round = None
        else:
            testcase.assertIsNone(
                open_round, f"{m.role!r} message arrived mid-round (role alternation broken)"
            )
    testcase.assertIsNone(open_round, "dangling tool round: calls never got their results")


def make_abort_scenario(rng, variant):
    """Build ``(session, loop, state, stack)`` for an ABRT-bound run.

    The cancel/staleness is injected at VARIANT:

    - "pre-cancel": cancel before ``run()``
    - "chat-cancel": the client cancels while a request is in flight
    - "tool-cancel": cancel during a randomly chosen tool call
    - "deliver-cancel": cancel during a randomly chosen result delivery
    - "stale-pre": a newer run starts (run_generation bumped) before run()
    - "stale-mid": a newer run starts during a randomly chosen tool call

    ``state["snapshot"]`` is the shared history at the moment the run
    lost its license to write it (None for owner-cancelled variants —
    those commit a salvage instead).  It lives in a dict because the
    stale-mid variant rebinds it from inside the tool-execution
    closure, after the tuple was returned.  STACK carries active
    patchers and must be entered before ``run()``.
    """
    session = RecordingSession()
    session.tools_enabled = False
    stack = contextlib.ExitStack()
    state = {"snapshot": None}
    script = []

    if variant == "chat-cancel":
        script = ["CANCEL", "late text"] + random_script(rng, rounds=rng.randint(1, 2))
    elif variant in ("tool-cancel", "deliver-cancel", "stale-mid"):
        # every call is schema-valid so the chosen cancel/stale point is
        # always reachable inside the sequential tool loop
        calls = [well_formed_call(rng, 0, i) for i in range(rng.randint(2, 4))]
        script = [("", calls), "after round"] + random_script(rng, rounds=1)
    else:
        script = random_script(rng, rounds=rng.randint(1, 3))

    client = ScriptedClient(script)
    client.session = session
    session.client = client
    loop = AgentLoop(session, messages=[Message(role="user", content="q")])

    if variant == "pre-cancel":
        session.cancel()
    elif variant == "stale-pre":
        state["snapshot"] = [m.to_api() for m in session.last_messages]
        session.run_generation += 1
    elif variant in ("tool-cancel", "deliver-cancel", "stale-mid"):
        target = rng.randint(1, len(script[0][1]))
        orig = RecordingSession.execute_tool
        calls = {"n": 0}

        def mid_round(name, args, call_id=None):
            calls["n"] += 1
            if calls["n"] == target:
                if variant == "tool-cancel":
                    session.cancel()
                elif variant == "stale-mid":
                    state["snapshot"] = [m.to_api() for m in session.last_messages]
                    session.run_generation += 1
            return orig(session, name, args, call_id=call_id)

        if variant == "deliver-cancel":
            orig_deliver = AgentLoop._deliver_tool_result

            def cancelling_deliver(self, p, result):
                calls["n"] += 1
                if calls["n"] == target:
                    session.cancel()
                return orig_deliver(self, p, result)

            stack.enter_context(
                mock.patch.object(AgentLoop, "_deliver_tool_result", cancelling_deliver)
            )
        else:
            session.execute_tool = mid_round
    return session, loop, state, stack


def _delta_frame(delta):
    """One chunked-encoding SSE frame carrying DELTA (a ``data:`` line
    terminated by blank lines, so ``iter_lines`` splits each event)."""
    line = f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()
    return f"{len(line):x}\r\n".encode() + line + b"\r\n"


def _split_random(rng, text, pieces):
    """Split TEXT into up to PIECES non-empty chunks at random cuts."""
    cuts = sorted(rng.sample(range(1, len(text)), min(pieces, len(text) - 1)))
    parts = []
    prev = 0
    for c in cuts:
        parts.append(text[prev:c])
        prev = c
    parts.append(text[prev:])
    return parts


class DropHandler(BaseHTTPRequestHandler):
    """Scripted streaming server for retry property tests.

    Each streaming request consumes the next ``(kind, chunks)`` item:
    ``"drop"`` streams CHUNKS then aborts the connection mid-body
    (the client sees a truncated chunked stream); ``"full"`` streams
    CHUNKS then the terminating 0-chunk.  A chunk is a str (content
    delta) or a ``(index, id, name, args_fragment)`` tuple (a tool-call
    delta).  Non-streaming requests always get a plain sync reply.
    """

    protocol_version = "HTTP/1.1"
    script = []
    stream_count = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not body.get("stream", False):
            data = json.dumps(
                {
                    "choices": [{"message": {"role": "assistant", "content": "sync reply"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        type(self).stream_count += 1
        kind, chunks = type(self).script.pop(0)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in chunks:
            if isinstance(chunk, str):
                delta = {"content": chunk}
            else:
                index, call_id, name, fragment = chunk
                delta = {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call_id,
                            "function": {"name": name, "arguments": fragment},
                        }
                    ]
                }
            self.wfile.write(_delta_frame(delta))
            self.wfile.flush()
            time.sleep(0.005)
        if kind == "drop":
            with contextlib.suppress(OSError):
                self.connection.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                self.connection.close()
        else:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    def log_message(self, *a):
        pass


@contextlib.contextmanager
def serve_drop_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), DropHandler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv.server_address
    finally:
        srv.shutdown()
        srv.server_close()


def make_fast_client(base_url):
    return Client(
        base_url=base_url,
        api_key="test",
        model="fake",
        retry_max=5,
        retry_base_delay=0.01,
        retry_max_delay=0.05,
    )


# ----------------------------------------------------------------------
# invariants
# ----------------------------------------------------------------------
class TestFSMStructuralInvariants(unittest.TestCase):
    """DONE/ERRS/ABRT have no outgoing execution transition: no table
    entry, so the machine can never route out of a terminal state;
    the driver stops at the first terminal state."""

    def test_terminal_states_have_no_outgoing_transitions(self):
        """DONE/ERRS/ABRT are sinks: the table has no entry for them,
        so the machine can never route out of a terminal state.  (They
        may be transition TARGETS — WAIT routes to ERRS on API error —
        but never sources.)"""
        for state in AgentLoop.TERMINAL:
            self.assertNotIn(state, AgentLoop.TRANSITIONS)

    def test_every_non_terminal_state_has_transitions_with_true_default(self):
        for state in set(AgentLoop.HANDLERS) - AgentLoop.TERMINAL:
            table = AgentLoop.TRANSITIONS.get(state)
            self.assertIsNotNone(table, f"{state} has no outgoing transitions")
            self.assertIs(table[-1][0], True, f"{state} table lacks a True default")

    def test_every_transition_target_is_a_defined_state_with_handler(self):
        for state, table in AgentLoop.TRANSITIONS.items():
            for _pred, nxt in table:
                self.assertIn(nxt, AgentLoop.HANDLERS, f"{state} -> unknown state {nxt}")

    def test_next_state_from_terminal_state_fails_loudly(self):
        """The driver never routes out of a terminal state; if it ever
        tried, the missing table entry would surface as an error."""
        for state in AgentLoop.TERMINAL:
            session = RecordingSession()
            loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
            loop.state = state
            with self.assertRaises(KeyError):
                loop._next_state()

    def test_driver_stops_at_first_terminal_state(self):
        """No handler executes after a terminal state's handler: DONE
        has no outgoing EXECUTION transition, not just no table entry."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["done text"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        executed = []
        real = dict(AgentLoop.HANDLERS)

        def spy(state):
            def handler(self):
                executed.append(state)
                return real[state](self)

            return handler

        with mock.patch.object(AgentLoop, "HANDLERS", {s: spy(s) for s in real}):
            loop.run()
        self.assertEqual(executed, [AgentLoop.WAIT, AgentLoop.SUPERVISE, AgentLoop.DONE])

    def test_randomized_runs_terminate_in_a_terminal_state(self):
        """Property: whatever the script (text, tool rounds, API
        errors, cancels), run() always terminates in a terminal state
        and the recorded result matches the state — None for ABRT, an
        error string for ERRS, the final assistant text for DONE."""
        for seed in range(30):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                session = RecordingSession()
                session.tools_enabled = False
                script = random_script(rng)
                if rng.random() < 0.3:
                    script.insert(rng.randint(0, len(script) - 1), RuntimeError("api down"))
                if rng.random() < 0.15:
                    session.cancel()
                client = ScriptedClient(script)
                client.session = session
                session.client = client
                loop = AgentLoop(session, messages=[Message(role="user", content="q")])
                result = loop.run()
                self.assertIn(loop.state, AgentLoop.TERMINAL)
                if loop.state == AgentLoop.ABRT:
                    self.assertIsNone(result)
                    self.assertIsNone(loop.result)
                    assert_conversation_valid(self, session.last_messages)
                elif loop.state == AgentLoop.ERRS:
                    self.assertTrue(result.startswith("Error:"))
                    assert_conversation_valid(self, loop.messages)
                else:
                    self.assertIsInstance(result, str)
                    self.assertEqual(result, loop.terminal_text)
                    assert_conversation_valid(self, loop.messages)
                    assert_conversation_valid(self, session.last_messages)


class TestAbortSafety(unittest.TestCase):
    """ABRT cannot mutate the session's active run state: the handler
    records ``result = None`` and nothing else; a stale run never
    touches shared state; a cancelled owner only commits a
    round-complete salvage."""

    def test_abrt_handler_only_records_none(self):
        session = RecordingSession()
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        before = {
            "last_messages": [m.to_api() for m in session.last_messages],
            "todos": list(session.todos),
            "pending_user_prompts": list(session.pending_user_prompts),
            "messages": [m.to_api() for m in loop.messages],
        }
        real_abrt = AgentLoop._handle_abrt

        def abrt_handler(self):
            return real_abrt(self)

        with mock.patch.object(AgentLoop, "_handle_abrt", wraps=abrt_handler, autospec=True) as spy:
            loop.state = AgentLoop.ABRT
            loop._handle_abrt()
        spy.assert_called_once_with(loop)
        self.assertIsNone(loop.result)
        self.assertEqual([m.to_api() for m in session.last_messages], before["last_messages"])
        self.assertEqual(session.todos, before["todos"])
        self.assertEqual(session.pending_user_prompts, before["pending_user_prompts"])
        self.assertEqual([m.to_api() for m in loop.messages], before["messages"])

    def test_randomized_abort_never_mutates_active_state(self):
        variants = (
            "pre-cancel",
            "chat-cancel",
            "tool-cancel",
            "deliver-cancel",
            "stale-pre",
            "stale-mid",
        )
        for seed in range(24):
            variant = variants[seed % len(variants)]
            with self.subTest(seed=seed, variant=variant):
                rng = random.Random(seed)
                session, loop, state, stack = make_abort_scenario(rng, variant)
                with stack:
                    result = loop.run()
                self.assertIsNone(result)
                self.assertEqual(loop.state, AgentLoop.ABRT)
                self.assertIsNone(loop.result)
                snapshot = state["snapshot"]
                if snapshot is not None:
                    self.assertEqual(
                        [m.to_api() for m in session.last_messages],
                        snapshot,
                        f"{variant}: a stale run mutated the shared history",
                    )
                else:
                    assert_conversation_valid(self, session.last_messages)
                    self.assertEqual(
                        [m.to_api() for m in session.last_messages],
                        [m.to_api() for m in loop.messages[: len(session.last_messages)]],
                    )


class TestStaleRunIsolation(unittest.TestCase):
    """A cancelled run cannot append to a newer run: once a newer run
    owns the session (run_generation bumped), the dying worker's
    partial history never lands in the shared conversation."""

    def test_randomized_cancelled_worker_never_appends_to_newer_run(self):
        for seed in range(15):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                session = ParallelToolSession(duration=None)
                session.tools_enabled = False
                n_calls = rng.randint(1, 3)
                session.client.script = [
                    ("", [agent_call(f"a{i}", f"task-{seed}-{i}") for i in range(n_calls)]),
                ]

                # The fake tool blocks until this run is cancelled, and
                # stays unblocked even if the next run's start clears the
                # shared event afterwards (mirrors abort() interrupting
                # the real in-flight tool, which is a one-shot event).
                def blocking_execute(name, args, call_id=None, session=session):
                    with session._lock:
                        session.active += 1
                        session.max_active = max(session.max_active, session.active)
                        session.executed_count += 1
                    session.started.set()
                    try:
                        deadline = time.monotonic() + 30
                        while (
                            session.cancel_generation == 0
                            and not session.cancel_event.is_set()
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.02)
                        if name == "Agent":
                            return f"done:{args.get('description', 'task')}"
                        return f"result of {name}"
                    finally:
                        with session._lock:
                            session.active -= 1

                session.execute_tool = blocking_execute
                out = {}
                worker = threading.Thread(
                    target=lambda out=out, session=session, seed=seed: out.update(
                        r=AgentLoop(
                            session, messages=[Message(role="user", content=f"qA-{seed}")]
                        ).run()
                    )
                )
                worker.start()
                self.assertTrue(session.started.wait(timeout=10))
                time.sleep(rng.uniform(0.05, 0.35))  # the stale run gets some way in
                if seed % 2 == 0:
                    # Ctrl-C first: the stale worker's tool is aborted,
                    # then the new run starts and clears the event
                    session.cancel()
                    time.sleep(0.1)
                    session.run_generation += 1
                    session.cancel_event.clear()
                else:
                    # the new run starts and completes first (clearing the
                    # event), then the cancel lands on the still-blocked
                    # worker from the earlier run
                    session.run_generation += 1
                    session.cancel_event.clear()
                session.client.script = [f"answer-B-{seed}"]
                loop_b = AgentLoop(session, messages=[Message(role="user", content=f"qB-{seed}")])
                result_b = loop_b.run()
                if seed % 2 == 1:
                    session.cancel()
                worker.join(timeout=10)
                self.assertFalse(worker.is_alive(), "stale worker did not terminate")
                # the stale run ends ABRT with no result...
                self.assertIsNone(out.get("r"))
                self.assertEqual(result_b, f"answer-B-{seed}")
                # ...and the shared history holds ONLY the newer run's
                # messages: the cancelled run appended nothing
                self.assertEqual(
                    [m.text() for m in session.last_messages],
                    [f"qB-{seed}", f"answer-B-{seed}"],
                )


class TestPlanModeWriteGuard(unittest.TestCase):
    """Plan mode cannot execute arbitrary write tools: every mutating
    tool is blocked on every target except the per-session plan file
    itself, whatever the arguments; read-only tools pass through."""

    PROMPTS = {"plan": "P", "plan-mode": "P ${planInfo}", "build-switch": "B"}

    def make_plan_session(self, tmpdir):
        session = RecordingSession(project_dir=tmpdir)
        session.plan_mode.set_mode(session.plan_mode.mode.PLAN, self.PROMPTS)
        # the plan file lives in a /tmp plan dir (PlanMode.plan_temp_dir),
        # separate from the test's TemporaryDirectory — remove it when the
        # test finishes so plan-mode tests don't leak dirs into /tmp
        self.addCleanup(session.plan_mode.cleanup_plan_file)

        def execute(name, args, call_id=None):
            return AgentSession.execute_tool(session, name, args, call_id=call_id)

        session.execute_tool = execute
        return session

    def random_mutating_call(self, rng, tmpdir, outside, seed, i):
        """A mutating call and its side-effect target: for creation
        tools (Write/Mkdir/Bash) TARGET must not exist afterwards; for
        mutation tools (Edit/Insert) TARGET is a pre-created file whose
        content must stay untouched (checked by the caller).  TARGETS
        live inside the two per-test tempdirs (the project dir and a
        separate "outside" dir) so the test never depends on — or
        pollutes — shared locations like /tmp."""
        name = rng.choice(("Write", "Edit", "Insert", "Mkdir", "Bash"))
        tag = f"seed-{seed}-call-{i}"
        if name == "Write":
            path = rng.choice(
                (tmpdir, os.path.join(tmpdir, "sub"), outside, os.path.join(outside, "sub"))
            )
            filename = f"{tag}.txt"
            return (
                name,
                {"path": path, "filename": filename, "content": "x"},
                os.path.realpath(os.path.join(path, filename)),
            )
        if name == "Edit":
            target = os.path.join(tmpdir, f"seed-{seed}-file-{rng.randint(0, 2)}.txt")
            if rng.random() < 0.3:
                diff = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"
                return name, {"path": target, "new_str": diff, "diff": True}, None
            return name, {"path": target, "old_str": "content", "new_str": "HACKED"}, None
        if name == "Insert":
            target = os.path.join(tmpdir, f"seed-{seed}-file-{rng.randint(0, 2)}.txt")
            return name, {"path": target, "line_number": 0, "new_str": "HACKED"}, None
        if name == "Mkdir":
            parent = rng.choice((tmpdir, outside, os.path.join(outside, "sub")))
            name_dir = f"{tag}-dir"
            return (
                name,
                {"parent": parent, "name": name_dir},
                os.path.realpath(os.path.join(parent, name_dir)),
            )
        return name, {"command": f"touch {os.path.join(outside, tag)}"}, os.path.join(outside, tag)

    def test_plan_mode_blocks_every_mutating_tool_on_arbitrary_targets(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                with (
                    tempfile.TemporaryDirectory(prefix="pah-prop-plan-") as tmpdir,
                    tempfile.TemporaryDirectory(prefix="pah-prop-plan-out-") as outside,
                ):
                    session = self.make_plan_session(tmpdir)
                    plan_file = session.plan_mode.plan_file
                    self.assertTrue(os.path.isfile(plan_file))
                    existing = {}
                    for i in range(3):
                        p = os.path.join(tmpdir, f"seed-{seed}-file-{i}.txt")
                        with open(p, "w") as f:
                            f.write(f"content-{i}\n")
                        existing[p] = f"content-{i}\n"
                    for i in range(25):
                        name, args, target = self.random_mutating_call(
                            rng, tmpdir, outside, seed, i
                        )
                        result = session.execute_tool(name, args, call_id=f"plan-{seed}-{i}")
                        self.assertIn(
                            "blocked by plan mode",
                            result,
                            f"{name} {args} was not blocked by plan mode",
                        )
                        if target is not None:
                            self.assertNotEqual(
                                target, plan_file, "random call must never target the plan file"
                            )
                            self.assertFalse(os.path.exists(target), f"{name} created {target}")
                    for p, content in existing.items():
                        with open(p) as f:
                            self.assertEqual(f.read(), content, f"{p} was modified in plan mode")

    def test_plan_file_itself_stays_writable_in_plan_mode(self):
        with tempfile.TemporaryDirectory(prefix="pah-prop-plan-") as tmpdir:
            session = self.make_plan_session(tmpdir)
            plan_file = session.plan_mode.plan_file
            dirname, basename = os.path.split(plan_file)
            r = session.execute_tool(
                "Write",
                {"path": dirname, "filename": basename, "content": "# plan\n"},
                call_id="w1",
            )
            self.assertNotIn("blocked by plan mode", r)
            with open(plan_file) as f:
                self.assertEqual(f.read(), "# plan\n")
            r = session.execute_tool(
                "Edit",
                {"path": plan_file, "old_str": "# plan", "new_str": "# plan v2"},
                call_id="e1",
            )
            self.assertNotIn("blocked by plan mode", r)
            r = session.execute_tool(
                "Insert",
                {"path": plan_file, "line_number": -1, "new_str": "- step 1"},
                call_id="i1",
            )
            self.assertNotIn("blocked by plan mode", r)
            with open(plan_file) as f:
                self.assertIn("step 1", f.read())
            # a symlink resolving to the plan file IS the plan file
            link = os.path.join(tmpdir, "plan-link.txt")
            os.symlink(plan_file, link)
            r = session.execute_tool(
                "Write",
                {"path": tmpdir, "filename": "plan-link.txt", "content": "via link"},
                call_id="w2",
            )
            self.assertNotIn("blocked by plan mode", r)
            with open(plan_file) as f:
                self.assertEqual(f.read(), "via link")

    def test_read_only_tools_pass_through_plan_mode(self):
        with tempfile.TemporaryDirectory(prefix="pah-prop-plan-") as tmpdir:
            session = self.make_plan_session(tmpdir)
            target = os.path.join(tmpdir, "readme.txt")
            with open(target, "w") as f:
                f.write("hello plan world\n")
            r = session.execute_tool("Read", {"file_path": target}, call_id="r1")
            self.assertNotIn("blocked by plan mode", r)
            self.assertEqual(r, "hello plan world\n")
            for name, args in (
                ("Glob", {"pattern": "*.txt", "path": tmpdir}),
                ("Grep", {"regex": "plan", "path": tmpdir}),
            ):
                r = session.execute_tool(name, args, call_id=name.lower())
                self.assertNotIn("blocked by plan mode", r)

    def test_build_mode_executes_the_same_calls(self):
        """The guard is plan-mode-specific: in build mode the same
        mutating calls execute and their side effects land."""
        with tempfile.TemporaryDirectory(prefix="pah-prop-plan-") as tmpdir:
            session = RecordingSession(project_dir=tmpdir)

            def execute(name, args, call_id=None):
                return AgentSession.execute_tool(session, name, args, call_id=call_id)

            session.execute_tool = execute
            r = session.execute_tool(
                "Write", {"path": tmpdir, "filename": "built.txt", "content": "built"}, call_id="b1"
            )
            self.assertNotIn("blocked by plan mode", r)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "built.txt")))
            r = session.execute_tool("Mkdir", {"parent": tmpdir, "name": "sub"}, call_id="b2")
            self.assertNotIn("blocked by plan mode", r)
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "sub")))


class TestToolResultCompleteness(unittest.TestCase):
    """Every tool call eventually gets exactly one terminal result: a
    completed conversation never has a duplicate tool row, an orphan
    result, or a dangling unanswered call; cancelled runs only ever
    commit round-complete histories to the shared session."""

    def test_randomized_complete_runs_deliver_every_call_exactly_once(self):
        for seed in range(25):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                session = RecordingSession()
                session.tools_enabled = False
                session.client.script = random_script(rng)
                loop = AgentLoop(session, messages=[Message(role="user", content=f"q-{seed}")])
                result = loop.run()
                self.assertEqual(loop.state, AgentLoop.DONE)
                self.assertIsInstance(result, str)
                assert_conversation_valid(self, loop.messages)
                assistant_ids = [
                    tc.id for m in loop.messages if m.tool_calls for tc in m.tool_calls
                ]
                tool_ids = [m.tool_call_id for m in loop.messages if m.role == "tool"]
                self.assertEqual(len(assistant_ids), len(tool_ids))
                self.assertEqual(len(set(tool_ids)), len(tool_ids))
                self.assertEqual(
                    [m.to_api() for m in session.last_messages],
                    [m.to_api() for m in loop.messages],
                )

    def test_randomized_cancelled_runs_never_commit_dangling_rounds(self):
        variants = ("pre-cancel", "chat-cancel", "tool-cancel", "deliver-cancel")
        for seed in range(16):
            variant = variants[seed % len(variants)]
            with self.subTest(seed=seed, variant=variant):
                rng = random.Random(seed)
                session, loop, state, stack = make_abort_scenario(rng, variant)
                self.assertIsNone(state["snapshot"])  # owner-cancelled: salvage applies
                with stack:
                    self.assertIsNone(loop.run())
                # inside the dead run no call id is ever delivered twice
                tool_ids = [m.tool_call_id for m in loop.messages if m.role == "tool"]
                self.assertEqual(len(tool_ids), len(set(tool_ids)))
                # the shared history never carries an unanswered call
                assert_conversation_valid(self, session.last_messages)


class TestRetryNoDuplication(unittest.TestCase):
    """A mid-stream retry never duplicates assistant stream content:
    text (or tool-call fragments) from a dropped attempt never leaks
    into the final message, the final content appears exactly once,
    and the caller is told to clear partial output once per drop."""

    def test_randomized_mid_stream_drop_retry_never_duplicates_content(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                content = f"retried final answer for seed {seed}"
                n_drops = rng.randint(1, 3)
                script = []
                markers = []
                for d in range(n_drops):
                    marker = f"dropped-attempt-{seed}-{d}-"
                    markers.append(marker)
                    script.append(
                        ("drop", _split_random(rng, marker + "partial-stream", rng.randint(2, 5)))
                    )
                if seed % 2:
                    # final attempt streams a tool call in fragments; like
                    # real OpenAI streams, id/name appear only in the FIRST
                    # fragment (the client accumulates the rest)
                    args_full = '{"file_path": "/tmp/prop-retry.py"}'
                    script.append(
                        (
                            "full",
                            [
                                (0, "call_final", "Read", '{"file_path": "/tmp/prop'),
                                (0, "", "", '-retry.py"}'),
                            ],
                        )
                    )
                else:
                    script.append(("full", _split_random(rng, content, rng.randint(2, 6))))
                DropHandler.script = script
                DropHandler.stream_count = 0
                events = []
                with serve_drop_server() as (host, port):
                    c = make_fast_client(f"http://{host}:{port}/v1")
                    try:
                        msg, _ = c.chat(
                            [Message(role="user", content="hi")],
                            on_delta=lambda t, events=events: events.append(("delta", t)),
                            on_retry=lambda events=events: events.append(("retry",)),
                        )
                    finally:
                        c.close()
                self.assertEqual(DropHandler.stream_count, n_drops + 1)
                if seed % 2:
                    self.assertEqual(len(msg.tool_calls), 1)
                    self.assertEqual(msg.tool_calls[0].id, "call_final")
                    self.assertEqual(msg.tool_calls[0].name, "Read")
                    self.assertEqual(msg.tool_calls[0].arguments, args_full)
                else:
                    self.assertEqual(msg.content, content)
                    for marker in markers:
                        self.assertNotIn(marker, msg.content)
                # after the LAST on_retry the caller's stream contains
                # the final response exactly once — no duplicated text
                tail = []
                for ev in events:
                    if ev[0] == "retry":
                        tail = []
                    else:
                        tail.append(ev[1])
                self.assertEqual("".join(tail), msg.content)
                self.assertEqual(sum(1 for e in events if e[0] == "retry"), n_drops)

    def test_agent_loop_retry_stores_content_exactly_once(self):
        """End to end: a mid-stream drop inside a real agent loop is
        retried, the stored assistant message carries the retried text
        exactly once, and the dropped partial text never reaches the
        history."""
        DropHandler.script = [
            ("drop", ["partial-dropped-text-"]),
            ("full", ["the retried ", "final answer"]),
        ]
        DropHandler.stream_count = 0
        with tempfile.TemporaryDirectory(prefix="pah-prop-retry-") as d:
            config.SESSION_DIR = Path(d)
            with serve_drop_server() as (host, port):
                client = make_fast_client(f"http://{host}:{port}/v1")
                session = AgentSession(
                    project_dir=d,
                    client=client,
                    model="fake",
                    registry=default_registry(),
                    stream=True,
                )
                session.tools_enabled = False
                notified = []
                session.notify_fn = lambda kind, data=None: notified.append(kind)
                try:
                    loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
                    result = loop.run()
                finally:
                    session.close()
        self.assertEqual(result, "the retried final answer")
        self.assertEqual(loop.state, AgentLoop.DONE)
        self.assertEqual(DropHandler.stream_count, 2)
        self.assertIn("retry", notified)
        self.assertEqual(
            [m.text() for m in loop.messages if m.role == "assistant"], ["the retried final answer"]
        )
        all_text = "".join(m.text() for m in loop.messages)
        self.assertNotIn("partial-dropped-text", all_text)
        shared = "".join(m.text() for m in session.last_messages)
        self.assertEqual(shared.count("the retried final answer"), 1)


if __name__ == "__main__":
    unittest.main()
