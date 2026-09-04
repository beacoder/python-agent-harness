"""Guard: the test suite must never write into the user's real
session/input-history storage.

A bare ``Session(...)`` (built directly by several test modules instead
of via ``RecordingSession``) auto-saves conversations to
``config.SESSION_DIR`` and generates an LLM-style title by renaming the
session file — previously landing ``SYNC_*.md`` / ``fakeproj_*.md`` junk
in the real ``~/.local/share/python-agent-harness/sessions/``.

Mirrors the ``plan_cleanup.py`` pattern: imported (for side effects) by
every test module that can build a session — it redirects
``config.SESSION_DIR`` to a fresh temp dir for the whole test process
and restores the original at exit.  Sticky by design (no per-test
restore), so tests that save/restore ``SESSION_DIR`` around their own
assignment (test_persistence, test_tui_run, ...) cannot leave it
pointed at production afterwards.
"""

import atexit
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)

from python_agent_harness import config

if os.environ.get("PYTHON_AGENT_HARNESS_TEST_KEEP_SESSIONS") != "1":
    _orig_session_dir = config.SESSION_DIR
    # Path, not str: persistence.session_dir() uses SESSION_DIR / SUBDIR
    config.SESSION_DIR = Path(tempfile.mkdtemp(prefix="pah-sessions-guard-"))

    def _restore() -> None:
        config.SESSION_DIR = _orig_session_dir

    atexit.register(_restore)
