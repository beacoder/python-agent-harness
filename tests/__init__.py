"""tests package: mirrors python_agent_harness subpackage layout.

Discovered with ``unittest discover -t . -s tests`` (repo root on
sys.path) so subpackages import as ``tests.<pkg>`` instead of shadowing
stdlib/SDK names (``io``, ``mcp``).

Hermetic config: every code path that resolves the config file without
an explicit path (e.g. a Bash call through ``ToolContext()`` reading
``bash.env_passthrough``) falls back to $PYTHON_AGENT_HARNESS_CONFIG
before the user's real ~/.config file.  Point it at a path that never
exists so the suite never reads the developer's own config (or API
key).  Tests that exercise the resolution order set/clear it themselves.

Hermetic spool dir: ``Session.__init__`` sweeps stale spool files from
``filesystem._spool_dir()``, which defaults to the real temp dir.  Left
alone, every ``Session(...)`` in the suite would delete the developer's
own stranded spool files.  Point TMPDIR at a throwaway dir for the whole
test process — ``_spool_dir()`` prefers it, and the tests that exercise
its candidate order patch the environment themselves.
"""

import atexit
import os
import shutil
import tempfile

os.environ["PYTHON_AGENT_HARNESS_CONFIG"] = os.path.join(
    tempfile.gettempdir(), "python-agent-harness-tests-no-such-config.json"
)

_spool_sandbox = tempfile.mkdtemp(prefix="pah-spool-guard-")
os.environ["TMPDIR"] = _spool_sandbox
# gettempdir() caches on first use; drop any value cached before TMPDIR was set
tempfile.tempdir = None


def _restore_spool_dir() -> None:
    shutil.rmtree(_spool_sandbox, ignore_errors=True)


atexit.register(_restore_spool_dir)
