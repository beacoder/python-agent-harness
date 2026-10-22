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
"""

import os
import tempfile

os.environ["PYTHON_AGENT_HARNESS_CONFIG"] = os.path.join(
    tempfile.gettempdir(), "python-agent-harness-tests-no-such-config.json"
)
