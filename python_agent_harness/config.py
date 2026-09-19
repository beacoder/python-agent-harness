"""Compatibility shim: moved to :mod:`python_agent_harness.session.config`.

Registered as a module alias in :data:`sys.modules` so patch targets like
``"python_agent_harness.config.MAX_NUDGES"`` mutate the live config module
rather than a stale copy of its values.
"""

import sys

from . import session as _session_pkg

sys.modules[__name__] = _session_pkg.config
