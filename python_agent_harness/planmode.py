"""Compatibility alias for :mod:`python_agent_harness.core.planmode`.

Registered in :data:`sys.modules` so that patch targets such as
``"python_agent_harness.planmode.X"`` mutate the live module.
"""

import importlib as _importlib
import sys as _sys

_mod = _importlib.import_module("python_agent_harness.core.planmode")
_sys.modules[__name__] = _mod
