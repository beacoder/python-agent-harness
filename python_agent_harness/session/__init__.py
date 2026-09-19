"""Session management and configuration.

Re-exports the public surface of ``session.session`` and
``session.config``.  Attributes resolve lazily (PEP 562): importing
this package stays cheap and cycle-free, mirroring the historical
``python_agent_harness.session`` module layout.
"""

import importlib as _importlib
from typing import Any


def _session() -> Any:
    return _importlib.import_module("python_agent_harness.session.session")


def _config() -> Any:
    return _importlib.import_module("python_agent_harness.session.config")


def __getattr__(name: str) -> Any:
    if name == "config":
        return _config()
    for mod in (_session(), _config()):
        try:
            return getattr(mod, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(dir(_session())) | set(dir(_config())))
