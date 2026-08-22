"""Automatic cleanup of /tmp plan dirs created by the test suite.

``PlanMode.ensure_plan_file()`` puts PLAN.md in a fresh
``python-agent-plans-*`` dir under the temp dir (usually /tmp).
Production ``Session.close()`` intentionally keeps the plan file
(one unique file per session), so tests must clean up their own:
this module is imported (for its side effects) by the test files that
can create plan files — it records every plan file created by
``ensure_plan_file`` and removes them after each test (via a
``unittest.TestCase.run`` hook) with an atexit safety net.
"""

import atexit
import os
import unittest

from python_agent_harness.planmode import PlanMode

_created: list[str] = []

_orig_ensure = PlanMode.ensure_plan_file


def _tracked_ensure(self) -> str:
    path = _orig_ensure(self)
    _created.append(path)
    return path


def _cleanup_plan_files() -> None:
    for path in _created:
        try:
            os.remove(path)
            d = os.path.dirname(path)
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        except OSError:
            pass
    _created.clear()


_orig_run = unittest.TestCase.run


def _run(self, result=None):
    try:
        return _orig_run(self, result)
    finally:
        _cleanup_plan_files()


PlanMode.ensure_plan_file = _tracked_ensure
unittest.TestCase.run = _run
atexit.register(_cleanup_plan_files)
