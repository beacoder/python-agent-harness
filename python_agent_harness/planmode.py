"""Build/plan mode management.

Ported from gptel-agent-harness.el's build/plan mode section.

- Plan mode: read-only except the per-session plan file
  (<tmp>/python-agent-plans-<proj>-<rand>/PLAN.md).
- Switching to plan truncates the plan file and queues plan.md +
  plan-mode.md (${planInfo} -> plan file path) for injection into the
  next top-level request; switching back queues build-switch.md.
- Queued prompts are injected before the last user message (appended
  when the last message is a tool result) and consumed exactly once.
- Sub-agent requests in plan mode get the plan-mode reminder once per
  sub-agent loop.
"""

from __future__ import annotations

import os
import random
import string
import tempfile
from pathlib import Path

from . import config
from .models import AgentMode


def _plan_temp_dir() -> str:
    """Reliable temp dir (first candidate that is set, else /tmp)."""
    for d in (
        os.environ.get("TMPDIR"),
        os.environ.get("TMP"),
        os.environ.get("TEMP"),
        tempfile.gettempdir(),
    ):
        if d:
            return os.path.abspath(d)
    return "/tmp"


class PlanMode:
    """Per-session plan-mode state."""

    def __init__(self, project_dir: str) -> None:
        self.mode = AgentMode.BUILD
        self.project_dir = project_dir
        self.plan_file: str | None = None
        self.pending_prompts: list[str] = []

    # -- plan file lifecycle --------------------------------------------------
    def plan_temp_dir(self) -> str:
        return _plan_temp_dir()

    def plan_file_path(self) -> str:
        if self.plan_file:
            return self.plan_file
        proj_name = os.path.basename(os.path.normpath(self.project_dir))
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        d = os.path.join(self.plan_temp_dir(), f"python-agent-plans-{proj_name}-{suffix}")
        return os.path.join(d, config.PLAN_FILE_NAME)

    def ensure_plan_file(self) -> str:
        path = self.plan_file_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            Path(path).write_text("", encoding="utf-8")
        # Store the canonical path so the plan-mode write guard (which
        # compares against realpath'd tool targets) matches — on macOS
        # TMPDIR (/var) is a symlink to /private/var.
        self.plan_file = os.path.realpath(path)
        return self.plan_file

    def cleanup_plan_file(self) -> None:
        if not self.plan_file:
            return
        try:
            if os.path.exists(self.plan_file):
                os.remove(self.plan_file)
            d = os.path.dirname(self.plan_file)
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        except OSError:
            pass
        self.plan_file = None

    # -- mode switching ---------------------------------------------------------
    def set_mode(self, mode: AgentMode, prompts: dict[str, str]) -> None:
        """Set the mode; PROMPTS maps 'plan'/'plan-mode'/'build-switch' to text."""
        if mode == AgentMode.PLAN:
            plan_file = self.ensure_plan_file()
            # start each planning round from an empty file
            if self.plan_file and os.path.exists(plan_file):
                Path(plan_file).write_text("", encoding="utf-8")
            self.mode = mode
            self.plan_file = plan_file
            self.pending_prompts = [
                prompts["plan"],
                prompts["plan-mode"].replace("${planInfo}", plan_file),
            ]
        else:
            self.mode = AgentMode.BUILD
            self.pending_prompts = [prompts["build-switch"]]

    def consume_prompts(self) -> list[str]:
        prompts = self.pending_prompts
        self.pending_prompts = []
        return prompts

    # -- helpers ----------------------------------------------------------------
    @property
    def is_plan(self) -> bool:
        return self.mode == AgentMode.PLAN

    def plan_reminder(self) -> str:
        return config.PLAN_MODE_SUBAGENT_REMINDER.replace(
            "%s", self.plan_file or self.plan_file_path()
        )
