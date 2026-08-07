"""Safety layer: forbidden paths, bash policy tiers, plan-mode read-only.

Ported from gptel-agent-harness-safety.el.

Violations are always hard blocks delivered as error strings to the
model — never warn-and-continue.

Bash refusal tiers (checked in this order):
1. forbidden path token in command          -> always refused
2. catastrophic pattern                     -> always refused (before plan gate)
3. plan mode                                -> run only read-only commands
4. destructive pattern                      -> run unless approval == block
5. dangerous pattern                        -> verdict: session allow/deny,
                                               confirm (4-way y/n/a/d), block
6. otherwise                                -> run
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field

from . import config


class SafetyViolation(Exception):
    """Raised for path-level violations (file tools)."""


# ---------------------------------------------------------------------------
# Path guards
# ---------------------------------------------------------------------------

def _canonical(path: str) -> str:
    """Expand ~ and resolve //, /./, /../ and symlinks (best effort).

    ``os.path.realpath`` normalizes the path so syntactic tricks
    (``//mnt/x``, ``/tmp/../mnt/x``, a symlink pointing into a
    forbidden tree) cannot evade the forbidden-path patterns.
    """
    expanded = _expand(path)
    try:
        return os.path.realpath(expanded)
    except (OSError, ValueError):
        return os.path.normpath(expanded)


def path_forbidden(path: str) -> str | None:
    """Return the first matching forbidden-path regexp, or None.

    The path is matched as given AND canonicalized; the trailing-
    separator form is checked too so the bare directory itself
    (``/mnt``) is caught by patterns like ``^\\s*/mnt/``.
    """
    if not path:
        # empty tokens (e.g. from a command starting with a quote) must
        # not resolve to the process cwd and false-positive
        return None
    canonical = _canonical(path)
    for cand in (path, canonical, canonical + os.sep):
        for pattern in config.FORBIDDEN_PATHS:
            if re.search(pattern, cand):
                return pattern
    return None


def command_forbidden(command: str) -> str | None:
    """Check forbidden path patterns against command tokens."""
    expanded = _expand(command)
    for pattern in config.FORBIDDEN_PATHS:
        if re.search(pattern, expanded):
            return pattern
    for token in _split_tokens(command):
        if path_forbidden(token):
            return token
    return None


def check_path(path: str, tool_name: str) -> None:
    """Raise SafetyViolation if PATH is forbidden."""
    pattern = path_forbidden(path)
    if pattern:
        raise SafetyViolation(
            f"Error: {tool_name} blocked by harness safety — path {path} "
            f"matches forbidden pattern {pattern!r}"
        )


def _expand(path: str) -> str:
    return re.sub(r"^~", str(__import__("pathlib").Path.home()), path)


def _split_tokens(command: str) -> list[str]:
    return re.split(r"[ \t\n\r;&|<>()\"']+", command)


# ---------------------------------------------------------------------------
# Plan-mode read-only
# ---------------------------------------------------------------------------

_GIT_VALUE_OPTIONS = {
    "-c", "-C", "--git-dir", "--git-common-dir", "--work-tree",
    "--namespace", "--config-env", "--exec-path",
}


def _git_subcommand(tokens: list[str]) -> str | None:
    """Return the git subcommand, skipping global options and their values.

    Handles both ``-C path`` (separate value) and ``--git-dir=path``
    (inline) forms, so ``git -C /tmp/x push`` and
    ``git --git-dir=/tmp/x push`` resolve to ``push`` instead of
    slipping past the mutating-subcommand check.
    """
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("-"):
            return tok.lower()
        if "=" in tok:
            i += 1
            continue
        if tok in _GIT_VALUE_OPTIONS:
            i += 2  # option + its value
            continue
        i += 1
    return None


def bash_mutating_p(command: str) -> bool:
    """True if COMMAND mutates state (downcased analysis)."""
    lowered = command.lower()
    if ">>" in lowered:
        return True
    if re.search(r"(?<!2)>\s*(?!&)", lowered) and "2>&1" not in lowered:
        return True
    for word in ("tee", "xargs", "sudo"):
        if re.search(rf"\b{word}\b", lowered):
            return True
    tokens = shlex.split(command)
    if tokens and tokens[0].lower() == "git":
        sub = _git_subcommand(tokens)
        if sub and sub in config.GIT_MUTATING_SUBCOMMANDS:
            return True
    return False


def bash_has_subshell_p(command: str) -> bool:
    return "$(" in command or "`" in command or "<(" in command or ">(" in command


def bash_segments(command: str) -> list[str]:
    """Quote-unaware split on && || | ; & \\n — fail-closed."""
    parts = re.split(r"&&|\|\||\||;|&|\n", command)
    return [p for p in parts if p.strip()]


def bash_first_command(segment: str) -> str:
    """Return the first real command word, skipping env assignments, cd, time."""
    tokens = shlex.split(segment)
    for tok in tokens:
        if tok in ("time",):
            continue
        if "=" in tok and not tok.startswith("-") and tokens.index(tok) == 0:
            continue
        return tok
    return ""


def bash_arg_denylisted(segment: str) -> bool:
    tokens = shlex.split(segment)
    if not tokens:
        return False
    cmd = tokens[0].lower()
    for name, bad_args in config.BASH_ARG_DENYLIST:
        if cmd == name:
            for tok in tokens[1:]:
                if tok in bad_args:
                    return True
    return False


def bash_read_only_p(command: str) -> bool:
    """True iff COMMAND is safe to run in plan mode."""
    if bash_has_subshell_p(command):
        return False
    for segment in bash_segments(command):
        first = bash_first_command(segment)
        if not first:
            continue
        if bash_arg_denylisted(segment):
            return False
        if bash_mutating_p(segment):
            return False
        if first not in config.PLAN_READONLY_COMMANDS:
            return False
    return True


# ---------------------------------------------------------------------------
# Bash policy
# ---------------------------------------------------------------------------

@dataclass
class BashPolicy:
    """Evaluates a bash command against the refusal tiers."""

    approval: str = config.BASH_APPROVAL  # "nil" | "confirm" | "block"
    timeout: int = config.BASH_TIMEOUT
    session_allow: set[str] = field(default_factory=set)
    session_deny: set[str] = field(default_factory=set)
    confirm_allowed: bool = True  # mirrors gptel-confirm-tool-calls opt-out
    plan_mode: bool = False

    def verdict(self, command: str) -> str | None:
        """Return an error string to deliver, or None to run."""
        # tier 1: forbidden path tokens
        pat = command_forbidden(command)
        if pat:
            return (
                f"Error: Bash blocked by harness safety — command references "
                f"forbidden path pattern {pat!r}"
            )
        lowered = command.lower()
        # tier 2: catastrophic (before plan gate)
        for pattern in config.CATASTROPHIC_PATTERNS:
            if re.search(pattern, lowered):
                return (
                    "Error: Bash blocked by harness safety — command matches "
                    "a catastrophic pattern and is never allowed."
                )
        # tier 3: plan mode read-only
        if self.plan_mode:
            if not bash_read_only_p(command):
                return (
                    "Error: Bash blocked by harness safety — command is not "
                    "read-only and plan mode is active."
                )
            return None
        # tier 4: destructive (never prompts)
        destructive = any(
            re.search(p, lowered) for p in config.DESTRUCTIVE_PATTERNS
        )
        if destructive:
            if self.approval == "block":
                return (
                    "Error: Bash command rejected by user approval "
                    "(blocked for this session)."
                )
            return None
        # tier 5: dangerous -> verdict
        dangerous = any(
            re.search(p, lowered) for p in config.DANGEROUS_PATTERNS
        )
        if dangerous:
            return self._dangerous_verdict(command)
        return None

    def _dangerous_verdict(self, command: str) -> str | None:
        if command in self.session_allow:
            return None
        if command in self.session_deny:
            return "Error: Bash command rejected by user approval (denied for this session)."
        if self.approval == "block":
            return "Error: Bash command rejected by user approval (blocked for this session)."
        if self.approval != "confirm":
            return None
        if not self.confirm_allowed:
            return None
        return "CONFIRM"  # sentinel: caller must prompt the user

    def record(self, command: str, answer: str) -> None:
        """Record a user verdict: allow/deny (session) or run/deny-once."""
        if answer == "allow":
            self.session_allow.add(command)
        elif answer == "deny":
            self.session_deny.add(command)
