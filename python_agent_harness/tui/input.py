"""Input handling: SlashCompleter, UiQuestion, key bindings, and the
InputMixin that provides prompt reading and question blocking.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.text import Text

from .. import config

SLASH_COMMANDS = [
    "/plan",
    "/build",
    "/init",
    "/review",
    "/explain",
    "/compact",
    "/save",
    "/summary",
    "/sessions",
    "/restore",
    "/clear",
    "/model",
    "/exit",
]


def _custom_slash_commands() -> list[str]:
    from ..commands import load_custom_commands

    return sorted(f"/{c.name}" for c in load_custom_commands())


def _history_path() -> str:
    d = config.SESSION_DIR / "python-agent-harness"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / "input_history")


def _make_key_bindings() -> KeyBindings:
    """Esc+Enter (or Alt+Enter) submits; plain Enter inserts a newline.

    Tab triggers completion explicitly (first Tab inserts the common
    part / opens the menu, further Tabs cycle), Shift+Tab cycles
    backwards — prompt_toolkit's defaults don't reliably bind Tab in
    every mode/version.
    """
    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _submit(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    @kb.add("c-i")
    def _complete(event: Any) -> None:
        b = event.current_buffer
        if b.complete_state:
            b.complete_next()
        else:
            b.start_completion(insert_common_part=True)

    @kb.add("s-tab")
    def _complete_backward(event: Any) -> None:
        b = event.current_buffer
        if b.complete_state:
            b.complete_previous()
        else:
            b.start_completion(select_first=True)

    return kb


def _make_prompt_session(
    history: FileHistory, completer: Completer, **kwargs: Any
) -> PromptSession:
    """Create the TUI's input session.

    ``complete_while_typing`` is off on purpose: it races with Tab's
    ``start_completion`` (a keystroke-triggered completion can create
    the completion state just before the Tab-triggered task runs, which
    then bails out without inserting the common part).  Tab must be the
    single, deterministic trigger.
    """
    return PromptSession(
        history=history,
        key_bindings=_make_key_bindings(),
        completer=completer,
        complete_while_typing=False,
        multiline=True,
        enable_suspend=True,
        **kwargs,
    )


class SlashCompleter(Completer):
    """Tab-completion for the input line.

    - A first token starting with ``/`` completes against the known
      slash commands (builtins + custom commands from
      prompts/commands/*.md); if no command matches, it is treated as
      an absolute path.
    - After a slash command's space, Tab completes paths relative to
      the session's project dir (absolute and ``~`` paths work too).
    - Any other ``~``-prefixed or ``/``-containing token (e.g.
      ``~/wor``, ``docs/``) completes as a path: ``~`` against $HOME,
      otherwise relative to the project dir. Plain words without ``/``
      are left alone.
    - Directories get a trailing ``/`` so repeated Tab drills deeper;
      ``~`` alone completes to ``~/``.
    """

    def __init__(self, get_project_dir: Callable[[], str]) -> None:
        self.get_project_dir = get_project_dir

    def _slash_commands(self) -> list[str]:
        return sorted(set(SLASH_COMMANDS + _custom_slash_commands()))

    def _complete_paths(self, arg: str) -> Iterable[Completion]:
        expanded = os.path.expanduser(arg)
        if not arg:
            directory, prefix = self.get_project_dir() or os.getcwd(), ""
        elif expanded.endswith(os.sep):
            base = (
                expanded
                if os.path.isabs(expanded)
                else os.path.join(self.get_project_dir() or os.getcwd(), expanded)
            )
            directory, prefix = base, ""
        elif os.path.isdir(expanded):
            # "~" or an existing dir without a trailing slash: complete
            # the trailing slash itself (bash-style), not its siblings.
            yield Completion(text="/", start_position=0, display=arg + "/")
            return
        else:
            base = (
                expanded
                if os.path.isabs(expanded)
                else os.path.join(self.get_project_dir() or os.getcwd(), expanded)
            )
            directory, prefix = os.path.dirname(base), os.path.basename(base)
        try:
            entries = sorted(os.listdir(directory or "."))
        except OSError:
            return
        for name in entries:
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if os.path.isdir(os.path.join(directory, name)):
                suffix += "/"
                display = name + "/"
            else:
                display = name
            # start_position=0 appends at the cursor; the typed prefix is
            # already in the buffer, so only the remaining suffix is inserted.
            yield Completion(text=suffix, start_position=0, display=display)

    def get_completions(self, document: Any, complete_event: Any):
        text = document.text_before_cursor
        if text.startswith("/"):
            if " " not in text:
                cmds = [c for c in self._slash_commands() if c.startswith(text)]
                for cmd in cmds:
                    yield Completion(cmd, start_position=-len(text))
                if cmds:
                    return
                yield from self._complete_paths(text)  # absolute path
                return
            arg = text.split(" ", 1)[1]
            yield from self._complete_paths(arg)
            return
        token = text.rsplit(" ", 1)[-1] if " " in text else text
        if token.startswith("~") or "/" in token:
            yield from self._complete_paths(token)


class UiQuestion:
    def __init__(
        self,
        prompt: str,
        multiple: bool = False,
        options: list[str] | None = None,
        custom: bool = True,
        keys: list[str] | None = None,
    ) -> None:
        self.prompt = prompt
        self.multiple = multiple
        self.options = options or []
        self.custom = custom
        # keyed choices (e.g. ["y", "n"] for a confirm): render the
        # options as a keyed list and resolve typed keys to labels,
        # instead of the numbered-list style of the Question tool
        self.keys = keys or []
        self.answer: str | None = None
        self.event = threading.Event()


def _resolve_keyed_choice(answer: str, options: list[str], keys: list[str]) -> str:
    """Map bare keys in ANSWER to the matching option label.

    Comma-separated keys pick several options (multiple select);
    non-key tokens pass through unchanged as free-text answers.
    """
    if not options or not keys or not answer.strip():
        return answer
    resolved: list[str] = []
    for part in answer.split(","):
        part = part.strip()
        if part.lower() in keys:
            resolved.append(options[keys.index(part.lower())])
            continue
        resolved.append(part)
    return ", ".join(resolved)


def _resolve_numbered_choice(answer: str, options: list[str]) -> str:
    """Map bare numbers in ANSWER (1-based) to the matching option label.

    Comma-separated numbers pick several options (multiple select);
    non-numeric tokens pass through unchanged as free-text answers;
    out-of-range numbers are kept as typed.  Empty answers stay empty.
    """
    if not options or not answer.strip():
        return answer
    resolved: list[str] = []
    for part in answer.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(options):
                resolved.append(options[idx - 1])
                continue
        resolved.append(part)
    return ", ".join(resolved)


class InputMixin:
    """Input handling methods for the TUI.

    Expects the host class to provide: ``session``, ``console``,
    ``question``, ``prompt_session``, ``_data_event``.
    """

    def _input_prompt(self) -> FormattedText:
        """Styled input prompt: short model name + caret.

        Shows the model actually in use (the part after the last '/',
        e.g. ``deepseek-ai/deepseek-flash-v4`` → ``deepseek-flash-v4``)
        so the active model stays visible while typing.
        """
        model = self.session.model or ""
        short = model.rsplit("/", 1)[-1] if "/" in model else model
        return FormattedText(
            [
                ("bold cyan", f"{short} " if short else ""),
                ("ansibrightblack", "> "),
            ]
        )

    def _read_multiline(self) -> str | None:
        try:
            with patch_stdout():
                text = self.prompt_session.prompt(self._input_prompt())
        except EOFError:
            # Ctrl-D: quit
            return None
        except KeyboardInterrupt:
            # Ctrl-C: cancel this input, stay in the app
            self.console.print("[dim]input cancelled[/dim]")
            return ""
        return text

    def _ask_question_blocking(self) -> None:
        q = self.question
        if q is None:
            return
        self.console.print(self._render_frame())
        self.console.print()
        self._flush()
        options = q.options or []
        keys = q.keys or []
        if keys and options and len(keys) == len(options):
            # keyed choices (e.g. y/n confirm): type the key to pick —
            # same list look as the Question tool, keys instead of numbers
            self.console.print(Text(q.prompt))
            for key, opt in zip(keys, options, strict=True):
                line = Text(f"  {key}) ", style="cyan")
                line.append(opt)
                self.console.print(line)
            hint = "Enter keys, comma-separated" if q.multiple else "Enter a key"
            if q.custom:
                hint += ", or type your own answer"
            self.console.print(f"[dim]{hint}[/dim]")
            prompt = "> "
        elif options:
            # option labels get a numbered list: type the number to pick
            self.console.print(Text(q.prompt))
            for i, opt in enumerate(options, 1):
                line = Text(f"  {i}) ", style="cyan")
                line.append(opt)
                self.console.print(line)
            hint = "Enter numbers, comma-separated" if q.multiple else "Enter a number"
            if q.custom:
                hint += ", or type your own answer"
            self.console.print(f"[dim]{hint}[/dim]")
            prompt = "> "
        else:
            prompt = q.prompt + " > "
        try:
            with patch_stdout():
                answer = self.prompt_session.prompt(prompt, multiline=False)
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if keys:
            q.answer = _resolve_keyed_choice(answer, options, keys)
        else:
            q.answer = _resolve_numbered_choice(answer, options)
        q.event.set()
        self.question = None
        self._data_event.set()  # re-render promptly after the answer

    def _ask_sync(self, q: UiQuestion) -> str:
        """Block the worker thread until the main thread answers.

        Cancel-aware: the wait polls the session cancel event, so a
        Ctrl-C outside the answer prompt (e.g. during the render loop)
        unblocks the worker immediately instead of wedging it until a
        question is answered.  A cancelled run returns an empty answer.
        """
        self.question = q
        cancel = getattr(self.session, "cancel_event", None)
        while not q.event.wait(0.1):
            if cancel is not None and cancel.is_set():
                return ""
        return q.answer or ""

    def _ui_confirm(self, prompt: str) -> bool:
        """PlanExit confirmation: same look as the Question tool, but a
        y/n keyed choice list instead of numbers (two choices only)."""
        q = UiQuestion(
            prompt,
            options=list(config.PLAN_EXIT_OPTIONS),
            keys=["y", "n"],
            custom=False,
        )
        answer = self._ask_sync(q).strip().lower()
        # resolved answers arrive as the option label; legacy free-text
        # (y/yes/a/1/true) keeps working for muscle memory
        return answer == config.PLAN_EXIT_OPTIONS[0].lower() or answer in (
            "y",
            "yes",
            "a",
            "true",
            "1",
        )

    def _ui_ask(self, questions: list[dict]) -> str:
        lines = []
        for q in questions:
            prompt = q.get("question", "")
            options = q.get("options") or []
            multiple = bool(q.get("multiple"))
            custom = q.get("custom", True)
            ui_q = UiQuestion(
                prompt,
                multiple=multiple,
                options=list(options),
                custom=custom,
            )
            answer = self._ask_sync(ui_q)
            if multiple:
                answer = ", ".join(a.strip() for a in answer.split(",") if a.strip())
            lines.append(f'"{prompt}" = "{answer}"')
        return "\n".join(lines) if lines else "Unanswered"
