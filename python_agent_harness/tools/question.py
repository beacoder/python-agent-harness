"""Question tool: ask the user one or more questions during execution.

Synchronous (mirrors gptel's Question tool, which is NOT ``:async t``):
``run`` blocks until the user answers and returns the answers as a
plain string — it executes one at a time, in call order, like every
other non-Bash/non-Agent tool.
"""

from __future__ import annotations

from .base import Tool, ToolContext

DESCRIPTION = (
    "Ask the user one or more questions during execution.\n\n"
    "Use this tool when you need to:\n"
    "1. Gather user preferences or requirements\n"
    "2. Clarify ambiguous instructions\n"
    "3. Get decisions on implementation choices as you work\n"
    "4. Offer choices to the user about what direction to take\n\n"
    "Each question can have predefined options for the user to select from. "
    "By default, a \"Type your own answer\" option is added; set custom to "
    "false to disable it. Set multiple to true to allow selecting more than "
    "one option.\n\n"
    "If no options are provided, the user will be prompted for free-text "
    "input.\n\n"
    "If you recommend a specific option, make that the first option in the "
    "list and add \"(Recommended)\" at the end of the label."
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "multiple": {"type": "boolean"},
                    "custom": {"type": "boolean"},
                },
                "required": ["question"],
            },
        }
    },
    "required": ["questions"],
}


class Question(Tool):
    name = "Question"
    description = DESCRIPTION
    parameters = PARAMETERS

    def run(self, args: dict, ctx: ToolContext) -> str:
        raw = args.get("questions")
        if isinstance(raw, list):
            questions = raw
        elif isinstance(raw, dict) and isinstance(raw.get("questions"), list):
            questions = raw["questions"]
        else:
            return "Error: questions must be an array"

        # containment boundary: a failure in the interactive prompt
        # becomes an error string for the model, never a crash
        try:
            return ctx.ask_questions(questions)
        except Exception as e:  # noqa: BLE001 - error string for the model
            return f"Error: Question failed — {e}"
