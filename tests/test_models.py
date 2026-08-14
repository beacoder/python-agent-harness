"""Data-model unit tests: Message text extraction / serialization."""

import json
import unittest

from python_agent_harness.models import Message, ToolCall, ToolSpec, Usage


class TestMessageText(unittest.TestCase):
    """Message.text() flattens multimodal content lists the same way the
    client does: strings, text parts and thinking parts joined; anything
    else skipped."""

    def test_list_content_parts_flattened(self):
        m = Message(
            role="user",
            content=[
                "plain ",
                {"type": "text", "text": "text part "},
                {"type": "thinking", "thinking": "inner thought"},
                {"type": "image", "image_url": "x"},
            ],
        )
        self.assertEqual(m.text(), "plain text part inner thought")

    def test_dict_without_text_or_thinking_skipped(self):
        m = Message(role="user", content=[{"type": "image", "url": "x"}])
        self.assertEqual(m.text(), "")

    def test_list_with_no_text_parts_is_empty(self):
        m = Message(role="user", content=[{"role": "x"}])
        self.assertEqual(m.text(), "")

    def test_none_content_is_empty(self):
        self.assertEqual(Message(role="user").text(), "")

    def test_empty_list_content_is_empty(self):
        m = Message(role="user", content=[])
        self.assertEqual(m.text(), "")


class TestMessageToApi(unittest.TestCase):
    def test_tool_calls_with_dict_arguments_json_encoded(self):
        m = Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="Bash", arguments={"command": "ls"})],
        )
        api = m.to_api()
        self.assertEqual(
            json.loads(api["tool_calls"][0]["function"]["arguments"]),
            {"command": "ls"},
        )

    def test_tool_call_id_and_name_serialized(self):
        m = Message(role="tool", content="result", tool_call_id="c1", name="Bash")
        api = m.to_api()
        self.assertEqual(api["tool_call_id"], "c1")
        self.assertEqual(api["name"], "Bash")

    def test_none_content_omitted(self):
        api = Message(role="assistant").to_api()
        self.assertEqual(api, {"role": "assistant"})

    def test_reasoning_prefix_stripped_from_api_content(self):
        """The client merges reasoning ahead of the answer into content;
        to_api() must not re-send that reasoning over the wire."""
        reasoning = "Let me think about this carefully."
        m = Message(
            role="assistant",
            content=reasoning + "\n\nThe answer is 42.",
            reasoning=reasoning,
        )
        # stored content keeps the reasoning (TUI strips it for display)
        self.assertTrue(m.content.startswith(reasoning))
        # wire content drops it
        self.assertEqual(m.to_api()["content"], "The answer is 42.")

    def test_reasoning_stripped_when_not_exact_prefix(self):
        """Leading whitespace/newlines before the reasoning block are
        tolerated, and the newline separator after it is removed."""
        reasoning = "thinking"
        m = Message(
            role="assistant",
            content="\n\n" + reasoning + "\n\nanswer",
            reasoning=reasoning,
        )
        self.assertEqual(m.to_api()["content"], "answer")

    def test_no_reasoning_leaves_content_untouched(self):
        m = Message(role="assistant", content="plain answer")
        self.assertEqual(m.to_api()["content"], "plain answer")

    def test_reasoning_only_content_becomes_empty_over_wire(self):
        """A reasoning-only message (no answer) sends empty content, not
        the reasoning text."""
        reasoning = "just thinking, no answer"
        m = Message(role="assistant", content=reasoning, reasoning=reasoning)
        self.assertEqual(m.to_api()["content"], "")


class TestToolParameters(unittest.TestCase):
    def test_tool_parameters_default_is_a_plain_dict(self):
        """Tool is an ABC, not a dataclass: the parameters default must
        be a real dict, not a dataclasses.field() sentinel (which would
        break JSON serialization for any tool that forgot to override
        it)."""
        import dataclasses

        from python_agent_harness.tools.base import Tool

        self.assertIsInstance(Tool.parameters, dict)
        self.assertFalse(isinstance(Tool.parameters, dataclasses.Field))


class TestToolSpecAndUsage(unittest.TestCase):
    def test_tool_spec_to_api(self):
        spec = ToolSpec(name="Read", description="read a file", parameters={"type": "object"})
        api = spec.to_api()
        self.assertEqual(api["type"], "function")
        self.assertEqual(api["function"]["name"], "Read")
        self.assertEqual(api["function"]["parameters"], {"type": "object"})

    def test_usage_defaults(self):
        u = Usage()
        self.assertEqual((u.input_tokens, u.output_tokens), (0, 0))


if __name__ == "__main__":
    unittest.main()
