import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.chat import (
    ChatSession,
    Role,
    build_tools_system_message,
    parse_tool_calls,
)
from server.config import settings
from server.services import _is_internal_tool


def _make_mcp_tool(name: str, description: str = "", params: dict | None = None):
    """Build a minimal MCP Tool mock with the given name and input schema."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = {
        "type": "object",
        "properties": params or {},
        "required": list((params or {}).keys()),
    }
    return tool


def _make_tool_result(text: str):
    content_item = MagicMock()
    content_item.text = text
    result = MagicMock()
    result.content = [content_item]
    return result


def _make_empty_tool_result():
    result = MagicMock()
    result.content = []
    return result


def _stub_call_tool(tool_responses: dict):
    async def _call_tool(name, *, arguments=None, **kwargs):
        text = tool_responses.get(name, "")
        if text == "":
            return _make_empty_tool_result()
        return _make_tool_result(text)

    return AsyncMock(side_effect=_call_tool)


def _make_session_with_tools(tools, **kwargs) -> ChatSession:
    mcp_session = AsyncMock()
    mcp_session.call_tool = AsyncMock(return_value=_make_tool_result(""))
    ollama_client = MagicMock()
    return ChatSession(
        session_id=kwargs.pop("session_id", uuid.uuid4()),
        mcp_session=kwargs.pop("mcp_session", mcp_session),
        ollama_client=kwargs.pop("ollama_client", ollama_client),
        thread_pool=kwargs.pop("thread_pool", ThreadPoolExecutor()),
        mcp_tools=tools,
        **kwargs,
    )


class TestParseToolCalls:
    def test_returns_empty_for_plain_text(self):
        assert parse_tool_calls("Hello there!") == []

    def test_returns_empty_for_invalid_json(self):
        assert parse_tool_calls('{"toolCalls": [bad json}') == []

    def test_returns_empty_when_no_tool_calls_key(self):
        assert parse_tool_calls('{"message": "hi"}') == []

    def test_returns_empty_when_tool_calls_not_a_list(self):
        assert parse_tool_calls('{"toolCalls": "not a list"}') == []

    def test_parses_single_valid_tool_call(self):
        payload = json.dumps(
            {
                "toolCalls": [
                    {
                        "id": "c1",
                        "name": "search_knowledge_base",
                        "arguments": {"query": "foo"},
                    }
                ]
            }
        )
        calls = parse_tool_calls(payload)
        assert len(calls) == 1
        assert calls[0].id == "c1"
        assert calls[0].name == "search_knowledge_base"
        assert calls[0].arguments == {"query": "foo"}

    def test_parses_multiple_tool_calls(self):
        payload = json.dumps(
            {
                "toolCalls": [
                    {"id": "c1", "name": "tool_a", "arguments": {}},
                    {"id": "c2", "name": "tool_b", "arguments": {"x": 1}},
                ]
            }
        )
        calls = parse_tool_calls(payload)
        assert len(calls) == 2
        assert calls[0].name == "tool_a"
        assert calls[1].name == "tool_b"

    def test_skips_entries_missing_name(self):
        payload = json.dumps({"toolCalls": [{"id": "c1", "arguments": {}}]})
        assert parse_tool_calls(payload) == []

    def test_skips_entries_with_non_dict_arguments(self):
        payload = json.dumps(
            {"toolCalls": [{"id": "c1", "name": "tool", "arguments": "bad"}]}
        )
        assert parse_tool_calls(payload) == []

    def test_auto_generates_id_when_missing(self):
        payload = json.dumps({"toolCalls": [{"name": "some_tool", "arguments": {}}]})
        calls = parse_tool_calls(payload)
        assert len(calls) == 1
        assert calls[0].id  # non-empty auto-generated ID

    def test_json_with_leading_whitespace_is_parsed(self):
        payload = "  " + json.dumps(
            {"toolCalls": [{"id": "c1", "name": "t", "arguments": {}}]}
        )
        calls = parse_tool_calls(payload)
        assert len(calls) == 1

    def test_accepts_snake_case_tool_calls_key(self):
        payload = json.dumps(
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "search_knowledge_base",
                        "arguments": {"query": "foo"},
                    }
                ]
            }
        )
        calls = parse_tool_calls(payload)
        assert len(calls) == 1
        assert calls[0].name == "search_knowledge_base"

    def test_accepts_snake_case_tool_name_key(self):
        payload = json.dumps(
            {
                "toolCalls": [
                    {
                        "id": "c1",
                        "tool_name": "search_knowledge_base",
                        "arguments": {"query": "foo"},
                    }
                ]
            }
        )
        calls = parse_tool_calls(payload)
        assert len(calls) == 1
        assert calls[0].name == "search_knowledge_base"

    def test_accepts_snake_case_tool_arguments_key(self):
        payload = json.dumps(
            {
                "toolCalls": [
                    {
                        "id": "c1",
                        "name": "search_knowledge_base",
                        "tool_arguments": {"query": "bar"},
                    }
                ]
            }
        )
        calls = parse_tool_calls(payload)
        assert len(calls) == 1
        assert calls[0].arguments == {"query": "bar"}

    def test_accepts_fully_snake_case_response(self):
        """Models that ignore casing instructions should still have their tool calls parsed."""
        payload = json.dumps(
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "tool_name": "search_knowledge_base",
                        "tool_arguments": {"query": "prime minister of britain"},
                    }
                ]
            }
        )
        calls = parse_tool_calls(payload)
        assert len(calls) == 1
        assert calls[0].name == "search_knowledge_base"
        assert calls[0].arguments == {"query": "prime minister of britain"}

    def test_parses_json_wrapped_in_generic_code_fence(self):
        """Model output wrapped in ``` fences is still parsed correctly."""
        inner = json.dumps({"toolCalls": [{"id": "c1", "name": "t", "arguments": {}}]})
        payload = f"```\n{inner}\n```"
        calls = parse_tool_calls(payload)
        assert len(calls) == 1
        assert calls[0].name == "t"

    def test_parses_json_wrapped_in_json_code_fence(self):
        """Model output wrapped in ```json fences is still parsed correctly."""
        inner = json.dumps({"toolCalls": [{"id": "c1", "name": "t", "arguments": {}}]})
        payload = f"```json\n{inner}\n```"
        calls = parse_tool_calls(payload)
        assert len(calls) == 1

    def test_parses_json_wrapped_in_tool_code_fence(self):
        """Model output wrapped in ```tool_code fences (as seen in the wild) is parsed."""
        inner = json.dumps(
            {
                "toolCalls": [
                    {
                        "id": "call_1",
                        "name": "search_knowledge_base",
                        "arguments": {"query": "Prime Minister of Britain"},
                    }
                ]
            }
        )
        payload = f"```tool_code\n{inner}\n```"
        calls = parse_tool_calls(payload)
        assert len(calls) == 1
        assert calls[0].name == "search_knowledge_base"
        assert calls[0].arguments == {"query": "Prime Minister of Britain"}


class TestBuildToolsSystemMessage:
    def test_returns_empty_string_for_no_tools(self):
        assert build_tools_system_message([]) == ""

    def test_contains_tool_name(self):
        tool = _make_mcp_tool("search_knowledge_base", "Search docs")
        msg = build_tools_system_message([tool])
        assert "search_knowledge_base" in msg

    def test_contains_tool_description(self):
        tool = _make_mcp_tool("my_tool", "Does something important")
        msg = build_tools_system_message([tool])
        assert "Does something important" in msg

    def test_contains_json_format_instructions(self):
        tool = _make_mcp_tool("t", "d")
        msg = build_tools_system_message([tool])
        assert "toolCalls" in msg

    def test_lists_all_tool_names(self):
        tools = [
            _make_mcp_tool("alpha"),
            _make_mcp_tool("beta"),
            _make_mcp_tool("gamma"),
        ]
        msg = build_tools_system_message(tools)
        assert "alpha" in msg
        assert "beta" in msg
        assert "gamma" in msg


@pytest.mark.asyncio
class TestToolCallingLoop:
    async def test_no_tool_calls_when_mcp_tools_empty(self):
        """Without mcp_tools the loop is skipped and Ollama is called once via stream."""
        session = _make_session_with_tools(tools=[])
        session.mcp_session.call_tool = _stub_call_tool({})
        session.client.chat.return_value = iter(
            [{"message": {"content": "direct answer"}}]
        )

        stream = await session.chat("hello")
        tokens = [t async for t in stream]
        # stream=True call, not the non-streaming probe
        call_kwargs = session.client.chat.call_args[1]
        assert call_kwargs["stream"] is True

    async def test_plain_text_response_is_streamed_directly(self):
        """When the LLM responds without tool calls, its text is streamed as-is."""
        tool = _make_mcp_tool("search_knowledge_base", "Search")
        session = _make_session_with_tools(tools=[tool])
        session.mcp_session.call_tool = _stub_call_tool({})

        # First call (probe, stream=False) returns plain text
        probe_response = {"message": {"content": "Here is a plain answer."}}
        session.client.chat.return_value = probe_response

        stream = await session.chat("what is X?")
        tokens = [t async for t in stream]

        # Should stream the pre-resolved text character by character
        assert "".join(tokens) == "Here is a plain answer."

    async def test_tool_call_result_is_fed_back_to_llm(self):
        """A tool call response triggers execution and a second LLM call."""
        tool = _make_mcp_tool(
            "search_knowledge_base", "Search", {"query": {"type": "string"}}
        )
        session = _make_session_with_tools(tools=[tool])

        tool_call_json = json.dumps(
            {
                "toolCalls": [
                    {
                        "id": "c1",
                        "name": "search_knowledge_base",
                        "arguments": {"query": "foo"},
                    }
                ]
            }
        )
        plain_answer = "Based on the docs, the answer is 42."

        call_count = 0

        def _chat_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"message": {"content": tool_call_json}}
            return {"message": {"content": plain_answer}}

        session.client.chat.side_effect = _chat_side_effect

        async def _call_tool_side_effect(name, *, arguments=None, **kwargs):
            if name == "search_knowledge_base":
                return _make_tool_result("The answer is in document 3.")
            return _make_empty_tool_result()

        session.mcp_session.call_tool = AsyncMock(side_effect=_call_tool_side_effect)

        stream = await session.chat("find answer")
        tokens = [t async for t in stream]

        assert "".join(tokens) == plain_answer
        assert call_count == 2

        # Second LLM call must include the tool result message
        second_call_messages = session.client.chat.call_args_list[1][1]["messages"]
        tool_result_msg = next(
            (m for m in second_call_messages if m["role"] == Role.TOOL.value), None
        )
        assert tool_result_msg is not None
        assert "The answer is in document 3." in tool_result_msg["content"]

    async def test_multiple_tool_calls_executed_concurrently(self):
        """Multiple tool calls in one response are all executed before the next LLM call."""
        tool_a = _make_mcp_tool("tool_a")
        tool_b = _make_mcp_tool("tool_b")
        session = _make_session_with_tools(tools=[tool_a, tool_b])

        tool_call_json = json.dumps(
            {
                "toolCalls": [
                    {"id": "c1", "name": "tool_a", "arguments": {}},
                    {"id": "c2", "name": "tool_b", "arguments": {}},
                ]
            }
        )

        call_count = 0

        def _chat_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"message": {"content": tool_call_json}}
            return {"message": {"content": "done"}}

        session.client.chat.side_effect = _chat_side_effect

        async def _call_tool_side_effect(name, *, arguments=None, **kwargs):
            if name in ("tool_a", "tool_b"):
                return _make_tool_result(f"{name} result")
            return _make_empty_tool_result()

        session.mcp_session.call_tool = AsyncMock(side_effect=_call_tool_side_effect)

        stream = await session.chat("run both")
        async for _ in stream:
            pass

        second_call_messages = session.client.chat.call_args_list[1][1]["messages"]
        tool_messages = [
            m for m in second_call_messages if m["role"] == Role.TOOL.value
        ]
        assert len(tool_messages) == 2

    async def test_loop_terminates_after_max_iterations(self):
        """The tool-calling loop stops after TOOL_CALL_MAX_LOOPS iterations."""
        tool = _make_mcp_tool("looping_tool")
        session = _make_session_with_tools(tools=[tool])

        # Always return a tool call to force maximum iterations
        tool_call_json = json.dumps(
            {"toolCalls": [{"id": "c1", "name": "looping_tool", "arguments": {}}]}
        )

        call_count = 0

        def _chat_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("stream"):
                # Final streaming call after loop exhaustion — returns plain text
                return iter([{"message": {"content": "final answer"}}])
            # All probe calls return tool call JSON
            return {"message": {"content": tool_call_json}}

        session.client.chat.side_effect = _chat_side_effect
        session.mcp_session.call_tool = _stub_call_tool({"looping_tool": "result"})

        stream = await session.chat("loop forever")
        async for _ in stream:
            pass

        # settings.server_tool_call_max_loops probe calls + 1 final streaming call
        assert session.client.chat.call_count == settings.server_tool_call_max_loops + 1

    async def test_unknown_tool_name_returns_error_to_llm(self):
        """Calling a non-existent tool name adds an error result message to the context."""
        tool = _make_mcp_tool("real_tool")
        session = _make_session_with_tools(tools=[tool])

        call_count = 0

        def _chat_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "message": {
                        "content": json.dumps(
                            {
                                "toolCalls": [
                                    {"id": "c1", "name": "fake_tool", "arguments": {}}
                                ]
                            }
                        )
                    }
                }
            return {"message": {"content": "ok"}}

        session.client.chat.side_effect = _chat_side_effect
        session.mcp_session.call_tool = _stub_call_tool({})

        stream = await session.chat("call fake")
        async for _ in stream:
            pass

        second_call_messages = session.client.chat.call_args_list[1][1]["messages"]
        error_msg = next(
            (m for m in second_call_messages if m["role"] == Role.TOOL.value), None
        )
        assert error_msg is not None
        assert "ERROR" in error_msg["content"]
        assert "fake_tool" in error_msg["content"]

    async def test_tool_execution_failure_returns_error_to_llm(self):
        """An exception during tool execution is returned as an error message, not raised."""
        tool = _make_mcp_tool("fragile_tool")
        session = _make_session_with_tools(tools=[tool])

        call_count = 0

        def _chat_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "message": {
                        "content": json.dumps(
                            {
                                "toolCalls": [
                                    {
                                        "id": "c1",
                                        "name": "fragile_tool",
                                        "arguments": {},
                                    }
                                ]
                            }
                        )
                    }
                }
            return {"message": {"content": "graceful answer"}}

        session.client.chat.side_effect = _chat_side_effect

        async def _failing_call_tool(name, *, arguments=None, **kwargs):
            if name == "fragile_tool":
                raise RuntimeError("service unavailable")
            return _make_empty_tool_result()

        session.mcp_session.call_tool = AsyncMock(side_effect=_failing_call_tool)

        stream = await session.chat("use fragile tool")
        tokens = [t async for t in stream]

        assert "".join(tokens) == "graceful answer"

        second_call_messages = session.client.chat.call_args_list[1][1]["messages"]
        error_msg = next(
            (m for m in second_call_messages if m["role"] == Role.TOOL.value), None
        )
        assert error_msg is not None
        assert "ERROR" in error_msg["content"]

    async def test_orientation_message_contains_tool_guidance(self):
        """When mcp_tools are present the orientation system message includes tool instructions."""
        tool = _make_mcp_tool("my_tool", "Does a thing")
        session = _make_session_with_tools(tools=[tool])
        session.mcp_session.call_tool = _stub_call_tool({})
        session.client.chat.return_value = {"message": {"content": "answer"}}

        stream = await session.chat("hello")
        async for _ in stream:
            pass

        first_call_messages = session.client.chat.call_args_list[0][1]["messages"]
        orientation = first_call_messages[0]
        assert orientation["role"] == "system"
        assert "my_tool" in orientation["content"]
        assert "toolCalls" in orientation["content"]

    async def test_tool_call_json_not_streamed_to_caller(self):
        """Tool-call JSON produced after loop exhaustion is never yielded to the caller."""
        tool = _make_mcp_tool("looping_tool")
        session = _make_session_with_tools(tools=[tool])

        tool_call_json = json.dumps(
            {"toolCalls": [{"id": "c1", "name": "looping_tool", "arguments": {}}]}
        )

        call_count = 0

        def _chat_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("stream"):
                # Still returns tool JSON even in the streaming call
                return iter([{"message": {"content": tool_call_json}}])
            if (
                not kwargs.get("stream")
                and call_count > settings.server_tool_call_max_loops
            ):
                # The forced plain-text call
                return {"message": {"content": "Here is my plain answer."}}
            return {"message": {"content": tool_call_json}}

        session.client.chat.side_effect = _chat_side_effect
        session.mcp_session.call_tool = _stub_call_tool({"looping_tool": "result"})

        stream = await session.chat("loop forever")
        tokens = [t async for t in stream]
        output = "".join(tokens)

        # The raw tool-call JSON must never reach the caller
        assert "toolCalls" not in output
        assert "tool_calls" not in output

    async def test_empty_tool_result_replaced_with_fallback_message(self):
        """When a tool returns no content the LLM receives an explicit fallback instead of silence."""
        tool = _make_mcp_tool(
            "search_knowledge_base", "Search", {"query": {"type": "string"}}
        )
        session = _make_session_with_tools(tools=[tool])

        tool_call_json = json.dumps(
            {
                "toolCalls": [
                    {
                        "id": "c1",
                        "name": "search_knowledge_base",
                        "arguments": {"query": "pm of britain"},
                    }
                ]
            }
        )

        call_count = 0

        def _chat_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"message": {"content": tool_call_json}}
            return {
                "message": {"content": "I don't know based on available information."}
            }

        session.client.chat.side_effect = _chat_side_effect
        # Tool returns empty content
        session.mcp_session.call_tool = AsyncMock(
            return_value=_make_empty_tool_result()
        )

        stream = await session.chat("who is the pm of britain?")
        tokens = [t async for t in stream]

        assert call_count == 2
        second_call_messages = session.client.chat.call_args_list[1][1]["messages"]
        tool_result_msg = next(
            (m for m in second_call_messages if m["role"] == Role.TOOL.value), None
        )
        assert tool_result_msg is not None
        # The fallback text must be present — not an empty result
        assert "No results found" in tool_result_msg["content"]


class TestInternalToolFiltering:
    def _make_tool_with_meta(self, name: str, tags: list[str]):
        tool = MagicMock()
        tool.name = name
        tool.meta = {"fastmcp": {"tags": tags}}
        return tool

    def _make_tool_no_meta(self, name: str):
        tool = MagicMock()
        tool.name = name
        tool.meta = None
        return tool

    def test_tool_tagged_internal_is_filtered(self):
        tool = self._make_tool_with_meta("load_long_term_memory", ["internal"])
        assert _is_internal_tool(tool) is True

    def test_tool_tagged_internal_among_others_is_filtered(self):
        tool = self._make_tool_with_meta("persist_message", ["internal", "other"])
        assert _is_internal_tool(tool) is True

    def test_tool_with_no_tags_is_not_filtered(self):
        tool = self._make_tool_with_meta("search_knowledge_base", [])
        assert _is_internal_tool(tool) is False

    def test_tool_with_no_meta_is_not_filtered(self):
        tool = self._make_tool_no_meta("search_knowledge_base")
        assert _is_internal_tool(tool) is False

    def test_tool_with_unrelated_tag_is_not_filtered(self):
        tool = self._make_tool_with_meta("some_public_tool", ["readonly"])
        assert _is_internal_tool(tool) is False


@pytest.mark.asyncio
class TestNoUnconditionalRagCall:
    async def test_search_knowledge_base_not_called_at_startup(self):
        """The server must not call search_knowledge_base unconditionally on every turn."""
        tool = _make_mcp_tool("search_knowledge_base", "Search")
        session = _make_session_with_tools(tools=[tool])

        called_tool_names: list[str] = []

        async def _track_call_tool(name, *, arguments=None, **kwargs):
            called_tool_names.append(name)
            return _make_tool_result("")

        session.mcp_session.call_tool = AsyncMock(side_effect=_track_call_tool)
        session.client.chat.return_value = {"message": {"content": "plain answer"}}

        stream = await session.chat("hello")
        async for _ in stream:
            pass

        assert "search_knowledge_base" not in called_tool_names

    async def test_infrastructure_tools_still_called_unconditionally(self):
        """load_long_term_memory, load_conversation_window, and persist_message
        must still be called by the server on every turn regardless of model tools."""
        session = _make_session_with_tools(tools=[])

        called_tool_names: list[str] = []

        async def _track_call_tool(name, *, arguments=None, **kwargs):
            called_tool_names.append(name)
            return _make_tool_result("")

        session.mcp_session.call_tool = AsyncMock(side_effect=_track_call_tool)
        session.client.chat.return_value = iter([{"message": {"content": "answer"}}])

        stream = await session.chat("hello")
        async for _ in stream:
            pass

        assert "load_long_term_memory" in called_tool_names
        assert "load_conversation_window" in called_tool_names
        assert "persist_message" in called_tool_names


@pytest.mark.asyncio
class TestJsonSafetyGuard:
    async def test_json_shaped_non_tool_response_not_streamed(self):
        """A JSON response that parse_tool_calls cannot parse must not become the
        resolved_answer and must not leak raw JSON to the caller."""
        tool = _make_mcp_tool("search_knowledge_base", "Search")
        session = _make_session_with_tools(tools=[tool])
        session.mcp_session.call_tool = _stub_call_tool({})

        # JSON-shaped but not a valid tool-call payload — parse_tool_calls returns []
        json_shaped_but_not_tool_call = json.dumps({"message": "some json response"})

        call_count = 0

        def _chat_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"message": {"content": json_shaped_but_not_tool_call}}
            # Fallthrough to streaming after safety guard breaks the loop
            return iter([{"message": {"content": "plain text answer"}}])

        session.client.chat.side_effect = _chat_side_effect

        stream = await session.chat("hello")
        tokens = [t async for t in stream]
        output = "".join(tokens)

        # The raw JSON must not leak to the caller
        assert json_shaped_but_not_tool_call not in output
