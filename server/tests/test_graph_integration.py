import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import TextContent, Tool

from server.chat import Role, run_chat_graph
from server.services import _is_internal_tool


def _make_mcp_tool(
    name: str, description: str = "", params: dict | None = None
) -> Tool:
    """Build a minimal MCP Tool with the given name and input schema."""
    return Tool(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": params or {},
            "required": list((params or {}).keys()),
        },
    )


def _make_tool_result(text: str):
    """Return a mock MCP call_tool result with a real TextContent item."""
    content_item = TextContent(type="text", text=text)
    result = MagicMock()
    result.content = [content_item]
    return result


def _make_empty_tool_result():
    """Return a mock MCP call_tool result with no content."""
    result = MagicMock()
    result.content = []
    return result


def _make_stream_response(tokens: list[str]):
    """Return an async generator that yields mock stream chunks."""

    async def _gen():
        for token in tokens:
            chunk = MagicMock()
            chunk.message.content = token
            yield chunk

    return _gen()


def _stub_call_tool(tool_responses: dict):
    """Return an AsyncMock for call_tool that dispatches by tool name."""

    async def _call_tool(name, *, arguments=None, **kwargs):
        text = tool_responses.get(name, "")
        if text == "":
            return _make_empty_tool_result()
        return _make_tool_result(text)

    return AsyncMock(side_effect=_call_tool)


def _make_fc_response(tool_calls=None):
    response = MagicMock()
    response.message.tool_calls = tool_calls or []
    return response


def _make_fc_tool_call(name: str, arguments: dict | None = None):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments or {}
    return tc


async def _run_graph(
    user_input: str = "hello",
    mcp_tools: list | None = None,
    mcp_session=None,
    ollama_client=None,
    session_id=None,
):
    """Run run_chat_graph with sensible defaults and collect all tokens."""
    if mcp_session is None:
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool({})
    if ollama_client is None:
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

    # run_chat_graph is now an async generator, so don't await it
    stream = run_chat_graph(
        session_id=session_id or uuid.uuid4(),
        user_input=user_input,
        mcp_session=mcp_session,
        ollama_client=ollama_client,
        chat_model="test-chat",
        function_calling_model="test-fc",
        mcp_tools=mcp_tools or [],
    )
    return [token async for token in stream]


@pytest.mark.asyncio
class TestTokenStreaming:
    async def test_tokens_yielded_in_order(self):
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool({})

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(
            return_value=_make_stream_response(["Hello", " world"])
        )

        tokens = await _run_graph(
            mcp_session=mcp_session,
            ollama_client=ollama_client,
        )

        assert "".join(tokens) == "Hello world"

    async def test_infrastructure_tools_always_called(self):
        """load_long_term_memory, load_conversation_window, and persist_message
        are always invoked regardless of whether mcp_tools is empty."""
        mcp_session = AsyncMock()
        call_tool_mock = _stub_call_tool({})
        mcp_session.call_tool = call_tool_mock

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        await _run_graph(mcp_session=mcp_session, ollama_client=ollama_client)

        called_names = {c.args[0] for c in call_tool_mock.await_args_list}
        assert "load_long_term_memory" in called_names
        assert "load_conversation_window" in called_names
        assert "persist_message" in called_names

    async def test_search_kb_not_called_when_no_tools(self):
        """search_knowledge_base must never be called unless the FC model requests it."""
        mcp_session = AsyncMock()
        call_tool_mock = _stub_call_tool({})
        mcp_session.call_tool = call_tool_mock

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        await _run_graph(mcp_session=mcp_session, ollama_client=ollama_client)

        called_names = {c.args[0] for c in call_tool_mock.await_args_list}
        assert "search_knowledge_base" not in called_names

    async def test_tokens_streamed_incrementally_not_buffered(self):
        """Tokens should be yielded as they arrive, not buffered until completion.

        This test verifies the fix for the streaming regression: with astream()
        the graph yields partial results during streaming (no TTFB delay).
        """
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool({})

        ollama_client = AsyncMock()
        # Simulate Ollama streaming 3 tokens incrementally
        ollama_client.chat = AsyncMock(
            return_value=_make_stream_response(["Hello", " ", "world"])
        )

        tokens = await _run_graph(mcp_session=mcp_session, ollama_client=ollama_client)

        # Verify tokens are yielded in order and can be consumed incrementally
        assert len(tokens) > 0, "should have streamed at least one token"
        assert "".join(tokens) == "Hello world"

    async def test_long_response_does_not_buffer_excessively(self):
        """Large responses should not cause excessive memory usage during streaming.

        With astream() each token update is yielded immediately rather than
        accumulating all tokens before returning.
        """
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool({})

        # Simulate a large response with many small tokens
        large_response_tokens = ["word"] * 1000
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(
            return_value=_make_stream_response(large_response_tokens)
        )

        tokens = await _run_graph(mcp_session=mcp_session, ollama_client=ollama_client)

        # Verify all tokens were streamed
        assert len(tokens) > 0
        assert "".join(tokens) == "word" * 1000

    async def test_assistant_message_persisted_with_full_response(self):
        """After streaming completes, the full assembled response is persisted."""
        mcp_session = AsyncMock()
        call_tool_mock = _stub_call_tool({})
        mcp_session.call_tool = call_tool_mock

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(
            return_value=_make_stream_response(["Complete", " response"])
        )

        session_id = uuid.uuid4()
        await _run_graph(
            mcp_session=mcp_session,
            ollama_client=ollama_client,
            session_id=session_id,
        )

        # Find the persist_message call for the assistant turn (last one should be)
        persist_calls = [
            c for c in call_tool_mock.await_args_list if c[0][0] == "persist_message"
        ]

        assert len(persist_calls) >= 2, "should have persisted user and assistant turns"

        # The last persist_message call should be the assistant response
        last_persist = persist_calls[-1]
        assert last_persist[1]["arguments"]["role"] == "assistant"
        # Should contain the full streamed response
        assert last_persist[1]["arguments"]["content"] == "Complete response"


@pytest.mark.asyncio
class TestContextMessageOrder:
    async def _capture_generate_messages(self, mcp_session, ollama_client):
        """Return the messages passed to the final generate_response Ollama call."""
        await _run_graph(mcp_session=mcp_session, ollama_client=ollama_client)
        return ollama_client.chat.call_args[1]["messages"]

    async def test_orientation_is_always_first(self):
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool({})
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        messages = await self._capture_generate_messages(mcp_session, ollama_client)

        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]

    async def test_no_extra_messages_when_context_empty(self):
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool({})
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        messages = await self._capture_generate_messages(mcp_session, ollama_client)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "hello"}

    async def test_long_term_memory_after_orientation(self):
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool(
            {"load_long_term_memory": "User likes Python"}
        )
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        messages = await self._capture_generate_messages(mcp_session, ollama_client)

        assert messages[1] == {"role": "system", "content": "User likes Python"}

    async def test_window_turns_after_long_term_memory(self):
        window_turns = [
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a"},
        ]
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool(
            {
                "load_long_term_memory": "some memory",
                "load_conversation_window": json.dumps(window_turns),
            }
        )
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        messages = await self._capture_generate_messages(mcp_session, ollama_client)

        assert messages[2] == {"role": "user", "content": "prev q"}
        assert messages[3] == {"role": "assistant", "content": "prev a"}

    async def test_user_turn_is_last(self):
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool({})
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        await _run_graph(
            user_input="my question",
            mcp_session=mcp_session,
            ollama_client=ollama_client,
        )

        messages = ollama_client.chat.call_args[1]["messages"]
        assert messages[-1] == {"role": "user", "content": "my question"}

    async def test_full_context_order(self):
        window_turns = [
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a"},
        ]
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool(
            {
                "load_long_term_memory": "memory facts",
                "load_conversation_window": json.dumps(window_turns),
            }
        )
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        await _run_graph(
            user_input="current q",
            mcp_session=mcp_session,
            ollama_client=ollama_client,
        )

        messages = ollama_client.chat.call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]
        assert messages[1] == {"role": "system", "content": "memory facts"}
        assert messages[2] == {"role": "user", "content": "prev q"}
        assert messages[3] == {"role": "assistant", "content": "prev a"}
        assert messages[4] == {"role": "user", "content": "current q"}
        assert len(messages) == 5


@pytest.mark.asyncio
class TestPersistMessages:
    async def test_user_message_persisted_before_stream(self):
        mcp_session = AsyncMock()
        call_tool_mock = _stub_call_tool({})
        mcp_session.call_tool = call_tool_mock

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        await _run_graph(
            user_input="user input",
            mcp_session=mcp_session,
            ollama_client=ollama_client,
        )

        persist_calls = [
            c for c in call_tool_mock.await_args_list if c.args[0] == "persist_message"
        ]
        user_persist = next(
            (c for c in persist_calls if c.kwargs["arguments"]["role"] == "user"),
            None,
        )
        assert user_persist is not None
        assert user_persist.kwargs["arguments"]["content"] == "user input"

    async def test_assistant_message_persisted_after_stream(self):
        mcp_session = AsyncMock()
        call_tool_mock = _stub_call_tool({})
        mcp_session.call_tool = call_tool_mock

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(
            return_value=_make_stream_response(["Hello", " there"])
        )

        await _run_graph(mcp_session=mcp_session, ollama_client=ollama_client)

        persist_calls = [
            c for c in call_tool_mock.await_args_list if c.args[0] == "persist_message"
        ]
        assistant_persist = next(
            (c for c in persist_calls if c.kwargs["arguments"]["role"] == "assistant"),
            None,
        )
        assert assistant_persist is not None
        assert assistant_persist.kwargs["arguments"]["content"] == "Hello there"

    async def test_persist_failure_does_not_raise(self):
        """An error persisting the assistant message must not propagate to the caller."""
        mcp_session = AsyncMock()

        async def _failing_persist(name, *, arguments=None, **kwargs):
            if (
                name == "persist_message"
                and arguments
                and arguments.get("role") == "assistant"
            ):
                raise RuntimeError("db down")
            return _make_empty_tool_result()

        mcp_session.call_tool = AsyncMock(side_effect=_failing_persist)

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        tokens = await _run_graph(mcp_session=mcp_session, ollama_client=ollama_client)
        assert "".join(tokens) == "ok"


@pytest.mark.asyncio
class TestToolCallingLoop:
    async def test_no_fc_call_when_mcp_tools_empty(self):
        """With no mcp_tools the FC model is never called; only the chat model streams."""
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool({})

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["answer"]))

        await _run_graph(
            mcp_tools=[],
            mcp_session=mcp_session,
            ollama_client=ollama_client,
        )

        call_kwargs_list = [c[1] for c in ollama_client.chat.call_args_list]
        fc_calls = [k for k in call_kwargs_list if k.get("stream") is False]
        assert len(fc_calls) == 0

    async def test_plain_text_response_streamed_directly(self):
        """When the FC model responds with no tool calls, the text is streamed as-is."""
        tool = _make_mcp_tool("search_kb", "Search")
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool({})

        call_count = 0

        async def _chat_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if not kwargs.get("stream"):
                return _make_fc_response(tool_calls=[])
            return _make_stream_response(["plain answer"])

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(side_effect=_chat_side_effect)

        tokens = await _run_graph(
            mcp_tools=[tool],
            mcp_session=mcp_session,
            ollama_client=ollama_client,
        )

        assert "".join(tokens) == "plain answer"

    async def test_tool_result_fed_back_to_llm(self):
        """A tool call triggers execution and the result is included in the final LLM call."""
        tool = _make_mcp_tool("search_kb", "Search", {"query": {"type": "string"}})
        mcp_session = AsyncMock()

        async def _call_tool_side_effect(name, *, arguments=None, **kwargs):
            if name == "search_kb":
                return _make_tool_result("The answer is in document 3.")
            return _make_empty_tool_result()

        mcp_session.call_tool = AsyncMock(side_effect=_call_tool_side_effect)

        fc_call_count = 0

        async def _chat_side_effect(**kwargs):
            nonlocal fc_call_count
            if not kwargs.get("stream"):
                fc_call_count += 1
                if fc_call_count == 1:
                    return _make_fc_response(
                        tool_calls=[_make_fc_tool_call("search_kb", {"query": "foo"})]
                    )
                return _make_fc_response(tool_calls=[])
            return _make_stream_response(["Based on docs, answer is 42."])

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(side_effect=_chat_side_effect)

        tokens = await _run_graph(
            mcp_tools=[tool],
            mcp_session=mcp_session,
            ollama_client=ollama_client,
        )

        assert "".join(tokens) == "Based on docs, answer is 42."
        generate_call_kwargs = ollama_client.chat.call_args_list[-1][1]
        messages = generate_call_kwargs["messages"]
        tool_msgs = [m for m in messages if m.get("role") == Role.TOOL.value]
        assert len(tool_msgs) >= 1
        assert any("The answer is in document 3." in m["content"] for m in tool_msgs)

    async def test_loop_stops_at_max_loops(self):
        """The tool-calling loop terminates after server_tool_call_max_loops iterations."""
        tool = _make_mcp_tool("looping_tool")
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool({"looping_tool": "some result"})

        fc_call_count = 0

        async def _chat_side_effect(**kwargs):
            nonlocal fc_call_count
            if not kwargs.get("stream"):
                fc_call_count += 1
                return _make_fc_response(
                    tool_calls=[_make_fc_tool_call("looping_tool")]
                )
            return _make_stream_response(["final answer"])

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(side_effect=_chat_side_effect)

        from server.config import settings

        await _run_graph(
            mcp_tools=[tool],
            mcp_session=mcp_session,
            ollama_client=ollama_client,
        )

        assert fc_call_count == settings.server_tool_call_max_loops

    async def test_unknown_tool_error_returned_to_llm(self):
        """Calling a non-existent tool produces an error result visible to the LLM."""
        tool = _make_mcp_tool("real_tool")
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_call_tool({})

        fc_call_count = 0

        async def _chat_side_effect(**kwargs):
            nonlocal fc_call_count
            if not kwargs.get("stream"):
                fc_call_count += 1
                if fc_call_count == 1:
                    return _make_fc_response(
                        tool_calls=[_make_fc_tool_call("fake_tool")]
                    )
                return _make_fc_response(tool_calls=[])
            return _make_stream_response(["ok"])

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(side_effect=_chat_side_effect)

        await _run_graph(
            mcp_tools=[tool],
            mcp_session=mcp_session,
            ollama_client=ollama_client,
        )

        generate_call_messages = ollama_client.chat.call_args_list[-1][1]["messages"]
        error_msgs = [
            m
            for m in generate_call_messages
            if m.get("role") == Role.TOOL.value and "ERROR" in m["content"]
        ]
        assert len(error_msgs) >= 1
        assert any("fake_tool" in m["content"] for m in error_msgs)

    async def test_tool_execution_failure_returned_to_llm(self):
        """An exception during tool execution is surfaced to the LLM, not raised."""
        tool = _make_mcp_tool("fragile_tool")
        mcp_session = AsyncMock()

        fc_call_count = 0

        async def _chat_side_effect(**kwargs):
            nonlocal fc_call_count
            if not kwargs.get("stream"):
                fc_call_count += 1
                if fc_call_count == 1:
                    return _make_fc_response(
                        tool_calls=[_make_fc_tool_call("fragile_tool")]
                    )
                return _make_fc_response(tool_calls=[])
            return _make_stream_response(["graceful answer"])

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(side_effect=_chat_side_effect)

        async def _failing_call_tool(name, *, arguments=None, **kwargs):
            if name == "fragile_tool":
                raise RuntimeError("service unavailable")
            return _make_empty_tool_result()

        mcp_session.call_tool = AsyncMock(side_effect=_failing_call_tool)

        tokens = await _run_graph(
            mcp_tools=[tool],
            mcp_session=mcp_session,
            ollama_client=ollama_client,
        )

        assert "".join(tokens) == "graceful answer"
        generate_call_messages = ollama_client.chat.call_args_list[-1][1]["messages"]
        error_msgs = [
            m
            for m in generate_call_messages
            if m.get("role") == Role.TOOL.value and "ERROR" in m["content"]
        ]
        assert len(error_msgs) >= 1

    async def test_empty_tool_result_replaced_with_fallback(self):
        """When a tool returns no content the LLM receives an explicit fallback message."""
        tool = _make_mcp_tool("search_kb", "Search", {"query": {"type": "string"}})
        mcp_session = AsyncMock()

        fc_call_count = 0

        async def _chat_side_effect(**kwargs):
            nonlocal fc_call_count
            if not kwargs.get("stream"):
                fc_call_count += 1
                if fc_call_count == 1:
                    return _make_fc_response(
                        tool_calls=[
                            _make_fc_tool_call("search_kb", {"query": "pm of britain"})
                        ]
                    )
                return _make_fc_response(tool_calls=[])
            return _make_stream_response(["I don't know."])

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(side_effect=_chat_side_effect)
        mcp_session.call_tool = AsyncMock(return_value=_make_empty_tool_result())

        await _run_graph(
            mcp_tools=[tool],
            mcp_session=mcp_session,
            ollama_client=ollama_client,
        )

        generate_call_messages = ollama_client.chat.call_args_list[-1][1]["messages"]
        tool_msgs = [
            m for m in generate_call_messages if m.get("role") == Role.TOOL.value
        ]
        assert any("No results found" in m["content"] for m in tool_msgs)


class TestInternalToolFiltering:
    def _make_tool_with_meta(self, name: str, tags: list[str]) -> MagicMock:
        tool = MagicMock()
        tool.name = name
        tool.meta = {"fastmcp": {"tags": tags}}
        return tool

    def _make_tool_no_meta(self, name: str) -> MagicMock:
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
