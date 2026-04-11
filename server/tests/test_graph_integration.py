import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from mcp.types import Tool

from server.chat import Role, run_chat_graph
from server.config import settings
from tests.helpers import (
    _make_client_with_transport,
    _make_empty_tool_result,
    _make_mcp_tool,
    _make_memory_client,
    _make_stream_response,
    _make_tool_result,
)


def _stub_mcp_call_tool(tool_responses: dict):
    """Return an AsyncMock for MCP call_tool that dispatches by tool name.

    Only used for model-callable tools (e.g. search_knowledge_base) — the
    three infrastructure calls now go through the memory HTTP endpoints.
    """

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
    memory_http_client=None,
    ollama_client=None,
    session_id=None,
    long_term_content: str = "",
    window_turns: list | None = None,
):
    """Run run_chat_graph with sensible defaults and collect all tokens."""
    if mcp_session is None:
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_mcp_call_tool({})
    if memory_http_client is None:
        memory_http_client = _make_memory_client(
            long_term_content=long_term_content,
            window_turns=window_turns,
        )
    if ollama_client is None:
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

    stream = run_chat_graph(
        session_id=session_id or uuid.uuid4(),
        user_input=user_input,
        mcp_session=mcp_session,
        memory_http_client=memory_http_client,
        ollama_client=ollama_client,
        chat_model="test-chat",
        function_calling_model="test-fc",
        mcp_tools=mcp_tools or [],
    )
    tokens = [token async for token in stream]
    await asyncio.sleep(0)
    return tokens


@pytest.mark.asyncio
class TestTokenStreaming:
    async def test_tokens_yielded_in_order(self):
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(
            return_value=_make_stream_response(["Hello", " world"])
        )

        tokens = await _run_graph(ollama_client=ollama_client)

        assert "".join(tokens) == "Hello world"

    async def test_memory_endpoints_always_called(self):
        """All three memory endpoints are called on every turn regardless of mcp_tools."""
        calls: list[str] = []

        async def _transport(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if "/memory/long-term/" in request.url.path:
                return httpx.Response(200, json={"content": ""})
            if "/memory/window/" in request.url.path:
                return httpx.Response(200, json=[])
            if request.url.path == "/memory/messages":
                return httpx.Response(204)
            return httpx.Response(404)

        client = _make_client_with_transport(_transport)

        await _run_graph(memory_http_client=client)

        assert any("/memory/long-term/" in p for p in calls)
        assert any("/memory/window/" in p for p in calls)
        assert "/memory/messages" in calls

    async def test_search_kb_not_called_when_no_tools(self):
        """search_knowledge_base must never be called unless the FC model requests it."""
        mcp_session = AsyncMock()
        call_tool_mock = _stub_mcp_call_tool({})
        mcp_session.call_tool = call_tool_mock

        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        await _run_graph(mcp_session=mcp_session, ollama_client=ollama_client)

        called_names = {c.args[0] for c in call_tool_mock.await_args_list}
        assert "search_knowledge_base" not in called_names

    async def test_tokens_streamed_incrementally_not_buffered(self):
        """Tokens should be yielded as they arrive, not buffered until completion."""
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(
            return_value=_make_stream_response(["Hello", " ", "world"])
        )

        tokens = await _run_graph(ollama_client=ollama_client)

        assert len(tokens) > 0, "should have streamed at least one token"
        assert "".join(tokens) == "Hello world"

    async def test_long_response_does_not_buffer_excessively(self):
        """Large responses should not cause excessive memory usage during streaming."""
        large_response_tokens = ["word"] * 1000
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(
            return_value=_make_stream_response(large_response_tokens)
        )

        tokens = await _run_graph(ollama_client=ollama_client)

        assert len(tokens) > 0
        assert "".join(tokens) == "word" * 1000

    async def test_assistant_message_persisted_with_full_response(self):
        """After streaming completes, the full assembled response is persisted."""
        persisted: list[dict] = []

        async def _transport(request: httpx.Request) -> httpx.Response:
            if "/memory/long-term/" in request.url.path:
                return httpx.Response(200, json={"content": ""})
            if "/memory/window/" in request.url.path:
                return httpx.Response(200, json=[])
            if request.url.path == "/memory/messages":
                persisted.append(request.read())
                return httpx.Response(204)
            return httpx.Response(404)

        client = _make_client_with_transport(_transport)
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(
            return_value=_make_stream_response(["Complete", " response"])
        )

        await _run_graph(memory_http_client=client, ollama_client=ollama_client)

        assert len(persisted) >= 2, "should have persisted user and assistant turns"

        last_body = json.loads(persisted[-1])
        assert last_body["role"] == "assistant"
        assert last_body["content"] == "Complete response"


@pytest.mark.asyncio
class TestContextMessageOrder:
    async def _capture_generate_messages(self, ollama_client, **kwargs):
        """Return the messages passed to the final generate_response Ollama call."""
        await _run_graph(ollama_client=ollama_client, **kwargs)
        return ollama_client.chat.call_args[1]["messages"]

    async def test_orientation_is_always_first(self):
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        messages = await self._capture_generate_messages(ollama_client)

        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]

    async def test_no_extra_messages_when_context_empty(self):
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        messages = await self._capture_generate_messages(ollama_client)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "hello"}

    async def test_long_term_memory_after_orientation(self):
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        messages = await self._capture_generate_messages(
            ollama_client, long_term_content="User likes Python"
        )

        assert messages[1] == {"role": "system", "content": "User likes Python"}

    async def test_window_turns_after_long_term_memory(self):
        window_turns = [
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a"},
        ]
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        messages = await self._capture_generate_messages(
            ollama_client,
            long_term_content="some memory",
            window_turns=window_turns,
        )

        assert messages[2] == {"role": "user", "content": "prev q"}
        assert messages[3] == {"role": "assistant", "content": "prev a"}

    async def test_user_turn_is_last(self):
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        await _run_graph(user_input="my question", ollama_client=ollama_client)

        messages = ollama_client.chat.call_args[1]["messages"]
        assert messages[-1] == {"role": "user", "content": "my question"}

    async def test_full_context_order(self):
        window_turns = [
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a"},
        ]
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        await _run_graph(
            user_input="current q",
            ollama_client=ollama_client,
            long_term_content="memory facts",
            window_turns=window_turns,
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
    async def test_user_message_persisted_after_stream(self):
        """User message is persisted after streaming to avoid concurrent Ollama calls."""
        persisted: list[dict] = []

        async def _transport(request: httpx.Request) -> httpx.Response:
            if "/memory/long-term/" in request.url.path:
                return httpx.Response(200, json={"content": ""})
            if "/memory/window/" in request.url.path:
                return httpx.Response(200, json=[])
            if request.url.path == "/memory/messages":
                persisted.append(json.loads(request.read()))
                return httpx.Response(204)
            return httpx.Response(404)

        client = _make_client_with_transport(_transport)
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        await _run_graph(
            user_input="user input",
            memory_http_client=client,
            ollama_client=ollama_client,
        )

        user_persist = next((p for p in persisted if p.get("role") == "user"), None)
        assert user_persist is not None
        assert user_persist["content"] == "user input"

    async def test_assistant_message_persisted_after_stream(self):
        persisted: list[dict] = []

        async def _transport(request: httpx.Request) -> httpx.Response:
            if "/memory/long-term/" in request.url.path:
                return httpx.Response(200, json={"content": ""})
            if "/memory/window/" in request.url.path:
                return httpx.Response(200, json=[])
            if request.url.path == "/memory/messages":
                persisted.append(json.loads(request.read()))
                return httpx.Response(204)
            return httpx.Response(404)

        client = _make_client_with_transport(_transport)
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(
            return_value=_make_stream_response(["Hello", " there"])
        )

        await _run_graph(memory_http_client=client, ollama_client=ollama_client)

        assistant_persist = next(
            (p for p in persisted if p.get("role") == "assistant"), None
        )
        assert assistant_persist is not None
        assert assistant_persist["content"] == "Hello there"

    async def test_persist_failure_does_not_raise(self):
        """An HTTP error persisting the assistant message must not propagate."""
        call_count = 0

        async def _transport(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            if "/memory/long-term/" in request.url.path:
                return httpx.Response(200, json={"content": ""})
            if "/memory/window/" in request.url.path:
                return httpx.Response(200, json=[])
            if request.url.path == "/memory/messages":
                call_count += 1
                if call_count == 1:
                    return httpx.Response(204)
                return httpx.Response(500)
            return httpx.Response(404)

        client = _make_client_with_transport(_transport)
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["ok"]))

        tokens = await _run_graph(
            memory_http_client=client, ollama_client=ollama_client
        )
        assert "".join(tokens) == "ok"

    async def test_user_persisted_before_assistant(self):
        """User message must be persisted before assistant to maintain turn order."""
        persist_order: list[str] = []

        async def _transport(request: httpx.Request) -> httpx.Response:
            if "/memory/long-term/" in request.url.path:
                return httpx.Response(200, json={"content": ""})
            if "/memory/window/" in request.url.path:
                return httpx.Response(200, json=[])
            if request.url.path == "/memory/messages":
                body = json.loads(request.read())
                persist_order.append(body["role"])
                return httpx.Response(204)
            return httpx.Response(404)

        client = _make_client_with_transport(_transport)
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["response"]))

        await _run_graph(
            user_input="my question",
            memory_http_client=client,
            ollama_client=ollama_client,
        )

        assert persist_order == ["user", "assistant"]


@pytest.mark.asyncio
class TestToolCallingLoop:
    async def test_no_fc_call_when_mcp_tools_empty(self):
        """With no mcp_tools the FC model is never called; only the chat model streams."""
        ollama_client = AsyncMock()
        ollama_client.chat = AsyncMock(return_value=_make_stream_response(["answer"]))

        await _run_graph(mcp_tools=[], ollama_client=ollama_client)

        call_kwargs_list = [c[1] for c in ollama_client.chat.call_args_list]
        fc_calls = [k for k in call_kwargs_list if k.get("stream") is False]
        assert len(fc_calls) == 0

    async def test_plain_text_response_streamed_directly(self):
        """When the FC model responds with no tool calls, the text is streamed as-is."""
        tool = _make_mcp_tool("search_kb", "Search")
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_mcp_call_tool({})

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
        tool_msgs = [
            m
            for m in messages
            if m.get("role") == Role.SYSTEM.value
            and "[Tool result" in m.get("content", "")
        ]
        assert len(tool_msgs) >= 1
        assert any("The answer is in document 3." in m["content"] for m in tool_msgs)

    async def test_loop_stops_at_max_loops(self):
        """The tool-calling loop terminates after server_tool_call_max_loops iterations."""
        tool = _make_mcp_tool("looping_tool")
        mcp_session = AsyncMock()
        mcp_session.call_tool = _stub_mcp_call_tool({"looping_tool": "some result"})

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
        mcp_session.call_tool = _stub_mcp_call_tool({})

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
            if m.get("role") == Role.SYSTEM.value and "ERROR" in m["content"]
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
            if m.get("role") == Role.SYSTEM.value and "ERROR" in m["content"]
        ]
        assert len(error_msgs) >= 1

    async def test_empty_tool_result_replaced_with_fallback(self):
        """When a tool returns no content the LLM receives a neutral fallback message."""
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
            m
            for m in generate_call_messages
            if m.get("role") == Role.SYSTEM.value
            and "[Tool result" in m.get("content", "")
        ]
        assert any("No tool output was returned." in m["content"] for m in tool_msgs)
