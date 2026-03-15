import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.chat import ChatSession, Role


def _make_tool_result(text: str):
    """Return a mock MCP call_tool result with a single text content item."""
    content_item = MagicMock()
    content_item.text = text
    result = MagicMock()
    result.content = [content_item]
    return result


def _make_empty_tool_result():
    """Return a mock MCP call_tool result with no content."""
    result = MagicMock()
    result.content = []
    return result


def _make_session(**kwargs) -> ChatSession:
    """Return a ChatSession with mock MCP and Ollama dependencies."""
    mcp_session = AsyncMock()
    mcp_session.call_tool = AsyncMock(return_value=_make_tool_result(""))
    ollama_client = MagicMock()
    return ChatSession(
        session_id=kwargs.pop("session_id", uuid.uuid4()),
        mcp_session=kwargs.pop("mcp_session", mcp_session),
        ollama_client=kwargs.pop("ollama_client", ollama_client),
        thread_pool=kwargs.pop("thread_pool", ThreadPoolExecutor()),
        **kwargs,
    )


def _stub_call_tool(tool_responses: dict):
    """Return an AsyncMock for call_tool that dispatches by tool name.

    Args:
        tool_responses: Mapping of tool name -> text string to return.
                        Missing keys default to an empty string result.
    """

    async def _call_tool(name, *, arguments=None, **kwargs):
        text = tool_responses.get(name, "")
        if text == "":
            return _make_empty_tool_result()
        return _make_tool_result(text)

    return AsyncMock(side_effect=_call_tool)


@pytest.mark.asyncio
class TestChatTokenStream:
    async def test_stream_yields_tokens(self):
        """Tokens from ollama_client.chat() are yielded in order."""
        session = _make_session()
        session.mcp_session.call_tool = _stub_call_tool({})
        session.client.chat.return_value = iter(
            [
                {"message": {"content": "Hello"}},
                {"message": {"content": " world"}},
            ]
        )

        stream = await session.chat("hi")
        tokens = [t async for t in stream]

        assert "".join(tokens) == "Hello world"

    async def test_rag_not_called_when_search_returns_empty(self):
        """search_knowledge_base is not called when mcp_tools is empty."""
        session = _make_session()
        call_tool_mock = _stub_call_tool({})
        session.mcp_session.call_tool = call_tool_mock
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        stream = await session.chat("hello")
        async for _ in stream:
            pass

        search_calls = [
            c
            for c in call_tool_mock.await_args_list
            if c.args[0] == "search_knowledge_base"
        ]
        assert len(search_calls) == 0

    async def test_infrastructure_tools_called_unconditionally(self):
        """load_long_term_memory, load_conversation_window and persist_message
        are always called even when no mcp_tools are configured."""
        session = _make_session()
        call_tool_mock = _stub_call_tool({})
        session.mcp_session.call_tool = call_tool_mock
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        stream = await session.chat("hello")
        async for _ in stream:
            pass

        called_names = {c.args[0] for c in call_tool_mock.await_args_list}
        assert "load_long_term_memory" in called_names
        assert "load_conversation_window" in called_names
        assert "persist_message" in called_names


@pytest.mark.asyncio
class TestContextMessageOrder:
    async def test_orientation_is_always_first(self):
        """The orientation system message must always be the first message."""
        session = _make_session()
        session.mcp_session.call_tool = _stub_call_tool({})
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        stream = await session.chat("hello")
        async for _ in stream:
            pass

        messages = session.client.chat.call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]

    async def test_no_extra_messages_when_context_empty(self):
        """With no long-term memory, no window, and no RAG: orientation + user = 2 messages."""
        session = _make_session()
        session.mcp_session.call_tool = _stub_call_tool({})
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        stream = await session.chat("hello")
        async for _ in stream:
            pass

        messages = session.client.chat.call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "hello"}

    async def test_long_term_memory_injected_after_orientation(self):
        """Long-term memory system message comes immediately after the orientation."""
        session = _make_session()
        session.mcp_session.call_tool = _stub_call_tool(
            {"load_long_term_memory": "User is a Python developer"}
        )
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        stream = await session.chat("hello")
        async for _ in stream:
            pass

        messages = session.client.chat.call_args[1]["messages"]
        assert messages[1] == {
            "role": "system",
            "content": "User is a Python developer",
        }

    async def test_window_turns_inserted_after_long_term(self):
        """Window turns (parsed from JSON) are inserted after long-term memory."""
        window_turns = [
            {"role": "user", "content": "prev question"},
            {"role": "assistant", "content": "prev answer"},
        ]
        session = _make_session()
        session.mcp_session.call_tool = _stub_call_tool(
            {
                "load_long_term_memory": "some memory",
                "load_conversation_window": json.dumps(window_turns),
            }
        )
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        stream = await session.chat("current question")
        async for _ in stream:
            pass

        messages = session.client.chat.call_args[1]["messages"]
        assert messages[2] == {"role": "user", "content": "prev question"}
        assert messages[3] == {"role": "assistant", "content": "prev answer"}

    async def test_user_turn_is_always_last_without_tools(self):
        """The current user turn is the final message when no mcp_tools are configured."""
        session = _make_session()
        session.mcp_session.call_tool = _stub_call_tool({})
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        stream = await session.chat("my question")
        async for _ in stream:
            pass

        messages = session.client.chat.call_args[1]["messages"]
        assert messages[-1] == {"role": "user", "content": "my question"}

    async def test_full_context_order(self):
        """Full order: orientation → long_term → window → user."""
        window_turns = [
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a"},
        ]
        session = _make_session()
        session.mcp_session.call_tool = _stub_call_tool(
            {
                "load_long_term_memory": "memory facts",
                "load_conversation_window": json.dumps(window_turns),
            }
        )
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        stream = await session.chat("current q")
        async for _ in stream:
            pass

        messages = session.client.chat.call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]
        assert messages[1] == {"role": "system", "content": "memory facts"}
        assert messages[2] == {"role": "user", "content": "prev q"}
        assert messages[3] == {"role": "assistant", "content": "prev a"}
        assert messages[4] == {"role": "user", "content": "current q"}
        assert len(messages) == 5


@pytest.mark.asyncio
class TestMcpToolCallArgs:
    async def test_load_long_term_memory_called_with_configured_max(self):
        """load_long_term_memory is called with the session id and long_term_max."""
        session = _make_session()
        call_tool_mock = _stub_call_tool({})
        session.mcp_session.call_tool = call_tool_mock
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        with patch("server.chat.settings") as mock_settings:
            mock_settings.server_ollama_model = "test-model"
            mock_settings.server_ollama_num_ctx = 8192
            mock_settings.server_memory_long_term_max = 5
            mock_settings.server_memory_window_size = 10
            stream = await session.chat("hello")
            async for _ in stream:
                pass

        calls_by_name = {c.args[0]: c for c in call_tool_mock.await_args_list}
        lt_call = calls_by_name["load_long_term_memory"]
        assert lt_call.kwargs["arguments"]["long_term_max"] == 5
        assert lt_call.kwargs["arguments"]["session_id"] == str(session.session_id)

    async def test_load_conversation_window_called_with_configured_size(self):
        """load_conversation_window is called with the session id and window_size."""
        session = _make_session()
        call_tool_mock = _stub_call_tool({})
        session.mcp_session.call_tool = call_tool_mock
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        with patch("server.chat.settings") as mock_settings:
            mock_settings.server_ollama_model = "test-model"
            mock_settings.server_ollama_num_ctx = 8192
            mock_settings.server_memory_long_term_max = 3
            mock_settings.server_memory_window_size = 7
            stream = await session.chat("hello")
            async for _ in stream:
                pass

        calls_by_name = {c.args[0]: c for c in call_tool_mock.await_args_list}
        win_call = calls_by_name["load_conversation_window"]
        assert win_call.kwargs["arguments"]["window_size"] == 7
        assert win_call.kwargs["arguments"]["session_id"] == str(session.session_id)

    async def test_num_ctx_passed_to_ollama(self):
        """client.chat() must receive num_ctx in its options dict."""
        session = _make_session()
        session.mcp_session.call_tool = _stub_call_tool({})
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        with patch("server.chat.settings") as mock_settings:
            mock_settings.server_ollama_model = "test-model"
            mock_settings.server_ollama_num_ctx = 4096
            mock_settings.server_memory_long_term_max = 3
            mock_settings.server_memory_window_size = 10
            stream = await session.chat("hello")
            async for _ in stream:
                pass

        call_kwargs = session.client.chat.call_args[1]
        assert call_kwargs["options"]["num_ctx"] == 4096

    async def test_persist_message_called_for_user_turn(self):
        """persist_message is called concurrently for the user turn before streaming."""
        session = _make_session()
        call_tool_mock = _stub_call_tool({})
        session.mcp_session.call_tool = call_tool_mock
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        stream = await session.chat("user input")
        async for _ in stream:
            pass

        persist_calls = [
            c for c in call_tool_mock.await_args_list if c.args[0] == "persist_message"
        ]
        user_persist = next(
            (c for c in persist_calls if c.kwargs["arguments"]["role"] == "user"),
            None,
        )
        assert user_persist is not None
        assert user_persist.kwargs["arguments"]["content"] == "user input"

    async def test_persist_message_called_for_assistant_turn_after_stream(self):
        """persist_message is called for the assistant after the stream is consumed."""
        session = _make_session()
        call_tool_mock = _stub_call_tool({})
        session.mcp_session.call_tool = call_tool_mock
        session.client.chat.return_value = iter(
            [
                {"message": {"content": "Hello"}},
                {"message": {"content": " there"}},
            ]
        )

        stream = await session.chat("hi")
        async for _ in stream:
            pass

        persist_calls = [
            c for c in call_tool_mock.await_args_list if c.args[0] == "persist_message"
        ]
        assistant_persist = next(
            (c for c in persist_calls if c.kwargs["arguments"]["role"] == "assistant"),
            None,
        )
        assert assistant_persist is not None
        assert assistant_persist.kwargs["arguments"]["content"] == "Hello there"


@pytest.mark.asyncio
class TestEdgeCases:
    async def test_malformed_window_json_is_ignored(self):
        """If window_content is not valid JSON, it is silently skipped."""
        session = _make_session()
        session.mcp_session.call_tool = _stub_call_tool(
            {"load_conversation_window": "not valid json {{"}
        )
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        stream = await session.chat("hello")
        async for _ in stream:
            pass

        messages = session.client.chat.call_args[1]["messages"]
        assert messages[-1] == {"role": "user", "content": "hello"}

    async def test_persist_assistant_failure_does_not_raise(self):
        """An error persisting the assistant message must not propagate to the caller."""
        session = _make_session()

        async def _failing_call_tool(name, *, arguments=None, **kwargs):
            if (
                name == "persist_message"
                and arguments
                and arguments.get("role") == "assistant"
            ):
                raise RuntimeError("db down")
            return _make_tool_result("")

        session.mcp_session.call_tool = AsyncMock(side_effect=_failing_call_tool)
        session.client.chat.return_value = iter([{"message": {"content": "ok"}}])

        stream = await session.chat("hello")
        tokens = [t async for t in stream]

        assert "".join(tokens) == "ok"
