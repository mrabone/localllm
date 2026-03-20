import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent, Tool

from server.chat import (
    GraphState,
    Role,
    ToolCall,
    build_tool_result_message,
    decide_tools_node,
    execute_tools_node,
    generate_response_node,
    load_context_node,
    mcp_tools_to_ollama_schemas,
)


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


def _make_base_state(**overrides) -> GraphState:
    """Return a minimal valid GraphState for unit testing individual nodes."""
    mcp_session = AsyncMock()
    mcp_session.call_tool = AsyncMock(return_value=_make_tool_result(""))
    ollama_client = AsyncMock()

    state: GraphState = {
        "session_id": str(uuid.uuid4()),
        "user_input": "hello",
        "mcp_session": mcp_session,
        "ollama_client": ollama_client,
        "chat_model": "test-chat-model",
        "function_calling_model": "test-fc-model",
        "mcp_tools": [],
        "messages": [],
        "tool_results": [],
        "loop_count": 0,
        "resolved_answer": None,
    }
    for key, value in overrides.items():
        state[key] = value  # type: ignore[literal-required]
    return state


def _stub_call_tool(tool_responses: dict):
    """Return an AsyncMock for call_tool that dispatches by tool name."""

    async def _call_tool(name, *, arguments=None, **kwargs):
        text = tool_responses.get(name, "")
        if text == "":
            return _make_empty_tool_result()
        return _make_tool_result(text)

    return AsyncMock(side_effect=_call_tool)


class TestMcpToolsToOllamaSchemas:
    def test_returns_empty_for_no_tools(self):
        assert mcp_tools_to_ollama_schemas([]) == []

    def test_wraps_each_tool_as_function_type(self):
        tool = _make_mcp_tool("search_kb", "Search the knowledge base")
        schemas = mcp_tools_to_ollama_schemas([tool])
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"

    def test_preserves_tool_name(self):
        tool = _make_mcp_tool("my_tool")
        schemas = mcp_tools_to_ollama_schemas([tool])
        assert schemas[0]["function"]["name"] == "my_tool"

    def test_preserves_tool_description(self):
        tool = _make_mcp_tool("t", "Does something important")
        schemas = mcp_tools_to_ollama_schemas([tool])
        assert schemas[0]["function"]["description"] == "Does something important"

    def test_empty_description_becomes_empty_string(self):
        tool = _make_mcp_tool("t", "")
        schemas = mcp_tools_to_ollama_schemas([tool])
        assert schemas[0]["function"]["description"] == ""

    def test_preserves_input_schema(self):
        params = {"query": {"type": "string"}}
        tool = _make_mcp_tool("search", params=params)
        schemas = mcp_tools_to_ollama_schemas([tool])
        assert schemas[0]["function"]["parameters"]["properties"] == params

    def test_null_input_schema_becomes_empty_object_schema(self):
        tool = Tool(name="t", inputSchema={"type": "object", "properties": {}})
        schemas = mcp_tools_to_ollama_schemas([tool])
        assert schemas[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }

    def test_converts_multiple_tools(self):
        tools = [
            _make_mcp_tool("alpha"),
            _make_mcp_tool("beta"),
            _make_mcp_tool("gamma"),
        ]
        schemas = mcp_tools_to_ollama_schemas(tools)
        assert len(schemas) == 3
        names = [s["function"]["name"] for s in schemas]
        assert names == ["alpha", "beta", "gamma"]


@pytest.mark.asyncio
class TestLoadContextNode:
    async def test_orientation_is_always_first_message(self):
        state = _make_base_state()
        result = await load_context_node(state)
        messages = result["messages"]
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]

    async def test_user_turn_is_appended_last(self):
        state = _make_base_state(user_input="my question")
        result = await load_context_node(state)
        messages = result["messages"]
        assert messages[-1] == {"role": "user", "content": "my question"}

    async def test_no_extra_messages_when_context_empty(self):
        state = _make_base_state(user_input="hello")
        state["mcp_session"].call_tool = _stub_call_tool({})
        result = await load_context_node(state)
        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "hello"}

    async def test_long_term_memory_injected_after_orientation(self):
        state = _make_base_state()
        state["mcp_session"].call_tool = _stub_call_tool(
            {"load_long_term_memory": "User is a Python developer"}
        )
        result = await load_context_node(state)
        messages = result["messages"]
        assert messages[1] == {
            "role": "system",
            "content": "User is a Python developer",
        }

    async def test_window_turns_inserted_after_long_term_memory(self):
        window_turns = [
            {"role": "user", "content": "prev question"},
            {"role": "assistant", "content": "prev answer"},
        ]
        state = _make_base_state(user_input="current question")
        state["mcp_session"].call_tool = _stub_call_tool(
            {
                "load_long_term_memory": "some memory",
                "load_conversation_window": json.dumps(window_turns),
            }
        )
        result = await load_context_node(state)
        messages = result["messages"]
        assert messages[2] == {"role": "user", "content": "prev question"}
        assert messages[3] == {"role": "assistant", "content": "prev answer"}

    async def test_full_context_order(self):
        window_turns = [
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a"},
        ]
        state = _make_base_state(user_input="current q")
        state["mcp_session"].call_tool = _stub_call_tool(
            {
                "load_long_term_memory": "memory facts",
                "load_conversation_window": json.dumps(window_turns),
            }
        )
        result = await load_context_node(state)
        messages = result["messages"]
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]
        assert messages[1] == {"role": "system", "content": "memory facts"}
        assert messages[2] == {"role": "user", "content": "prev q"}
        assert messages[3] == {"role": "assistant", "content": "prev a"}
        assert messages[4] == {"role": "user", "content": "current q"}
        assert len(messages) == 5

    async def test_malformed_window_json_is_ignored(self):
        state = _make_base_state(user_input="hello")
        state["mcp_session"].call_tool = _stub_call_tool(
            {"load_conversation_window": "not valid json {{"}
        )
        result = await load_context_node(state)
        messages = result["messages"]
        assert messages[-1] == {"role": "user", "content": "hello"}

    async def test_mcp_failure_is_treated_as_empty_result(self):
        """A failing MCP call during context gather must not propagate — treated as ''."""
        state = _make_base_state(user_input="hello")

        async def _failing_call_tool(name, *, arguments=None, **kwargs):
            if name == "load_long_term_memory":
                raise RuntimeError("network error")
            return _make_empty_tool_result()

        state["mcp_session"].call_tool = AsyncMock(side_effect=_failing_call_tool)

        result = await load_context_node(state)
        messages = result["messages"]
        assert messages[0]["role"] == "system"
        assert messages[-1] == {"role": "user", "content": "hello"}

    async def test_load_long_term_memory_called_with_configured_max(self):
        state = _make_base_state()
        call_tool_mock = _stub_call_tool({})
        state["mcp_session"].call_tool = call_tool_mock

        with patch("server.chat.settings") as mock_settings:
            mock_settings.server_memory_long_term_max = 5
            mock_settings.server_memory_window_size = 10
            await load_context_node(state)

        calls_by_name = {c.args[0]: c for c in call_tool_mock.await_args_list}
        lt_call = calls_by_name["load_long_term_memory"]
        assert lt_call.kwargs["arguments"]["long_term_max"] == 5
        assert lt_call.kwargs["arguments"]["session_id"] == state["session_id"]

    async def test_load_conversation_window_called_with_configured_size(self):
        state = _make_base_state()
        call_tool_mock = _stub_call_tool({})
        state["mcp_session"].call_tool = call_tool_mock

        with patch("server.chat.settings") as mock_settings:
            mock_settings.server_memory_long_term_max = 3
            mock_settings.server_memory_window_size = 7
            await load_context_node(state)

        calls_by_name = {c.args[0]: c for c in call_tool_mock.await_args_list}
        win_call = calls_by_name["load_conversation_window"]
        assert win_call.kwargs["arguments"]["window_size"] == 7
        assert win_call.kwargs["arguments"]["session_id"] == state["session_id"]

    async def test_persist_message_called_for_user_turn(self):
        state = _make_base_state(user_input="user input")
        call_tool_mock = _stub_call_tool({})
        state["mcp_session"].call_tool = call_tool_mock

        await load_context_node(state)

        persist_calls = [
            c for c in call_tool_mock.await_args_list if c.args[0] == "persist_message"
        ]
        user_persist = next(
            (c for c in persist_calls if c.kwargs["arguments"]["role"] == "user"),
            None,
        )
        assert user_persist is not None
        assert user_persist.kwargs["arguments"]["content"] == "user input"
        assert user_persist.kwargs["arguments"]["session_id"] == state["session_id"]


@pytest.mark.asyncio
class TestDecideToolsNode:
    def _make_fc_response(self, tool_calls=None):
        response = MagicMock()
        response.message.tool_calls = tool_calls or []
        return response

    def _make_tc(self, name: str, arguments: dict | None = None):
        """Build a mock Ollama tool_call object."""
        tc = MagicMock()
        tc.function.name = name
        tc.function.arguments = arguments or {}
        return tc

    async def test_no_pending_calls_when_fc_returns_none(self):
        state = _make_base_state(
            messages=[{"role": "user", "content": "hello"}],
            mcp_tools=[_make_mcp_tool("search_kb")],
        )
        state["ollama_client"].chat = AsyncMock(
            return_value=self._make_fc_response(tool_calls=None)
        )

        result = await decide_tools_node(state)
        pending_entry = result["tool_results"][-1]
        assert pending_entry["_pending"] == []

    async def test_tool_calls_returned_as_pending(self):
        tool = _make_mcp_tool("search_kb")
        state = _make_base_state(
            messages=[{"role": "user", "content": "hello"}],
            mcp_tools=[tool],
        )
        state["ollama_client"].chat = AsyncMock(
            return_value=self._make_fc_response(
                tool_calls=[self._make_tc("search_kb", {"query": "foo"})]
            )
        )

        result = await decide_tools_node(state)
        pending = result["tool_results"][-1]["_pending"]
        assert len(pending) == 1
        assert pending[0].name == "search_kb"
        assert pending[0].arguments == {"query": "foo"}

    async def test_tool_calls_have_auto_generated_ids(self):
        tool = _make_mcp_tool("my_tool")
        state = _make_base_state(
            messages=[{"role": "user", "content": "hello"}],
            mcp_tools=[tool],
        )
        state["ollama_client"].chat = AsyncMock(
            return_value=self._make_fc_response(tool_calls=[self._make_tc("my_tool")])
        )

        result = await decide_tools_node(state)
        pending = result["tool_results"][-1]["_pending"]
        assert len(pending) == 1
        assert pending[0].id  # non-empty UUID string

    async def test_ollama_schemas_passed_to_fc_model(self):
        tool = _make_mcp_tool("search_kb", "Search", {"query": {"type": "string"}})
        state = _make_base_state(
            messages=[{"role": "user", "content": "hello"}],
            mcp_tools=[tool],
        )
        fc_mock = AsyncMock(return_value=self._make_fc_response())
        state["ollama_client"].chat = fc_mock

        await decide_tools_node(state)

        call_kwargs = fc_mock.call_args[1]
        tools_arg = call_kwargs["tools"]
        assert len(tools_arg) == 1
        assert tools_arg[0]["function"]["name"] == "search_kb"

    async def test_fc_model_receives_merged_messages_and_tool_results(self):
        existing_tool_result = {"role": "tool", "content": "previous result"}
        state = _make_base_state(
            messages=[{"role": "user", "content": "hello"}],
            tool_results=[existing_tool_result],
            mcp_tools=[_make_mcp_tool("t")],
        )
        fc_mock = AsyncMock(return_value=self._make_fc_response())
        state["ollama_client"].chat = fc_mock

        await decide_tools_node(state)

        call_kwargs = fc_mock.call_args[1]
        messages_passed = call_kwargs["messages"]
        assert {"role": "user", "content": "hello"} in messages_passed
        assert existing_tool_result in messages_passed

    async def test_existing_tool_results_preserved(self):
        existing = {"role": "tool", "content": "prior result"}
        state = _make_base_state(
            messages=[{"role": "user", "content": "hello"}],
            tool_results=[existing],
            mcp_tools=[_make_mcp_tool("t")],
        )
        state["ollama_client"].chat = AsyncMock(return_value=self._make_fc_response())

        result = await decide_tools_node(state)
        assert existing in result["tool_results"]

    async def test_multiple_tool_calls_parsed(self):
        tool_a = _make_mcp_tool("tool_a")
        tool_b = _make_mcp_tool("tool_b")
        state = _make_base_state(
            messages=[{"role": "user", "content": "hello"}],
            mcp_tools=[tool_a, tool_b],
        )
        state["ollama_client"].chat = AsyncMock(
            return_value=self._make_fc_response(
                tool_calls=[
                    self._make_tc("tool_a"),
                    self._make_tc("tool_b", {"x": 1}),
                ]
            )
        )

        result = await decide_tools_node(state)
        pending = result["tool_results"][-1]["_pending"]
        assert len(pending) == 2
        assert pending[0].name == "tool_a"
        assert pending[1].name == "tool_b"
        assert pending[1].arguments == {"x": 1}


@pytest.mark.asyncio
class TestExecuteToolsNode:
    async def test_known_tool_result_returned_as_tool_message(self):
        tool = _make_mcp_tool("search_kb")
        pending = ToolCall(id="c1", name="search_kb", arguments={"query": "foo"})
        state = _make_base_state(
            mcp_tools=[tool],
            tool_results=[{"_pending": [pending]}],
        )
        state["mcp_session"].call_tool = AsyncMock(
            return_value=_make_tool_result("Found document 3.")
        )

        result = await execute_tools_node(state)
        tool_messages = result["tool_results"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["role"] == Role.TOOL.value
        assert "Found document 3." in tool_messages[0]["content"]

    async def test_unknown_tool_returns_error_message(self):
        tool = _make_mcp_tool("real_tool")
        pending = ToolCall(id="c1", name="fake_tool", arguments={})
        state = _make_base_state(
            mcp_tools=[tool],
            tool_results=[{"_pending": [pending]}],
        )

        result = await execute_tools_node(state)
        error_msg = result["tool_results"][0]
        assert "ERROR" in error_msg["content"]
        assert "fake_tool" in error_msg["content"]

    async def test_tool_execution_failure_returns_error_message(self):
        tool = _make_mcp_tool("fragile_tool")
        pending = ToolCall(id="c1", name="fragile_tool", arguments={})
        state = _make_base_state(
            mcp_tools=[tool],
            tool_results=[{"_pending": [pending]}],
        )
        state["mcp_session"].call_tool = AsyncMock(
            side_effect=RuntimeError("service unavailable")
        )

        result = await execute_tools_node(state)
        error_msg = result["tool_results"][0]
        assert "ERROR" in error_msg["content"]
        assert "fragile_tool" in error_msg["content"]

    async def test_empty_tool_result_replaced_with_fallback_message(self):
        tool = _make_mcp_tool("search_kb")
        pending = ToolCall(id="c1", name="search_kb", arguments={})
        state = _make_base_state(
            mcp_tools=[tool],
            tool_results=[{"_pending": [pending]}],
        )
        state["mcp_session"].call_tool = AsyncMock(
            return_value=_make_empty_tool_result()
        )

        result = await execute_tools_node(state)
        tool_message = result["tool_results"][0]
        assert "No results found" in tool_message["content"]

    async def test_multiple_pending_calls_all_executed(self):
        tool_a = _make_mcp_tool("tool_a")
        tool_b = _make_mcp_tool("tool_b")
        pending = [
            ToolCall(id="c1", name="tool_a", arguments={}),
            ToolCall(id="c2", name="tool_b", arguments={}),
        ]
        state = _make_base_state(
            mcp_tools=[tool_a, tool_b],
            tool_results=[{"_pending": pending}],
        )

        async def _call_tool_side_effect(name, *, arguments=None, **kwargs):
            return _make_tool_result(f"{name} result")

        state["mcp_session"].call_tool = AsyncMock(side_effect=_call_tool_side_effect)

        result = await execute_tools_node(state)
        assert len(result["tool_results"]) == 2
        names_in_content = [r["content"] for r in result["tool_results"]]
        assert any("tool_a" in c for c in names_in_content)
        assert any("tool_b" in c for c in names_in_content)

    async def test_loop_count_incremented(self):
        tool = _make_mcp_tool("t")
        pending = ToolCall(id="c1", name="t", arguments={})
        state = _make_base_state(
            mcp_tools=[tool],
            tool_results=[{"_pending": [pending]}],
            loop_count=1,
        )
        state["mcp_session"].call_tool = AsyncMock(return_value=_make_tool_result("ok"))

        result = await execute_tools_node(state)
        assert result["loop_count"] == 2

    async def test_pending_sentinel_removed_from_tool_results(self):
        """The _pending entry must be popped and not left in the final tool_results."""
        tool = _make_mcp_tool("t")
        pending = ToolCall(id="c1", name="t", arguments={})
        state = _make_base_state(
            mcp_tools=[tool],
            tool_results=[{"_pending": [pending]}],
        )
        state["mcp_session"].call_tool = AsyncMock(
            return_value=_make_tool_result("result")
        )

        result = await execute_tools_node(state)
        for entry in result["tool_results"]:
            assert "_pending" not in entry

    async def test_prior_tool_results_preserved(self):
        """Results from previous loop iterations must remain in tool_results."""
        prior = build_tool_result_message("c0", "prior_tool", "prior result")
        tool = _make_mcp_tool("t")
        pending = ToolCall(id="c1", name="t", arguments={})
        state = _make_base_state(
            mcp_tools=[tool],
            tool_results=[prior, {"_pending": [pending]}],
        )
        state["mcp_session"].call_tool = AsyncMock(
            return_value=_make_tool_result("new result")
        )

        result = await execute_tools_node(state)
        assert prior in result["tool_results"]


@pytest.mark.asyncio
class TestGenerateResponseNode:
    async def test_streams_tokens_and_returns_assembled_answer(self):
        state = _make_base_state(
            messages=[{"role": "user", "content": "hi"}],
        )
        state["ollama_client"].chat = AsyncMock(
            return_value=_make_stream_response(["Hello", " world"])
        )

        result = await generate_response_node(state)
        assert result["resolved_answer"] == "Hello world"

    async def test_tool_result_messages_included_in_llm_call(self):
        tool_result = build_tool_result_message("c1", "search_kb", "doc content")
        state = _make_base_state(
            messages=[{"role": "user", "content": "hi"}],
            tool_results=[tool_result],
        )
        ollama_mock = AsyncMock(return_value=_make_stream_response(["ok"]))
        state["ollama_client"].chat = ollama_mock

        await generate_response_node(state)

        messages_passed = ollama_mock.call_args[1]["messages"]
        assert tool_result in messages_passed

    async def test_pending_sentinel_excluded_from_llm_call(self):
        """The _pending sentinel entry must never be forwarded to the chat model."""
        pending_entry = {"_pending": [ToolCall(id="c1", name="t", arguments={})]}
        state = _make_base_state(
            messages=[{"role": "user", "content": "hi"}],
            tool_results=[pending_entry],
        )
        ollama_mock = AsyncMock(return_value=_make_stream_response(["ok"]))
        state["ollama_client"].chat = ollama_mock

        await generate_response_node(state)

        messages_passed = ollama_mock.call_args[1]["messages"]
        assert pending_entry not in messages_passed
        for msg in messages_passed:
            assert "_pending" not in msg

    async def test_assistant_message_persisted_after_stream(self):
        state = _make_base_state(
            messages=[{"role": "user", "content": "hi"}],
        )
        state["ollama_client"].chat = AsyncMock(
            return_value=_make_stream_response(["Hello", " there"])
        )
        call_tool_mock = AsyncMock(return_value=_make_empty_tool_result())
        state["mcp_session"].call_tool = call_tool_mock

        await generate_response_node(state)

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
        state = _make_base_state(
            messages=[{"role": "user", "content": "hi"}],
        )
        state["ollama_client"].chat = AsyncMock(
            return_value=_make_stream_response(["ok"])
        )

        async def _failing_persist(name, *, arguments=None, **kwargs):
            raise RuntimeError("db down")

        state["mcp_session"].call_tool = AsyncMock(side_effect=_failing_persist)

        result = await generate_response_node(state)
        assert result["resolved_answer"] == "ok"

    async def test_num_ctx_passed_to_ollama(self):
        state = _make_base_state(
            messages=[{"role": "user", "content": "hi"}],
        )
        ollama_mock = AsyncMock(return_value=_make_stream_response(["ok"]))
        state["ollama_client"].chat = ollama_mock

        with patch("server.chat.settings") as mock_settings:
            mock_settings.server_ollama_num_ctx = 4096
            await generate_response_node(state)

        call_kwargs = ollama_mock.call_args[1]
        assert call_kwargs["options"]["num_ctx"] == 4096
