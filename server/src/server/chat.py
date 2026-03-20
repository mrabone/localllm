import asyncio
import json
import logging
import uuid
from enum import Enum
from dataclasses import dataclass
from typing import AsyncGenerator, Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from mcp import ClientSession
from mcp.types import TextContent, Tool
from ollama import AsyncClient
from typing_extensions import TypedDict

from server.config import settings

logger = logging.getLogger(__name__)

TOOL_LOAD_LONG_TERM_MEMORY = "load_long_term_memory"
TOOL_LOAD_CONVERSATION_WINDOW = "load_conversation_window"
TOOL_PERSIST_MESSAGE = "persist_message"


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


def build_tool_result_message(call_id: str, name: str, content: str) -> dict:
    """Build a tool result message dictionary to be appended to the conversation."""
    return {
        "role": Role.TOOL.value,
        "content": f"[Tool result for {name} (id={call_id})]\n{content}",
    }


def mcp_tools_to_ollama_schemas(tools: list[Tool]) -> list[dict]:
    """Convert MCP Tool objects to the Ollama JSON Schema tool format.

    The returned list is passed directly to the Ollama `tools=` parameter when
    calling FunctionGemma.  Ollama handles the FunctionGemma-specific prompt
    formatting internally.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


class GraphState(TypedDict):
    session_id: str
    user_input: str
    mcp_session: ClientSession
    ollama_client: AsyncClient
    chat_model: str
    function_calling_model: str
    mcp_tools: list[Tool]
    messages: list[dict]
    tool_results: list[dict]
    loop_count: int
    resolved_answer: str | None


async def load_context_node(state: GraphState) -> dict:
    """Load memory, conversation window, and persist the user turn concurrently.

    Issues three MCP calls in parallel to minimise pre-stream latency.  Failures
    are logged and treated as empty results so the turn can continue.
    """
    mcp_session = state["mcp_session"]
    session_id = state["session_id"]
    user_input = state["user_input"]

    async def _call_tool(name: str, arguments: dict) -> str:
        result = await mcp_session.call_tool(name, arguments=arguments)
        if result.content:
            first = result.content[0]
            return first.text if isinstance(first, TextContent) else ""
        return ""

    gather_results = await asyncio.gather(
        _call_tool(
            TOOL_LOAD_LONG_TERM_MEMORY,
            {
                "session_id": session_id,
                "long_term_max": settings.server_memory_long_term_max,
            },
        ),
        _call_tool(
            TOOL_LOAD_CONVERSATION_WINDOW,
            {
                "session_id": session_id,
                "window_size": settings.server_memory_window_size,
            },
        ),
        _call_tool(
            TOOL_PERSIST_MESSAGE,
            {
                "session_id": session_id,
                "role": Role.USER.value,
                "content": user_input,
            },
        ),
        return_exceptions=True,
    )

    tool_names = [
        TOOL_LOAD_LONG_TERM_MEMORY,
        TOOL_LOAD_CONVERSATION_WINDOW,
        TOOL_PERSIST_MESSAGE,
    ]
    resolved: list[str] = []
    for name, result in zip(tool_names, gather_results):
        if isinstance(result, BaseException):
            logger.warning(
                "MCP tool '%s' failed during context gather (session=%s): %s",
                name,
                session_id,
                result,
            )
            resolved.append("")
        else:
            resolved.append(result)  # type: ignore[arg-type]

    long_term_content, window_content, _ = resolved

    orientation = {
        "role": Role.SYSTEM.value,
        "content": (
            "You are a helpful assistant. "
            "On each turn you may be provided with additional context:\n"
            "- Long-term memories: facts extracted from previous conversations "
            "with this user\n"
            "- Recent conversation history: the last few turns verbatim\n"
            "- Knowledge base documents: relevant documents retrieved for this "
            "specific question\n"
            "Use all provided context to give the most accurate and helpful "
            "answer possible."
        ),
    }

    messages: list[dict] = [orientation]

    if long_term_content:
        messages.append({"role": Role.SYSTEM.value, "content": long_term_content})

    if window_content:
        try:
            window_turns = json.loads(window_content)
            messages.extend(window_turns)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse window content from MCP tool.")

    messages.append({"role": Role.USER.value, "content": user_input})

    return {"messages": messages}


async def decide_tools_node(state: GraphState) -> dict:
    """Ask FunctionGemma whether any tools should be called.

    Passes the full message list and MCP tool schemas to FunctionGemma via the
    standard Ollama `tools=` parameter.  Ollama abstracts FunctionGemma's
    internal token format and returns tool calls via `response.message.tool_calls`.

    Returns tool_results with the parsed ToolCall list stored under the key
    ``_pending_tool_calls`` for the execute_tools_node to consume, or an empty
    list if FunctionGemma decided no tool call is needed.
    """
    client = state["ollama_client"]
    model = state["function_calling_model"]
    messages = state["messages"] + state["tool_results"]
    ollama_schemas = mcp_tools_to_ollama_schemas(state["mcp_tools"])

    response = await client.chat(
        model=model,
        messages=messages,
        tools=ollama_schemas,
        stream=False,
        options={"num_ctx": settings.server_ollama_num_ctx},
    )

    raw_tool_calls = response.message.tool_calls or []
    pending: list[ToolCall] = []
    for tc in raw_tool_calls:
        call_id = str(uuid.uuid4())
        name = tc.function.name
        arguments = dict(tc.function.arguments) if tc.function.arguments else {}
        pending.append(ToolCall(id=call_id, name=name, arguments=arguments))

    if pending:
        logger.info(
            "Session %s: FunctionGemma requested %d tool call(s): %s",
            state["session_id"],
            len(pending),
            [c.name for c in pending],
        )

    return {"tool_results": state["tool_results"] + [{"_pending": pending}]}


def _route_after_decide(
    state: GraphState,
) -> Literal["execute_tools", "generate_response"]:
    """Route to execute_tools if pending calls exist, otherwise generate_response."""
    tool_results = state["tool_results"]
    if tool_results:
        last = tool_results[-1]
        pending = last.get("_pending", [])
        if pending:
            return "execute_tools"
    return "generate_response"


def _route_after_execute(
    state: GraphState,
) -> Literal["decide_tools", "generate_response"]:
    """Route back to decide_tools while under the loop cap, else generate_response."""
    if state["loop_count"] < settings.server_tool_call_max_loops:
        return "decide_tools"
    return "generate_response"


async def execute_tools_node(state: GraphState) -> dict:
    """Execute all pending tool calls concurrently via MCP.

    Pops the ``_pending`` sentinel from tool_results and replaces it with the
    actual tool result messages.  Unknown tools and execution failures are
    returned as error strings so the model can decide how to proceed.
    """
    mcp_session = state["mcp_session"]
    session_id = state["session_id"]

    tool_results = list(state["tool_results"])
    pending_entry = tool_results.pop()
    pending: list[ToolCall] = pending_entry.get("_pending", [])

    known_names = {t.name for t in state["mcp_tools"]}

    async def _run_one(call: ToolCall) -> dict:
        if call.name not in known_names:
            available = ", ".join(sorted(known_names)) or "none"
            error = (
                f"ERROR: Unknown tool '{call.name}'. Available tools are: {available}"
            )
            return build_tool_result_message(call.id, call.name, error)

        try:
            result = await mcp_session.call_tool(call.name, arguments=call.arguments)
            first_content = result.content[0] if result.content else None
            content = (
                first_content.text if isinstance(first_content, TextContent) else ""
            )
            if not content:
                content = "No results found. Answer the user's question using your own knowledge."
            return build_tool_result_message(call.id, call.name, content)
        except Exception as exc:
            logger.warning(
                "Tool call '%s' failed (session=%s, arguments=%s): %s",
                call.name,
                session_id,
                call.arguments,
                exc,
            )
            error = (
                f"ERROR: Tool '{call.name}' failed with: {exc}. "
                "Continue with the information you have."
            )
            return build_tool_result_message(call.id, call.name, error)

    new_results = list(await asyncio.gather(*(_run_one(c) for c in pending)))
    return {
        "tool_results": tool_results + new_results,
        "loop_count": state["loop_count"] + 1,
    }


async def generate_response_node(state: GraphState) -> dict:
    """Stream the final response from Gemma3 and persist the assistant turn.

    Combines the base messages with any accumulated tool result messages before
    calling Gemma3.  The full assembled text is persisted via MCP after streaming.
    Returns the assembled answer in ``resolved_answer`` for the caller to stream.
    """
    client = state["ollama_client"]
    model = state["chat_model"]
    mcp_session = state["mcp_session"]
    session_id = state["session_id"]

    tool_result_messages = [r for r in state["tool_results"] if "_pending" not in r]
    messages = state["messages"] + tool_result_messages

    assembled_tokens: list[str] = []
    stream = await client.chat(
        model=model,
        messages=messages,
        stream=True,
        options={"num_ctx": settings.server_ollama_num_ctx},
    )
    async for chunk in stream:
        token = chunk.message.content or ""
        assembled_tokens.append(token)

    assembled = "".join(assembled_tokens)

    try:
        await mcp_session.call_tool(
            TOOL_PERSIST_MESSAGE,
            arguments={
                "session_id": session_id,
                "role": Role.ASSISTANT.value,
                "content": assembled,
            },
        )
    except Exception as exc:
        logger.error(
            "Failed to persist assistant message (session=%s): %s",
            session_id,
            exc,
            exc_info=True,
        )

    return {"resolved_answer": assembled}


def build_chat_graph() -> CompiledStateGraph:
    """Assemble the LangGraph state graph for one chat turn."""
    graph = StateGraph(GraphState)

    graph.add_node("load_context", load_context_node)
    graph.add_node("decide_tools", decide_tools_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("generate_response", generate_response_node)

    graph.set_entry_point("load_context")

    graph.add_conditional_edges(
        "load_context",
        lambda state: "decide_tools" if state["mcp_tools"] else "generate_response",
        {"decide_tools": "decide_tools", "generate_response": "generate_response"},
    )

    graph.add_conditional_edges(
        "decide_tools",
        _route_after_decide,
        {"execute_tools": "execute_tools", "generate_response": "generate_response"},
    )

    graph.add_conditional_edges(
        "execute_tools",
        _route_after_execute,
        {"decide_tools": "decide_tools", "generate_response": "generate_response"},
    )
    graph.add_edge("generate_response", END)

    return graph.compile()


_chat_graph = build_chat_graph()


async def run_chat_graph(
    session_id: uuid.UUID,
    user_input: str,
    mcp_session: ClientSession,
    ollama_client: AsyncClient,
    chat_model: str,
    function_calling_model: str,
    mcp_tools: list[Tool],
) -> AsyncGenerator[str, None]:
    """Run one chat turn through the LangGraph graph and stream tokens.

    This is a direct replacement for ``ChatSession.chat()``.  The caller
    receives an async generator that yields token strings, identical to the
    previous interface.
    """
    initial_state: GraphState = {
        "session_id": str(session_id),
        "user_input": user_input,
        "mcp_session": mcp_session,
        "ollama_client": ollama_client,
        "chat_model": chat_model,
        "function_calling_model": function_calling_model,
        "mcp_tools": mcp_tools,
        "messages": [],
        "tool_results": [],
        "loop_count": 0,
        "resolved_answer": None,
    }

    final_state = await _chat_graph.ainvoke(initial_state)
    answer = final_state.get("resolved_answer") or ""

    async def _token_stream() -> AsyncGenerator[str, None]:
        for token in answer:
            yield token

    return _token_stream()
