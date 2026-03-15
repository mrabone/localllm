import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator

from mcp import ClientSession
from mcp.types import Tool
from ollama import Client

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


def build_tools_system_message(tools: list[Tool]) -> str:
    """Build the system message that instructs the LLM how to call tools.

    Renders each tool's name, description, and input schema into a plain-text
    block the model can follow.  Returns an empty string when no tools are
    available so the caller can skip appending it.
    """
    if not tools:
        return ""

    tool_descriptions = []
    for tool in tools:
        schema = tool.inputSchema or {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        params = []
        for param_name, param_schema in properties.items():
            param_type = param_schema.get("type", "any")
            is_required = param_name in required
            default = param_schema.get("default")
            description = param_schema.get("description", "")

            param_str = f"  - {param_name} ({param_type}"
            if not is_required and default is not None:
                param_str += f", default={default!r}"
            param_str += ")"
            if description:
                param_str += f": {description}"
            params.append(param_str)

        params_block = "\n".join(params) if params else "  (no parameters)"
        tool_descriptions.append(
            f"### {tool.name}\n"
            f"{tool.description or 'No description provided.'}\n"
            f"Parameters:\n{params_block}"
        )

    tools_block = "\n\n".join(tool_descriptions)

    return (
        "## Available Tools\n\n"
        "You may call any of the following tools when you need to retrieve "
        "information or perform an action.  To call a tool, respond with a JSON "
        "object — and nothing else — in exactly this format:\n\n"
        '{"toolCalls": [{"id": "call_1", "name": "<tool_name>", "arguments": {"param": "value"}}]}\n\n'
        "Replace <tool_name> with the tool you want to call and fill the "
        '"arguments" object with the parameter names and values for that tool.\n\n'
        "You may include multiple tool calls in a single response.  After receiving "
        "tool results you will be prompted again to continue your answer.\n\n"
        "Only output the JSON when you want to call a tool.  Do not wrap it in "
        "markdown code fences or any other formatting — raw JSON only.  "
        "When you have enough information to answer the user, respond normally in plain text.\n\n"
        f"{tools_block}"
    )


def parse_tool_calls(response_text: str) -> list[ToolCall]:
    """Extract tool calls from an LLM response string.

    The model is instructed to respond with a bare JSON object when it wants
    to call tools.  This function attempts to parse that object and returns a
    (possibly empty) list of ToolCall instances.

    Both camelCase (``toolCalls``, ``name``, ``arguments``) and snake_case
    (``tool_calls``, ``tool_name``, ``tool_arguments``) key variants are
    accepted so that the parser is tolerant of models that do not strictly
    follow the casing in the system prompt.

    Malformed or non-tool-call responses return an empty list so the caller
    can treat them as plain assistant messages.
    """
    stripped = response_text.strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner_lines = [line for line in lines[1:] if line.strip() != "```"]
        stripped = "\n".join(inner_lines).strip()

    if not stripped.startswith("{"):
        return []

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return []

    raw_calls = payload.get("toolCalls") or payload.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []

    tool_calls = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        call_id = item.get("id") or str(uuid.uuid4())
        name = item.get("name") or item.get("tool_name")
        arguments = item.get("arguments") or item.get("tool_arguments") or {}
        if not name or not isinstance(arguments, dict):
            continue
        tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))

    return tool_calls


def build_tool_result_message(call_id: str, name: str, content: str) -> dict:
    """Build a tool result message dictionary to be appended to the conversation."""
    return {
        "role": Role.TOOL.value,
        "content": f"[Tool result for {name} (id={call_id})]\n{content}",
    }


class ChatSession:
    """Handles a single chat turn for a persistent session.

    Context is assembled in two layers on every turn:

    1. **Long-term memories** — semantic facts extracted by Mem0 from previous
       conversations, retrieved via the MCP ``load_long_term_memory`` tool and
       injected as a ``system`` message.

    2. **Sliding window** — the last ``window_size`` verbatim user/assistant
       turns, retrieved via the MCP ``load_conversation_window`` tool.

    Both retrieval calls are issued concurrently with the user-turn persistence
    call using ``asyncio.gather`` to minimise first-token latency.

    When MCP tools are provided via ``mcp_tools``, the LLM is given a system
    message describing every available tool and instructed to respond with a
    JSON ``toolCalls`` object when it wants to invoke one.  The chat method
    will execute those calls (up to ``server_tool_call_max_loops`` rounds) and
    feed results back before streaming the final answer.  The model may invoke
    ``search_knowledge_base`` itself during this loop when it decides retrieval
    is needed.
    """

    def __init__(
        self,
        session_id: uuid.UUID,
        mcp_session: ClientSession,
        ollama_client: Client,
        thread_pool: ThreadPoolExecutor,
        model: str | None = None,
        mcp_tools: list[Tool] | None = None,
    ) -> None:
        self.session_id = session_id
        self.mcp_session = mcp_session
        self.client = ollama_client
        self.thread_pool = thread_pool
        self.model = model if model is not None else settings.server_ollama_model
        self.mcp_tools = mcp_tools or []

    async def _call_tool(self, name: str, arguments: dict) -> str:
        """Call a named MCP tool and return its first text content."""
        result = await self.mcp_session.call_tool(name, arguments=arguments)
        if result.content:
            return result.content[0].text
        return ""

    async def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[dict]:
        """Execute a list of tool calls concurrently and return result messages.

        Each call is wrapped in its own try/except so a single failure does not
        cancel the others.  Failures are returned as error strings so the LLM
        can decide how to proceed.
        """
        known_names = {t.name for t in self.mcp_tools}

        async def _run_one(call: ToolCall) -> dict:
            if call.name not in known_names:
                available = ", ".join(sorted(known_names)) or "none"
                error = (
                    f"ERROR: Unknown tool '{call.name}'. "
                    f"Available tools are: {available}"
                )
                return build_tool_result_message(call.id, call.name, error)

            try:
                result = await self._call_tool(call.name, call.arguments)
                if not result:
                    result = "No results found. Answer the user's question using your own knowledge."
                return build_tool_result_message(call.id, call.name, result)
            except Exception as exc:
                logger.warning(
                    "Tool call '%s' failed (session=%s, arguments=%s): %s",
                    call.name,
                    self.session_id,
                    call.arguments,
                    exc,
                )
                error = (
                    f"ERROR: Tool '{call.name}' failed with: {exc}. "
                    "Continue with the information you have."
                )
                return build_tool_result_message(call.id, call.name, error)

        return list(await asyncio.gather(*(_run_one(c) for c in tool_calls)))

    async def chat(self, user_input: str) -> AsyncGenerator[str, None]:
        """Process one user turn and return a token stream.

        Fires three MCP tool calls concurrently to minimise pre-stream latency:
        ``load_long_term_memory``, ``load_conversation_window``, and the
        user-turn ``persist_message``.

        If the session was created with ``mcp_tools``, up to
        ``server_tool_call_max_loops`` rounds of tool calling are performed
        before the final response is streamed back to the caller.  The model
        may invoke ``search_knowledge_base`` itself via the tool-calling loop
        when it decides retrieval is needed.

        Returns an async generator that yields token strings.
        """
        session_id_str = str(self.session_id)

        long_term_task = asyncio.create_task(
            self._call_tool(
                TOOL_LOAD_LONG_TERM_MEMORY,
                {
                    "session_id": session_id_str,
                    "long_term_max": settings.server_memory_long_term_max,
                },
            )
        )
        window_task = asyncio.create_task(
            self._call_tool(
                TOOL_LOAD_CONVERSATION_WINDOW,
                {
                    "session_id": session_id_str,
                    "window_size": settings.server_memory_window_size,
                },
            )
        )
        persist_user_task = asyncio.create_task(
            self._call_tool(
                TOOL_PERSIST_MESSAGE,
                {
                    "session_id": session_id_str,
                    "role": Role.USER.value,
                    "content": user_input,
                },
            )
        )

        gather_results = await asyncio.gather(
            long_term_task, window_task, persist_user_task, return_exceptions=True
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
                    session_id_str,
                    result,
                )
                resolved.append("")
            else:
                resolved.append(result)

        long_term_content, window_content, _ = resolved

        tools_guidance = build_tools_system_message(self.mcp_tools)

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
                "answer possible." + (f"\n\n{tools_guidance}" if tools_guidance else "")
            ),
        }

        context_messages: list[dict] = [orientation]

        if long_term_content:
            context_messages.append(
                {"role": Role.SYSTEM.value, "content": long_term_content}
            )

        if window_content:
            try:
                window_turns = json.loads(window_content)
                context_messages.extend(window_turns)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse window content from MCP tool.")

        context_messages.append({"role": Role.USER.value, "content": user_input})

        messages = context_messages
        resolved_answer: str | None = None
        loop = asyncio.get_running_loop()
        if self.mcp_tools:
            for _ in range(settings.server_tool_call_max_loops):
                probe_response = await loop.run_in_executor(
                    self.thread_pool,
                    lambda msgs=messages: self.client.chat(
                        model=self.model,
                        messages=msgs,
                        stream=False,
                        options={"num_ctx": settings.server_ollama_num_ctx},
                    ),
                )
                assistant_text = probe_response["message"]["content"]
                tool_calls = parse_tool_calls(assistant_text)

                if not tool_calls:
                    if not assistant_text.strip().startswith("{"):
                        resolved_answer = assistant_text
                    break

                logger.info(
                    "Session %s: executing %d tool call(s): %s",
                    self.session_id,
                    len(tool_calls),
                    [c.name for c in tool_calls],
                )

                tool_result_messages = await self._execute_tool_calls(tool_calls)
                messages = (
                    messages
                    + [{"role": Role.ASSISTANT.value, "content": assistant_text}]
                    + tool_result_messages
                )

        mcp_session = self.mcp_session

        async def _stream_tokens(
            ollama_messages: list[dict],
        ) -> AsyncGenerator[str, None]:
            """Stream tokens from Ollama incrementally via a queue.

            The background thread pushes each chunk into the queue as it arrives
            so the async generator can yield tokens to the caller without waiting
            for the entire response to be generated first.  A ``None`` sentinel
            signals end-of-stream; an ``Exception`` signals a streaming error.
            """
            queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()
            current_loop = asyncio.get_running_loop()

            def _run() -> None:
                try:
                    stream = self.client.chat(
                        model=self.model,
                        messages=ollama_messages,
                        stream=True,
                        options={"num_ctx": settings.server_ollama_num_ctx},
                    )
                    for chunk in stream:
                        current_loop.call_soon_threadsafe(
                            queue.put_nowait, chunk["message"]["content"]
                        )
                except Exception as exc:
                    current_loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    current_loop.call_soon_threadsafe(queue.put_nowait, None)

            self.thread_pool.submit(_run)

            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

        async def _persisting_stream() -> AsyncGenerator[str, None]:
            assembled = ""
            if resolved_answer is not None:
                assembled = resolved_answer
                yield resolved_answer
            else:
                streamed_tokens: list[str] = []
                async for token in _stream_tokens(messages):
                    streamed_tokens.append(token)

                streamed_text = "".join(streamed_tokens)

                if parse_tool_calls(streamed_text):
                    logger.warning(
                        "Session %s: model still returning tool calls after loop cap; "
                        "forcing plain-text answer.",
                        session_id_str,
                    )
                    forced_messages = messages + [
                        {"role": Role.ASSISTANT.value, "content": streamed_text},
                        {
                            "role": Role.USER.value,
                            "content": (
                                "Please answer the original question. "
                                "Do not call any more tools."
                            ),
                        },
                    ]
                    forced_response = await loop.run_in_executor(
                        self.thread_pool,
                        lambda msgs=forced_messages: self.client.chat(
                            model=self.model,
                            messages=msgs,
                            stream=False,
                            options={"num_ctx": settings.server_ollama_num_ctx},
                        ),
                    )
                    assembled = forced_response["message"]["content"]
                    yield assembled
                else:
                    assembled = streamed_text
                    for token in streamed_tokens:
                        yield token

            try:
                await mcp_session.call_tool(
                    TOOL_PERSIST_MESSAGE,
                    arguments={
                        "session_id": session_id_str,
                        "role": Role.ASSISTANT.value,
                        "content": assembled,
                    },
                )
            except Exception as exc:
                logger.error(
                    "Failed to persist assistant message (session=%s): %s",
                    session_id_str,
                    exc,
                    exc_info=True,
                )

        return _persisting_stream()
