import logging
import uuid
from enum import Enum
from typing import AsyncGenerator

from mcp import ClientSession
from ollama import Client

from server.config import settings

logger = logging.getLogger(__name__)


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSession:
    """Handles a single chat turn for a persistent session.

    Context is assembled in two tiers on every turn:

    1. **Long-term memories** (layer 1) — semantic facts extracted by Mem0
       from previous conversations, retrieved via the MCP ``load_long_term_memory``
       tool and injected as a ``system`` message.

    2. **Sliding window** (layer 2) — the last ``window_size`` verbatim
       user/assistant turns, retrieved via the MCP ``load_conversation_window``
       tool.

    3. **Current turn** (layer 3) — the user's input, optionally augmented
       with RAG context retrieved via the MCP ``search_knowledge_base`` tool.

    All MCP tool calls for context retrieval are issued concurrently using
    ``asyncio.gather`` to minimise first-token latency.  Message persistence
    is fire-and-forget after the stream completes.
    """

    def __init__(
        self,
        session_id: uuid.UUID,
        mcp_session: ClientSession,
        ollama_client: Client,
        model: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.mcp_session = mcp_session
        self.client = ollama_client
        self.model = model if model is not None else settings.server_ollama_model

    async def _call_tool(self, name: str, arguments: dict) -> str:
        """Call a named MCP tool and return its first text content."""
        result = await self.mcp_session.call_tool(name, arguments=arguments)
        if result.content:
            return result.content[0].text
        return ""

    async def chat(self, user_input: str) -> tuple[AsyncGenerator[str, None], bool]:
        """Process one user turn and return a (token_stream, rag_used) tuple.

        Fires three MCP tool calls concurrently to minimise pre-stream latency:
        ``load_long_term_memory``, ``load_conversation_window``, and
        ``search_knowledge_base``.  The user-turn persistence call is also
        issued concurrently with the reads.

        Returns an async generator that yields token strings, plus a boolean
        indicating whether RAG context was included.
        """
        import asyncio

        session_id_str = str(self.session_id)

        long_term_task = asyncio.create_task(
            self._call_tool(
                "load_long_term_memory",
                {
                    "session_id": session_id_str,
                    "long_term_max": settings.server_memory_long_term_max,
                },
            )
        )
        window_task = asyncio.create_task(
            self._call_tool(
                "load_conversation_window",
                {
                    "session_id": session_id_str,
                    "window_size": settings.server_memory_window_size,
                },
            )
        )
        rag_task = asyncio.create_task(
            self._call_tool("search_knowledge_base", {"query": user_input})
        )
        persist_user_task = asyncio.create_task(
            self._call_tool(
                "persist_message",
                {
                    "session_id": session_id_str,
                    "role": Role.USER.value,
                    "content": user_input,
                },
            )
        )

        long_term_content, window_content, rag_content, _ = await asyncio.gather(
            long_term_task, window_task, rag_task, persist_user_task
        )

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

        context_messages: list[dict] = [orientation]

        if long_term_content:
            context_messages.append(
                {"role": Role.SYSTEM.value, "content": long_term_content}
            )

        # window_content is a JSON-encoded list of {role, content} dicts from
        # the MCP tool.  Parse it back so we can extend context_messages.
        if window_content:
            import json

            try:
                window_turns = json.loads(window_content)
                context_messages.extend(window_turns)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse window content from MCP tool.")

        rag_used = bool(rag_content)
        if rag_used:
            context_messages.append({"role": Role.SYSTEM.value, "content": rag_content})

        context_messages.append({"role": Role.USER.value, "content": user_input})

        mcp_session = self.mcp_session
        session_id_str_ = session_id_str

        async def _persisting_stream() -> AsyncGenerator[str, None]:
            assembled = ""
            stream = self.client.chat(
                model=self.model,
                messages=context_messages,
                stream=True,
                options={"num_ctx": settings.server_ollama_num_ctx},
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                assembled += token
                yield token

            try:
                await mcp_session.call_tool(
                    "persist_message",
                    arguments={
                        "session_id": session_id_str_,
                        "role": Role.ASSISTANT.value,
                        "content": assembled,
                    },
                )
            except Exception as exc:
                logger.error(
                    "Failed to persist assistant message (session=%s): %s",
                    session_id_str_,
                    exc,
                    exc_info=True,
                )

        return _persisting_stream(), rag_used
