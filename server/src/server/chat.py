import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import Generator

from mem0 import Memory
from ollama import Client

from server.config import settings
from server.memory import (
    append_turn,
    load_long_term_memories,
    load_window,
    save_message,
)
from server.rag import RagResult, build_rag_system_message, get_rag_context

try:
    from langchain_postgres import PGVector
except ImportError:
    PGVector = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Shared executor for parallelising blocking I/O within chat turns.
# Using a modest cap so we don't spin up unbounded threads under concurrent
# load.  Each chat turn submits at most ~4 tasks simultaneously.
_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="chat-io")


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSession:
    """Handles a single chat turn for a persistent session.

    Context is assembled in two tiers on every turn:

    1. **Long-term memories** (layer 1) — up to ``long_term_max`` semantic
       facts extracted by Mem0 from previous conversations.  Returned as
       ``system`` role messages placed at the top of the context.

    2. **Sliding window** (layer 2) — the last ``window_size`` verbatim
       user/assistant turns, persisted in PostgreSQL so they survive server
       restarts and fresh ``ChatSession`` instantiations.

    3. **Current turn** (layer 3) — the user's input for this request,
       optionally RAG-enriched.

    Mem0 still receives every turn via ``save_message`` for long-term
    extraction; ``append_turn`` writes the verbatim text to PostgreSQL for the
    sliding window.
    """

    def __init__(
        self,
        session_id: uuid.UUID,
        mem0: Memory,
        ollama_client: Client,
        pg_dsn: str,
        pgvector_store: "PGVector | None" = None,
        model: str | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.session_id = session_id
        self.mem0 = mem0
        self.client = ollama_client
        self.pg_dsn = pg_dsn
        self.pgvector_store = pgvector_store
        self.model = model if model is not None else settings.server_ollama_model
        # Allow callers (e.g. tests) to inject a custom executor.
        # Falls back to the shared module-level pool.
        self._executor = executor if executor is not None else _executor

    def _stream_response(
        self, context_messages: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        """Stream token chunks from Ollama for the given message list."""
        stream = self.client.chat(
            model=self.model,
            messages=context_messages,
            stream=True,
            options={"num_ctx": settings.server_ollama_num_ctx},
        )
        for chunk in stream:
            yield chunk["message"]["content"]

    def chat(
        self, user_input: str
    ) -> tuple[Generator[str, None, None], RagResult | None]:
        """Process one user turn and return a (token_stream, rag_result) tuple.

        **Pre-stream phase** (parallel I/O to minimise first-token latency):

        The three independent read operations —
        ``load_long_term_memories``, ``load_window``, and ``get_rag_context`` —
        plus the user-turn ``save_message`` are all submitted to a shared
        thread-pool executor concurrently.  The results are collected before
        building the context and starting the Ollama stream, so the total
        pre-stream wait equals the *slowest* of these calls rather than
        their *sum*.

        **Post-stream phase** (fire-and-forget to eliminate end-of-stream lag):

        After the last token is yielded the assistant-turn persistence
        (``save_message`` + ``append_turn``) is submitted to the executor
        without waiting.  The SSE ``done`` event therefore reaches the client
        as soon as the final token is produced, not after Mem0 has finished
        its LLM-based extraction pass.  Errors are logged but do not affect
        the client.
        """
        f_long_term: Future = self._executor.submit(
            load_long_term_memories,
            self.mem0,
            self.session_id,
            long_term_max=settings.server_memory_long_term_max,
        )
        f_window: Future = self._executor.submit(
            load_window,
            self.pg_dsn,
            self.session_id,
            window_size=settings.server_memory_window_size,
        )
        f_rag: Future = self._executor.submit(
            get_rag_context, user_input, self.pgvector_store
        )
        # Persist the user turn to Mem0 concurrently with the reads.
        # This is the single most expensive pre-stream call (Mem0 runs an LLM
        # inference pass to extract semantic memories), so overlapping it with
        # the other reads can save a full round-trip.
        f_save_user: Future = self._executor.submit(
            save_message, self.mem0, self.session_id, Role.USER.value, user_input
        )

        # Collect results — blocks until all four futures complete.
        long_term: list[dict[str, str]] = f_long_term.result()
        window: list[dict[str, str]] = f_window.result()
        rag_result: RagResult | None = f_rag.result()
        f_save_user.result()  # surface any exception; result value unused

        # Persist user turn to the verbatim window (fast INSERT, done after
        # Mem0 save so ordering in tests is deterministic).
        append_turn(self.pg_dsn, self.session_id, Role.USER.value, user_input)

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

        rag_messages: list[dict[str, str]] = []
        if rag_result is not None:
            rag_messages = [
                {
                    "role": Role.SYSTEM.value,
                    "content": build_rag_system_message(rag_result.context),
                }
            ]

        context_messages = (
            [orientation]
            + long_term
            + window
            + rag_messages
            + [{"role": Role.USER.value, "content": user_input}]
        )

        inner_stream = self._stream_response(context_messages)

        # Capture references for the closure below.
        mem0 = self.mem0
        pg_dsn = self.pg_dsn
        session_id = self.session_id
        executor = self._executor

        def _persist_assistant(assembled: str) -> None:
            """Persist the completed assistant reply to both stores."""
            try:
                save_message(mem0, session_id, Role.ASSISTANT.value, assembled)
            except Exception as exc:
                logger.error(
                    "Background error saving assistant message to Mem0 "
                    "(session=%s): %s",
                    session_id,
                    exc,
                    exc_info=True,
                )
            try:
                append_turn(pg_dsn, session_id, Role.ASSISTANT.value, assembled)
            except Exception as exc:
                logger.error(
                    "Background error appending assistant turn to DB (session=%s): %s",
                    session_id,
                    exc,
                    exc_info=True,
                )

        def _persisting_stream() -> Generator[str, None, None]:
            assembled = ""
            for token in inner_stream:
                assembled += token
                yield token
            # Fire-and-forget: submit persistence work to the background
            # executor so the SSE 'done' event is sent immediately after the
            # final token rather than after the Mem0 LLM extraction completes.
            executor.submit(_persist_assistant, assembled)

        return _persisting_stream(), rag_result
