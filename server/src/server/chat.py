import uuid
from enum import Enum
from typing import Generator

from langchain_postgres import PGVector
from ollama import Client
from sqlalchemy.engine import Engine

from server.config import settings
from server.db import load_messages, save_message
from server.rag import RagResult, build_rag_prompt, get_rag_context


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSession:
    """Handles a single chat turn for a persistent session.

    History is loaded from Postgres at the start of each turn and the new
    user and assistant messages are written back at the end.  No in-memory
    state survives between requests.
    """

    def __init__(
        self,
        session_id: uuid.UUID,
        engine: Engine,
        ollama_client: Client,
        pgvector_store: PGVector | None,
        model: str | None = None,
        max_recent: int | None = None,
        threshold: int | None = None,
    ) -> None:
        self.session_id = session_id
        self.engine = engine
        self.client = ollama_client
        self.pgvector_store = pgvector_store
        self.model = model if model is not None else settings.server_ollama_model
        self.max_recent = (
            max_recent if max_recent is not None else settings.server_max_recent
        )
        self.threshold = (
            threshold if threshold is not None else settings.server_threshold
        )

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _summarize_messages(
        self, messages_to_summarize: list[dict[str, str]]
    ) -> dict[str, str] | None:
        """Summarize a list of messages into a single system message.

        Returns None if the input is empty or the model returns no content.
        """
        if not messages_to_summarize:
            return None

        conversation_text = "\n".join(
            f"{msg['role'].title()}: {msg['content']}" for msg in messages_to_summarize
        )
        summary_prompt = (
            "Summarize the following conversation concisely in 2-3 sentences, "
            "preserving key points and context:\n\n"
            f"{conversation_text}\n\n"
            "Summary:"
        )

        response = self.client.chat(
            model=self.model,
            messages=[{"role": Role.USER.value, "content": summary_prompt}],
            stream=False,
        )

        content = response.get("message", {}).get("content", "").strip()
        if not content:
            return None

        return {
            "role": Role.SYSTEM.value,
            "content": f"[Earlier conversation summary]: {content}",
        }

    def _build_context_window(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Return a context window, summarising old messages when the threshold is reached."""
        if len(messages) <= self.threshold:
            return messages

        recent_messages = messages[-self.max_recent :]
        old_messages = messages[: -self.max_recent]

        if old_messages:
            summary = self._summarize_messages(old_messages)
            if summary:
                return [summary] + recent_messages

        return recent_messages

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _stream_response(
        self, context_messages: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        """Stream token chunks from Ollama for the given message list."""
        stream = self.client.chat(
            model=self.model, messages=context_messages, stream=True
        )
        # The Ollama iterator uses an httpx streaming context internally.
        # PEP 479 converts any StopIteration raised inside a generator into a
        # RuntimeError, so we catch it explicitly to let the generator exit
        # cleanly instead of propagating an error through run_in_executor.
        try:
            for chunk in stream:
                yield chunk["message"]["content"]
        except StopIteration:
            return

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def chat(
        self, user_input: str
    ) -> tuple[Generator[str, None, None], RagResult | None]:
        """Process one user turn and return a (token_stream, rag_result) tuple.

        Loads history from Postgres, optionally enriches the prompt via RAG,
        streams the response from Ollama, and persists both the user message
        and the assembled assistant reply.

        The caller must fully consume the generator before the assistant
        message is written to the database (writes happen lazily as a wrapper
        around the inner stream).
        """
        history = load_messages(self.engine, self.session_id)

        rag_result = get_rag_context(user_input, self.pgvector_store)
        effective_input = (
            build_rag_prompt(user_input, rag_result.context)
            if rag_result is not None
            else user_input
        )

        save_message(self.engine, self.session_id, Role.USER.value, user_input)

        augmented_history = history + [
            {"role": Role.USER.value, "content": effective_input}
        ]
        context_messages = self._build_context_window(augmented_history)

        inner_stream = self._stream_response(context_messages)

        def _persisting_stream() -> Generator[str, None, None]:
            assembled = ""
            for token in inner_stream:
                assembled += token
                yield token
            save_message(self.engine, self.session_id, Role.ASSISTANT.value, assembled)

        return _persisting_stream(), rag_result
