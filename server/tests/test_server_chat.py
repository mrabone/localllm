import uuid
from concurrent.futures import Future
from unittest.mock import MagicMock, call, patch

from server.chat import ChatSession, Role

_FAKE_DSN = "host=localhost dbname=test"


class _ImmediateExecutor:
    """Minimal executor stub that runs submitted callables synchronously.

    Used in tests that need to assert on the results of fire-and-forget
    background tasks (e.g. assistant-turn Mem0 persistence) without
    introducing timing non-determinism.
    """

    def submit(self, fn, *args, **kwargs) -> Future:
        f: Future = Future()
        try:
            result = fn(*args, **kwargs)
            f.set_result(result)
        except Exception as exc:
            f.set_exception(exc)
        return f


def _make_session(**kwargs) -> ChatSession:
    """Return a ChatSession with mock dependencies.

    Injects an ``_ImmediateExecutor`` by default so that background tasks
    (e.g. fire-and-forget Mem0 saves) complete synchronously within tests.
    """
    kwargs.setdefault("executor", _ImmediateExecutor())
    return ChatSession(
        session_id=uuid.uuid4(),
        mem0=MagicMock(),
        ollama_client=MagicMock(),
        pg_dsn=_FAKE_DSN,
        pgvector_store=None,
        **kwargs,
    )


class TestChat:
    def test_saves_user_and_assistant_messages_to_mem0_and_window(self):
        """Both save_message (Mem0) and append_turn (window) are called for each turn."""
        session = _make_session()

        with (
            patch("server.chat.load_long_term_memories", return_value=[]),
            patch("server.chat.load_window", return_value=[]),
            patch("server.chat.save_message") as mock_save,
            patch("server.chat.append_turn") as mock_append,
            patch("server.chat.get_rag_context", return_value=None),
        ):
            session.client.chat.return_value = iter([{"message": {"content": "hi"}}])
            token_stream, _ = session.chat("hello")
            list(token_stream)

        assert mock_save.call_count == 2
        assert mock_save.call_args_list[0] == call(
            session.mem0, session.session_id, Role.USER.value, "hello"
        )
        assert mock_save.call_args_list[1] == call(
            session.mem0, session.session_id, Role.ASSISTANT.value, "hi"
        )

        assert mock_append.call_count == 2
        assert mock_append.call_args_list[0] == call(
            _FAKE_DSN, session.session_id, Role.USER.value, "hello"
        )
        assert mock_append.call_args_list[1] == call(
            _FAKE_DSN, session.session_id, Role.ASSISTANT.value, "hi"
        )

    def test_returns_none_rag_result_when_rag_disabled(self):
        session = _make_session()

        with (
            patch("server.chat.load_long_term_memories", return_value=[]),
            patch("server.chat.load_window", return_value=[]),
            patch("server.chat.save_message"),
            patch("server.chat.append_turn"),
            patch("server.chat.get_rag_context", return_value=None),
        ):
            session.client.chat.return_value = iter([{"message": {"content": ""}}])
            _, rag_result = session.chat("hi")

        assert rag_result is None

    def test_enriches_prompt_when_rag_returns_context(self):
        """RAG context is injected as a dedicated system message, not into the user turn."""
        from server.rag import RagResult

        session = ChatSession(
            session_id=uuid.uuid4(),
            mem0=MagicMock(),
            ollama_client=MagicMock(),
            pg_dsn=_FAKE_DSN,
            pgvector_store=MagicMock(),
        )
        mock_rag_result = RagResult(context="some context", document_count=2)

        with (
            patch("server.chat.load_long_term_memories", return_value=[]),
            patch("server.chat.load_window", return_value=[]),
            patch("server.chat.save_message"),
            patch("server.chat.append_turn"),
            patch("server.chat.get_rag_context", return_value=mock_rag_result),
            patch(
                "server.chat.build_rag_system_message", return_value="rag system msg"
            ) as mock_build,
        ):
            session.client.chat.return_value = iter(
                [{"message": {"content": "answer"}}]
            )
            token_stream, rag_result = session.chat("original question")
            list(token_stream)

        mock_build.assert_called_once_with("some context")
        assert rag_result is mock_rag_result

        call_kwargs = session.client.chat.call_args[1]
        messages = call_kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert roles.count("system") >= 2
        rag_msg = next(m for m in messages if m["content"] == "rag system msg")
        assert rag_msg["role"] == "system"
        assert messages[-1] == {"role": "user", "content": "original question"}

    def test_stream_yields_tokens(self):
        session = _make_session()

        with (
            patch("server.chat.load_long_term_memories", return_value=[]),
            patch("server.chat.load_window", return_value=[]),
            patch("server.chat.save_message"),
            patch("server.chat.append_turn"),
            patch("server.chat.get_rag_context", return_value=None),
        ):
            session.client.chat.return_value = iter(
                [
                    {"message": {"content": "Hello"}},
                    {"message": {"content": " world"}},
                ]
            )
            token_stream, _ = session.chat("hi")
            tokens = list(token_stream)

        assert tokens == ["Hello", " world"]

    def test_two_tier_context_order(self):
        """Context passed to Ollama must be:
        orientation + long_term + window + [current_turn].
        RAG is absent here (no pgvector store); orientation is always first.
        """
        session = _make_session()

        long_term = [
            {
                "role": "system",
                "content": (
                    "The following facts have been remembered from previous "
                    "conversations with this user:\n- User is a Python developer"
                ),
            }
        ]
        window = [
            {"role": "user", "content": "prev question"},
            {"role": "assistant", "content": "prev answer"},
        ]

        with (
            patch("server.chat.load_long_term_memories", return_value=long_term),
            patch("server.chat.load_window", return_value=window),
            patch("server.chat.save_message"),
            patch("server.chat.append_turn"),
            patch("server.chat.get_rag_context", return_value=None),
        ):
            session.client.chat.return_value = iter([{"message": {"content": "ok"}}])
            token_stream, _ = session.chat("current question")
            list(token_stream)

        call_kwargs = session.client.chat.call_args[1]
        messages = call_kwargs["messages"]

        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]

        assert messages[1] == long_term[0]

        assert messages[2] == {"role": "user", "content": "prev question"}
        assert messages[3] == {"role": "assistant", "content": "prev answer"}

        assert messages[4] == {"role": "user", "content": "current question"}

        assert len(messages) == 5

    def test_long_term_memories_fetched_with_configured_max(self):
        """load_long_term_memories is called with server_memory_long_term_max."""
        session = _make_session()

        with (
            patch("server.chat.load_long_term_memories", return_value=[]) as mock_lt,
            patch("server.chat.load_window", return_value=[]),
            patch("server.chat.save_message"),
            patch("server.chat.append_turn"),
            patch("server.chat.get_rag_context", return_value=None),
            patch("server.chat.settings") as mock_settings,
        ):
            mock_settings.server_ollama_model = "test-model"
            mock_settings.server_ollama_num_ctx = 8192
            mock_settings.server_memory_long_term_max = 3
            mock_settings.server_memory_window_size = 10
            session.client.chat.return_value = iter([{"message": {"content": "ok"}}])
            token_stream, _ = session.chat("hello")
            list(token_stream)

        mock_lt.assert_called_once_with(
            session.mem0, session.session_id, long_term_max=3
        )

    def test_window_fetched_with_configured_size(self):
        """load_window is called with server_memory_window_size."""
        session = _make_session()

        with (
            patch("server.chat.load_long_term_memories", return_value=[]),
            patch("server.chat.load_window", return_value=[]) as mock_win,
            patch("server.chat.save_message"),
            patch("server.chat.append_turn"),
            patch("server.chat.get_rag_context", return_value=None),
            patch("server.chat.settings") as mock_settings,
        ):
            mock_settings.server_ollama_model = "test-model"
            mock_settings.server_ollama_num_ctx = 8192
            mock_settings.server_memory_long_term_max = 3
            mock_settings.server_memory_window_size = 10
            session.client.chat.return_value = iter([{"message": {"content": "ok"}}])
            token_stream, _ = session.chat("hello")
            list(token_stream)

        mock_win.assert_called_once_with(_FAKE_DSN, session.session_id, window_size=10)

    def test_num_ctx_is_passed_to_ollama(self):
        """client.chat() must receive num_ctx in its options."""
        session = _make_session()

        with (
            patch("server.chat.load_long_term_memories", return_value=[]),
            patch("server.chat.load_window", return_value=[]),
            patch("server.chat.save_message"),
            patch("server.chat.append_turn"),
            patch("server.chat.get_rag_context", return_value=None),
            patch("server.chat.settings") as mock_settings,
        ):
            mock_settings.server_ollama_model = "test-model"
            mock_settings.server_ollama_num_ctx = 8192
            mock_settings.server_memory_long_term_max = 3
            mock_settings.server_memory_window_size = 10
            session.client.chat.return_value = iter([{"message": {"content": "ok"}}])
            token_stream, _ = session.chat("hello")
            list(token_stream)

        call_kwargs = session.client.chat.call_args[1]
        assert call_kwargs.get("options", {}).get("num_ctx") == 8192

    def test_rag_injected_as_system_message_before_user_turn(self):
        """When RAG returns results the context must appear as a system message
        immediately before the final user turn, and the user turn must contain
        the clean question only."""
        from server.rag import RagResult

        session = _make_session()
        mock_rag_result = RagResult(context="doc content", document_count=1)

        with (
            patch("server.chat.load_long_term_memories", return_value=[]),
            patch("server.chat.load_window", return_value=[]),
            patch("server.chat.save_message"),
            patch("server.chat.append_turn"),
            patch("server.chat.get_rag_context", return_value=mock_rag_result),
            patch(
                "server.chat.build_rag_system_message",
                return_value="kb: doc content",
            ),
        ):
            session.client.chat.return_value = iter([{"message": {"content": "ok"}}])
            token_stream, _ = session.chat("my question")
            list(token_stream)

        call_kwargs = session.client.chat.call_args[1]
        messages = call_kwargs["messages"]

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "system", "content": "kb: doc content"}
        assert messages[2] == {"role": "user", "content": "my question"}

    def test_no_rag_message_when_rag_disabled(self):
        """When RAG is disabled (no pgvector store) no RAG system message is present."""
        session = _make_session()

        with (
            patch("server.chat.load_long_term_memories", return_value=[]),
            patch("server.chat.load_window", return_value=[]),
            patch("server.chat.save_message"),
            patch("server.chat.append_turn"),
            patch("server.chat.get_rag_context", return_value=None),
        ):
            session.client.chat.return_value = iter([{"message": {"content": "ok"}}])
            token_stream, _ = session.chat("my question")
            list(token_stream)

        call_kwargs = session.client.chat.call_args[1]
        messages = call_kwargs["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "my question"}

    def test_orientation_message_is_always_first(self):
        """The orientation system message must always be the first message,
        even when there are no long-term memories and no RAG results."""
        session = _make_session()

        with (
            patch("server.chat.load_long_term_memories", return_value=[]),
            patch("server.chat.load_window", return_value=[]),
            patch("server.chat.save_message"),
            patch("server.chat.append_turn"),
            patch("server.chat.get_rag_context", return_value=None),
        ):
            session.client.chat.return_value = iter([{"message": {"content": "ok"}}])
            token_stream, _ = session.chat("hello")
            list(token_stream)

        call_kwargs = session.client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]
