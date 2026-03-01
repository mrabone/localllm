import uuid
from unittest.mock import MagicMock, call, patch

from server.chat import ChatSession, Role


def _make_session(**kwargs) -> ChatSession:
    """Return a ChatSession with mock dependencies."""
    return ChatSession(
        session_id=uuid.uuid4(),
        engine=MagicMock(),
        ollama_client=MagicMock(),
        pgvector_store=None,
        **kwargs,
    )


class TestBuildContextWindow:
    def test_returns_messages_unchanged_below_threshold(self):
        session = _make_session(threshold=10, max_recent=5)
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(5)]

        result = session._build_context_window(messages)

        assert result == messages

    def test_summarises_and_trims_above_threshold(self):
        session = _make_session(threshold=4, max_recent=2)

        summary_msg = {
            "role": "system",
            "content": "[Earlier conversation summary]: summary",
        }
        session._summarize_messages = MagicMock(return_value=summary_msg)

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(6)]
        result = session._build_context_window(messages)

        assert len(result) == 3
        assert result[0] == summary_msg
        assert result[1]["content"] == "msg 4"
        assert result[2]["content"] == "msg 5"

    def test_returns_recent_only_if_summarize_returns_none(self):
        session = _make_session(threshold=4, max_recent=2)
        session._summarize_messages = MagicMock(return_value=None)

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(6)]
        result = session._build_context_window(messages)

        assert len(result) == 2
        assert result[0]["content"] == "msg 4"


class TestChat:
    def test_saves_user_and_assistant_messages(self):
        session = _make_session()

        with (
            patch("server.chat.load_messages", return_value=[]) as mock_load,
            patch("server.chat.save_message") as mock_save,
            patch("server.chat.get_rag_context", return_value=None),
        ):
            session.client.chat.return_value = iter([{"message": {"content": "hi"}}])
            token_stream, _ = session.chat("hello")
            # Drain the stream so the persisting wrapper runs.
            list(token_stream)

        mock_load.assert_called_once_with(session.engine, session.session_id)
        assert mock_save.call_count == 2
        first_call = mock_save.call_args_list[0]
        assert first_call == call(
            session.engine, session.session_id, Role.USER.value, "hello"
        )
        second_call = mock_save.call_args_list[1]
        assert second_call == call(
            session.engine, session.session_id, Role.ASSISTANT.value, "hi"
        )

    def test_returns_none_rag_result_when_rag_disabled(self):
        session = _make_session()

        with (
            patch("server.chat.load_messages", return_value=[]),
            patch("server.chat.save_message"),
            patch("server.chat.get_rag_context", return_value=None),
        ):
            session.client.chat.return_value = iter([{"message": {"content": ""}}])
            _, rag_result = session.chat("hi")

        assert rag_result is None

    def test_enriches_prompt_when_rag_returns_context(self):
        from server.rag import RagResult

        session = ChatSession(
            session_id=uuid.uuid4(),
            engine=MagicMock(),
            ollama_client=MagicMock(),
            pgvector_store=MagicMock(),
        )
        mock_rag_result = RagResult(context="some context", document_count=2)

        with (
            patch("server.chat.load_messages", return_value=[]),
            patch("server.chat.save_message"),
            patch("server.chat.get_rag_context", return_value=mock_rag_result),
            patch(
                "server.chat.build_rag_prompt", return_value="enriched"
            ) as mock_build,
        ):
            session.client.chat.return_value = iter(
                [{"message": {"content": "answer"}}]
            )
            token_stream, rag_result = session.chat("original question")
            list(token_stream)

        mock_build.assert_called_once_with("original question", "some context")
        assert rag_result is mock_rag_result

    def test_stream_yields_tokens(self):
        session = _make_session()

        with (
            patch("server.chat.load_messages", return_value=[]),
            patch("server.chat.save_message"),
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
