from unittest.mock import MagicMock, patch

from cli.main import ChatApplication, Role


def _make_app(**kwargs) -> ChatApplication:
    """Return a ChatApplication with a mock Ollama client and no RAG store."""
    client = MagicMock()
    return ChatApplication(
        ollama_client=client,
        pgvector_store=None,
        **kwargs,
    )


class TestInit:
    def test_rag_disabled_when_store_is_none(self):
        app = _make_app()
        assert app.rag_enabled is False

    def test_rag_enabled_when_store_provided(self):
        app = ChatApplication(
            ollama_client=MagicMock(),
            pgvector_store=MagicMock(),
        )
        assert app.rag_enabled is True

    def test_message_history_starts_empty(self):
        app = _make_app()
        assert app.messages == []

    def test_custom_model_overrides_default(self):
        app = _make_app(model="my-model")
        assert app.model == "my-model"


class TestAddMessage:
    def test_appends_message_with_correct_role_and_content(self):
        app = _make_app()
        app.add_message(Role.USER, "hello")
        assert app.messages == [{"role": "user", "content": "hello"}]

    def test_appends_multiple_messages_in_order(self):
        app = _make_app()
        app.add_message(Role.USER, "hello")
        app.add_message(Role.ASSISTANT, "hi there")
        assert len(app.messages) == 2
        assert app.messages[1]["role"] == "assistant"


class TestManageConversationHistory:
    def test_returns_messages_unchanged_below_threshold(self):
        app = _make_app(threshold=10, max_recent=5)
        for i in range(5):
            app.add_message(Role.USER, f"msg {i}")

        result = app._manage_conversation_history()

        assert result == app.messages

    def test_summarises_and_trims_above_threshold(self):
        app = _make_app(threshold=4, max_recent=2)

        # Stub the summary call to return a fixed summary message
        summary_msg = {
            "role": "system",
            "content": "[Earlier conversation summary]: summary",
        }
        app._summarize_messages = MagicMock(return_value=summary_msg)

        for i in range(6):
            app.add_message(Role.USER, f"msg {i}")

        result = app._manage_conversation_history()

        # Should have [summary] + last 2 messages = 3 entries
        assert len(result) == 3
        assert result[0] == summary_msg
        assert result[1]["content"] == "msg 4"
        assert result[2]["content"] == "msg 5"

    def test_returns_recent_only_if_summarize_returns_none(self):
        app = _make_app(threshold=4, max_recent=2)
        app._summarize_messages = MagicMock(return_value=None)

        for i in range(6):
            app.add_message(Role.USER, f"msg {i}")

        result = app._manage_conversation_history()

        assert len(result) == 2
        assert result[0]["content"] == "msg 4"


class TestChat:
    def test_returns_none_rag_result_when_rag_disabled(self):
        app = _make_app()
        app._prepare_and_send = MagicMock(return_value=iter(["hello"]))

        _, rag_result = app.chat("hi")

        assert rag_result is None

    def test_passes_enriched_prompt_when_rag_returns_context(self):
        store = MagicMock()
        app = ChatApplication(
            ollama_client=MagicMock(),
            pgvector_store=store,
        )

        from cli.rag import RagResult

        mock_rag_result = RagResult(context="some context", document_count=1)

        app._prepare_and_send = MagicMock(return_value=iter(["response"]))

        with patch("cli.main.get_rag_context", return_value=mock_rag_result):
            with patch(
                "cli.main.build_rag_prompt", return_value="enriched prompt"
            ) as mock_build:
                _, rag_result = app.chat("original question")

        mock_build.assert_called_once_with("original question", "some context")
        assert rag_result is mock_rag_result
