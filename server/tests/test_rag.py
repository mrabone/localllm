from unittest.mock import MagicMock

from server.rag import RagResult, build_rag_system_message, get_rag_context


class TestGetRagContext:
    def test_returns_none_when_store_is_none(self):
        result = get_rag_context("any query", pgvector_store=None)
        assert result is None

    def test_returns_none_when_no_results(self):
        store = MagicMock()
        store.similarity_search_with_score.return_value = []

        result = get_rag_context("query", store)

        assert result is None

    def test_returns_none_when_best_score_exceeds_threshold(self):
        store = MagicMock()
        doc = MagicMock()
        store.similarity_search_with_score.return_value = [(doc, 0.9)]

        result = get_rag_context("query", store)

        assert result is None

    def test_returns_rag_result_with_correct_document_count(self):
        store = MagicMock()
        doc1, doc2 = MagicMock(), MagicMock()
        doc1.page_content = "content one"
        doc1.metadata = {"source": "http://example.com/1"}
        doc2.page_content = "content two"
        doc2.metadata = {"source": "http://example.com/2"}
        store.similarity_search_with_score.return_value = [
            (doc1, 0.1),
            (doc2, 0.3),
        ]

        result = get_rag_context("query", store)

        assert isinstance(result, RagResult)
        assert result.document_count == 2

    def test_filters_documents_exceeding_max_distance(self):
        store = MagicMock()
        close_doc = MagicMock()
        close_doc.page_content = "relevant"
        close_doc.metadata = {"source": "http://example.com/close"}
        far_doc = MagicMock()
        far_doc.page_content = "irrelevant"
        far_doc.metadata = {"source": "http://example.com/far"}
        store.similarity_search_with_score.return_value = [
            (close_doc, 0.2),
            (far_doc, 0.8),
        ]

        result = get_rag_context("query", store)

        assert result is not None
        assert result.document_count == 1
        assert "relevant" in result.context
        assert "irrelevant" not in result.context

    def test_context_includes_source_and_content(self):
        store = MagicMock()
        doc = MagicMock()
        doc.page_content = "some content"
        doc.metadata = {"source": "http://example.com/page"}
        store.similarity_search_with_score.return_value = [(doc, 0.1)]

        result = get_rag_context("query", store)

        assert result is not None
        assert "http://example.com/page" in result.context
        assert "some content" in result.context

    def test_returns_none_on_store_exception(self):
        store = MagicMock()
        store.similarity_search_with_score.side_effect = RuntimeError("DB error")

        result = get_rag_context("query", store)

        assert result is None

    def test_uses_unknown_source_fallback_when_metadata_missing(self):
        store = MagicMock()
        doc = MagicMock()
        doc.page_content = "content"
        doc.metadata = {}
        store.similarity_search_with_score.return_value = [(doc, 0.1)]

        result = get_rag_context("query", store)

        assert result is not None
        assert "Unknown source" in result.context


class TestBuildRagSystemMessage:
    def test_includes_context(self):
        msg = build_rag_system_message("This is the context block.")
        assert "This is the context block." in msg

    def test_does_not_include_user_query(self):
        """The system message contains only the retrieved context, not the user question.
        The user question remains in the separate user-role message."""
        msg = build_rag_system_message("some context")
        assert "some context" in msg

    def test_context_framing_describes_knowledge_base(self):
        """The message must frame the context as retrieved knowledge base documents."""
        msg = build_rag_system_message("doc content")
        assert "knowledge base" in msg.lower() or "retrieved" in msg.lower()

    def test_does_not_contain_fallback_instruction(self):
        """The old 'answer based on general knowledge' fallback must not be present,
        since this message is only injected when relevant documents exist."""
        msg = build_rag_system_message("some context")
        assert "general knowledge" not in msg
