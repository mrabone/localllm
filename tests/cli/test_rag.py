from unittest.mock import MagicMock

from server.rag import RagResult, build_rag_prompt, get_rag_context


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
        # Score of 0.9 is above the default max_distance of 0.5
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
            (far_doc, 0.8),  # above default threshold of 0.5
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
        doc.metadata = {}  # no "source" key
        store.similarity_search_with_score.return_value = [(doc, 0.1)]

        result = get_rag_context("query", store)

        assert result is not None
        assert "Unknown source" in result.context


class TestBuildRagPrompt:
    def test_includes_user_query(self):
        prompt = build_rag_prompt("What is RAG?", "Some context.")
        assert "What is RAG?" in prompt

    def test_includes_context(self):
        prompt = build_rag_prompt("query", "This is the context block.")
        assert "This is the context block." in prompt

    def test_query_appears_after_context(self):
        prompt = build_rag_prompt("my question", "my context")
        assert prompt.index("my context") < prompt.index("my question")
