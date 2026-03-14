from unittest.mock import MagicMock, patch

import pytest

from mcp_server.rag import RagResult, build_rag_system_message, get_rag_context


def _make_doc(content: str, source: str = "test.txt"):
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source}
    return doc


class TestGetRagContext:
    def test_returns_none_when_store_is_none(self):
        result = get_rag_context("query", pgvector_store=None)
        assert result is None

    def test_returns_none_when_no_results(self):
        store = MagicMock()
        store.similarity_search_with_score.return_value = []
        result = get_rag_context("query", pgvector_store=store)
        assert result is None

    def test_returns_none_when_best_score_exceeds_threshold(self):
        store = MagicMock()
        doc = _make_doc("content")
        store.similarity_search_with_score.return_value = [(doc, 0.99)]

        with patch("mcp_server.rag.settings") as mock_settings:
            mock_settings.rag_k = 5
            mock_settings.rag_max_distance = 0.5
            result = get_rag_context("query", pgvector_store=store)

        assert result is None

    def test_filters_out_distant_documents(self):
        store = MagicMock()
        close_doc = _make_doc("relevant content", source="a.txt")
        far_doc = _make_doc("irrelevant content", source="b.txt")
        store.similarity_search_with_score.return_value = [
            (close_doc, 0.2),
            (far_doc, 0.9),
        ]

        with patch("mcp_server.rag.settings") as mock_settings:
            mock_settings.rag_k = 5
            mock_settings.rag_max_distance = 0.5
            result = get_rag_context("query", pgvector_store=store)

        assert result is not None
        assert result.document_count == 1
        assert "relevant content" in result.context
        assert "irrelevant content" not in result.context

    def test_returns_rag_result_with_correct_count(self):
        store = MagicMock()
        docs = [(_make_doc(f"doc {i}"), 0.1 * i) for i in range(1, 4)]
        store.similarity_search_with_score.return_value = docs

        with patch("mcp_server.rag.settings") as mock_settings:
            mock_settings.rag_k = 5
            mock_settings.rag_max_distance = 0.5
            result = get_rag_context("query", pgvector_store=store)

        assert result is not None
        assert result.document_count == 3

    def test_context_contains_document_source(self):
        store = MagicMock()
        doc = _make_doc("some content", source="knowledge.pdf")
        store.similarity_search_with_score.return_value = [(doc, 0.1)]

        with patch("mcp_server.rag.settings") as mock_settings:
            mock_settings.rag_k = 5
            mock_settings.rag_max_distance = 0.5
            result = get_rag_context("query", pgvector_store=store)

        assert result is not None
        assert "knowledge.pdf" in result.context

    def test_context_contains_document_content(self):
        store = MagicMock()
        doc = _make_doc("the answer is 42", source="facts.txt")
        store.similarity_search_with_score.return_value = [(doc, 0.05)]

        with patch("mcp_server.rag.settings") as mock_settings:
            mock_settings.rag_k = 5
            mock_settings.rag_max_distance = 0.5
            result = get_rag_context("query", pgvector_store=store)

        assert result is not None
        assert "the answer is 42" in result.context

    def test_returns_none_when_all_docs_filtered_out(self):
        store = MagicMock()
        docs = [(_make_doc(f"doc {i}"), 0.8 + i * 0.05) for i in range(3)]
        store.similarity_search_with_score.return_value = docs

        with patch("mcp_server.rag.settings") as mock_settings:
            mock_settings.rag_k = 5
            mock_settings.rag_max_distance = 0.5
            result = get_rag_context("query", pgvector_store=store)

        assert result is None

    def test_calls_similarity_search_with_configured_k(self):
        store = MagicMock()
        store.similarity_search_with_score.return_value = []

        with patch("mcp_server.rag.settings") as mock_settings:
            mock_settings.rag_k = 7
            mock_settings.rag_max_distance = 0.5
            get_rag_context("my query", pgvector_store=store)

        store.similarity_search_with_score.assert_called_once_with("my query", k=7)

    def test_returns_none_on_store_exception(self):
        store = MagicMock()
        store.similarity_search_with_score.side_effect = RuntimeError("db error")

        with patch("mcp_server.rag.settings") as mock_settings:
            mock_settings.rag_k = 5
            mock_settings.rag_max_distance = 0.5
            result = get_rag_context("query", pgvector_store=store)

        assert result is None

    def test_unknown_source_falls_back_to_default(self):
        store = MagicMock()
        doc = MagicMock()
        doc.page_content = "content"
        doc.metadata = {}
        store.similarity_search_with_score.return_value = [(doc, 0.1)]

        with patch("mcp_server.rag.settings") as mock_settings:
            mock_settings.rag_k = 5
            mock_settings.rag_max_distance = 0.5
            result = get_rag_context("query", pgvector_store=store)

        assert result is not None
        assert "Unknown source" in result.context


class TestBuildRagSystemMessage:
    def test_contains_context(self):
        msg = build_rag_system_message("some facts here")
        assert "some facts here" in msg

    def test_mentions_knowledge_base(self):
        msg = build_rag_system_message("ctx")
        assert "knowledge base" in msg.lower()

    def test_is_a_string(self):
        msg = build_rag_system_message("ctx")
        assert isinstance(msg, str)
