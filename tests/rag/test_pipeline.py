import json
import tempfile
from pathlib import Path

from rag.main import extract_text_from_html, load_reading_list


class TestLoadReadingList:
    def test_returns_empty_list_for_missing_file(self):
        result = load_reading_list(Path("/non/existent/path.json"))
        assert result == []

    def test_returns_empty_list_for_invalid_json(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("not valid json {{{")
            path = Path(f.name)

        result = load_reading_list(path)
        assert result == []
        path.unlink()

    def test_returns_parsed_list_for_valid_json(self):
        data = [
            {"title": "Article One", "url": "http://example.com/1"},
            {"title": "Article Two", "url": "http://example.com/2"},
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            path = Path(f.name)

        result = load_reading_list(path)
        assert result == data
        path.unlink()

    def test_returns_empty_list_for_empty_array(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as f:
            json.dump([], f)
            path = Path(f.name)

        result = load_reading_list(path)
        assert result == []
        path.unlink()


class TestExtractTextFromHtml:
    def test_extracts_text_from_main_tag(self):
        html = "<html><body><main><p>Main content here</p></main></body></html>"
        result = extract_text_from_html(html)
        assert result == "Main content here"

    def test_falls_back_to_body_when_no_main(self):
        html = "<html><body><p>Body content</p></body></html>"
        result = extract_text_from_html(html)
        assert "Body content" in result

    def test_collapses_multiple_spaces(self):
        html = "<html><body><p>too   many    spaces</p></body></html>"
        result = extract_text_from_html(html)
        assert "too many spaces" in result

    def test_collapses_excessive_newlines(self):
        html = "<html><body><p>line one</p>\n\n\n\n<p>line two</p></body></html>"
        result = extract_text_from_html(html)
        # Should not have more than two consecutive newlines
        assert "\n\n\n" not in result

    def test_returns_none_for_empty_content(self):
        html = "<html><body></body></html>"
        result = extract_text_from_html(html)
        assert result is None

    def test_strips_leading_and_trailing_whitespace(self):
        html = "<html><body><p>   trimmed   </p></body></html>"
        result = extract_text_from_html(html)
        assert result is not None
        assert result == result.strip()
