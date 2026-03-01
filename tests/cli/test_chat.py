import json
import time
import uuid
from unittest.mock import MagicMock, patch

import httpx

from cli.main import ChatApplication
from cli.services import get_or_create_session, list_sessions


def _make_sse_response(events: list[tuple[str, str]], status_code: int = 200) -> str:
    """Build a raw SSE response body from a list of (event, data) tuples."""
    lines = []
    for event, data in events:
        lines.append(f"event: {event}")
        lines.append(f"data: {data}")
        lines.append("")
    return "\n".join(lines)


def _make_client_with_handler(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _settings_patch(server_url, registry_path, ttl=300):
    """Return a context-manager patch for cli.services.settings."""
    mock = MagicMock()
    mock.server_url = server_url
    mock.sessions_registry = str(registry_path)
    mock.session_cache_ttl = ttl
    return patch("cli.services.settings", mock)


class TestGetOrCreateSessionAnonymous:
    def test_always_creates_new_session_when_no_name(self, tmp_path, capsys):
        """Without --session, a fresh session is created every time."""
        session_id = uuid.uuid4()
        registry_file = tmp_path / "sessions.json"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path == "/sessions":
                return httpx.Response(201, json={"session_id": str(session_id)})
            return httpx.Response(404)

        client = _make_client_with_handler(handler)

        with _settings_patch("http://test", registry_file):
            result_id, result_name = get_or_create_session(client, name=None)

        assert result_id == session_id
        assert result_name == ""
        # No registry entry should be written for anonymous sessions.
        assert not registry_file.exists() or json.loads(registry_file.read_text()) == {}

    def test_prints_hint_with_no_existing_sessions(self, tmp_path, capsys):
        session_id = uuid.uuid4()
        registry_file = tmp_path / "sessions.json"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(201, json={"session_id": str(session_id)})
            return httpx.Response(404)

        client = _make_client_with_handler(handler)

        with _settings_patch("http://test", registry_file):
            get_or_create_session(client, name=None)

        captured = capsys.readouterr()
        assert "--session" in captured.out

    def test_prints_hint_listing_existing_sessions(self, tmp_path, capsys):
        session_id = uuid.uuid4()
        existing_id = uuid.uuid4()
        registry_file = tmp_path / "sessions.json"
        registry_file.write_text(json.dumps({"work": str(existing_id)}))

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(201, json={"session_id": str(session_id)})
            return httpx.Response(404)

        client = _make_client_with_handler(handler)

        with _settings_patch("http://test", registry_file):
            get_or_create_session(client, name=None)

        captured = capsys.readouterr()
        assert "work" in captured.out
        assert "--session" in captured.out


class TestGetOrCreateSessionNamed:
    def test_creates_new_named_session_when_not_in_registry(self, tmp_path):
        session_id = uuid.uuid4()
        registry_file = tmp_path / "sessions.json"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path == "/sessions":
                return httpx.Response(201, json={"session_id": str(session_id)})
            return httpx.Response(404)

        client = _make_client_with_handler(handler)

        with _settings_patch("http://test", registry_file):
            result_id, result_name = get_or_create_session(client, name="work")

        assert result_id == session_id
        assert result_name == "work"
        registry = json.loads(registry_file.read_text())
        assert registry["work"] == str(session_id)

    def test_reuses_valid_named_session_via_head(self, tmp_path):
        session_id = uuid.uuid4()
        registry_file = tmp_path / "sessions.json"
        # Pre-populate registry without a cache entry so it must HEAD-check.
        registry_file.write_text(json.dumps({"work": str(session_id)}))

        head_called = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "HEAD" and str(session_id) in request.url.path:
                head_called.append(True)
                return httpx.Response(200)
            return httpx.Response(404)

        client = _make_client_with_handler(handler)

        with _settings_patch("http://test", registry_file, ttl=300):
            result_id, result_name = get_or_create_session(client, name="work")

        assert result_id == session_id
        assert result_name == "work"
        assert head_called, "Expected a HEAD request to validate the session"

    def test_skips_server_call_within_cache_ttl(self, tmp_path):
        session_id = uuid.uuid4()
        registry_file = tmp_path / "sessions.json"
        # Write a registry with a fresh cache timestamp.
        registry_file.write_text(
            json.dumps(
                {
                    "work": str(session_id),
                    "_cache": {"work": {"validated_at": time.time()}},
                }
            )
        )

        requests_made = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests_made.append(request.method)
            return httpx.Response(404)

        client = _make_client_with_handler(handler)

        with _settings_patch("http://test", registry_file, ttl=300):
            result_id, _ = get_or_create_session(client, name="work")

        assert result_id == session_id
        assert not requests_made, "No server call should be made within TTL"

    def test_revalidates_after_cache_ttl_expires(self, tmp_path):
        session_id = uuid.uuid4()
        registry_file = tmp_path / "sessions.json"
        # Write a registry with an expired cache timestamp.
        registry_file.write_text(
            json.dumps(
                {
                    "work": str(session_id),
                    "_cache": {"work": {"validated_at": time.time() - 9999}},
                }
            )
        )

        head_called = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "HEAD":
                head_called.append(True)
                return httpx.Response(200)
            return httpx.Response(404)

        client = _make_client_with_handler(handler)

        with _settings_patch("http://test", registry_file, ttl=300):
            result_id, _ = get_or_create_session(client, name="work")

        assert result_id == session_id
        assert head_called, "Expected HEAD call after TTL expiry"

    def test_creates_replacement_when_named_session_missing_on_server(self, tmp_path):
        old_id = uuid.uuid4()
        new_id = uuid.uuid4()
        registry_file = tmp_path / "sessions.json"
        registry_file.write_text(json.dumps({"work": str(old_id)}))

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "HEAD":
                return httpx.Response(404)
            if request.method == "POST" and request.url.path == "/sessions":
                return httpx.Response(201, json={"session_id": str(new_id)})
            return httpx.Response(404)

        client = _make_client_with_handler(handler)

        with _settings_patch("http://test", registry_file):
            result_id, result_name = get_or_create_session(client, name="work")

        assert result_id == new_id
        assert result_name == "work"
        registry = json.loads(registry_file.read_text())
        assert registry["work"] == str(new_id)

    def test_multiple_named_sessions_are_independent(self, tmp_path):
        id_a = uuid.uuid4()
        id_b = uuid.uuid4()
        registry_file = tmp_path / "sessions.json"
        registry_file.write_text(
            json.dumps(
                {
                    "alpha": str(id_a),
                    "beta": str(id_b),
                    "_cache": {
                        "alpha": {"validated_at": time.time()},
                        "beta": {"validated_at": time.time()},
                    },
                }
            )
        )

        client = _make_client_with_handler(lambda r: httpx.Response(404))

        with _settings_patch("http://test", registry_file, ttl=300):
            result_a, _ = get_or_create_session(client, name="alpha")
            result_b, _ = get_or_create_session(client, name="beta")

        assert result_a == id_a
        assert result_b == id_b


class TestListSessions:
    def test_returns_empty_list_when_no_registry(self, tmp_path):
        registry_file = tmp_path / "sessions.json"
        with patch("cli.services.settings") as mock_settings:
            mock_settings.sessions_registry = str(registry_file)
            result = list_sessions()
        assert result == []

    def test_returns_sorted_session_names(self, tmp_path):
        registry_file = tmp_path / "sessions.json"
        registry_file.write_text(
            json.dumps({"zebra": str(uuid.uuid4()), "alpha": str(uuid.uuid4())})
        )
        with patch("cli.services.settings") as mock_settings:
            mock_settings.sessions_registry = str(registry_file)
            result = list_sessions()
        assert result == ["alpha", "zebra"]

    def test_excludes_cache_key_from_listing(self, tmp_path):
        registry_file = tmp_path / "sessions.json"
        registry_file.write_text(
            json.dumps(
                {
                    "work": str(uuid.uuid4()),
                    "_cache": {"work": {"validated_at": time.time()}},
                }
            )
        )
        with patch("cli.services.settings") as mock_settings:
            mock_settings.sessions_registry = str(registry_file)
            result = list_sessions()
        assert result == ["work"]


class TestChatApplication:
    def _make_app(self, handler, tmp_path, session_name=None) -> ChatApplication:
        session_id = uuid.uuid4()

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)

        app = ChatApplication.__new__(ChatApplication)
        app.client = client
        app.session_id = session_id
        app.session_name = session_name or ""
        return app

    def test_chat_prints_tokens(self, tmp_path, capsys):
        session_id = uuid.uuid4()
        sse_body = _make_sse_response(
            [
                ("token", "Hello"),
                ("token", " world"),
                ("done", "[DONE]"),
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, text=sse_body, headers={"content-type": "text/event-stream"}
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)

        app = ChatApplication.__new__(ChatApplication)
        app.client = client
        app.session_id = session_id
        app.session_name = ""

        with patch("cli.main.settings") as mock_settings:
            mock_settings.server_url = "http://test"
            app.chat("hi")

        captured = capsys.readouterr()
        assert "Hello" in captured.out
        assert " world" in captured.out

    def test_chat_shows_rag_info(self, tmp_path, capsys):
        session_id = uuid.uuid4()
        sse_body = _make_sse_response(
            [
                ("rag", "3"),
                ("token", "Answer"),
                ("done", "[DONE]"),
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, text=sse_body, headers={"content-type": "text/event-stream"}
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)

        app = ChatApplication.__new__(ChatApplication)
        app.client = client
        app.session_id = session_id
        app.session_name = ""

        with patch("cli.main.settings") as mock_settings:
            mock_settings.server_url = "http://test"
            app.chat("hi")

        captured = capsys.readouterr()
        assert "3 document(s)" in captured.out

    def test_run_displays_session_name_when_named(self, tmp_path, capsys):
        app = self._make_app(
            lambda r: httpx.Response(404), tmp_path, session_name="work"
        )

        with patch("cli.main.settings"):
            with patch("builtins.input", side_effect=["exit"]):
                app.run()

        captured = capsys.readouterr()
        assert "'work'" in captured.out

    def test_run_displays_uuid_when_anonymous(self, tmp_path, capsys):
        app = self._make_app(lambda r: httpx.Response(404), tmp_path)
        uuid_str = str(app.session_id)

        with patch("cli.main.settings"):
            with patch("builtins.input", side_effect=["exit"]):
                app.run()

        captured = capsys.readouterr()
        assert uuid_str in captured.out
