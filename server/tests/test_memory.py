import uuid
from unittest.mock import MagicMock, call, patch

import pytest

from server.memory import (
    append_turn,
    create_session,
    ensure_turns_table,
    load_long_term_memories,
    load_messages,
    load_window,
    save_message,
    session_exists,
)


def _make_mock_conn():
    """Return a (mock_conn, mock_cur) pair wired up as context managers."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur
    return mock_conn, mock_cur


class TestCreateSession:
    def test_returns_uuid(self):
        """create_session should return a UUID object."""
        mock_conn, _ = _make_mock_conn()
        with patch("server.memory._get_conn", return_value=mock_conn):
            session_id = create_session("dsn")
        assert isinstance(session_id, uuid.UUID)

    def test_each_call_returns_unique_uuid(self):
        """create_session should return a different UUID each time."""
        mock_conn, _ = _make_mock_conn()
        with patch("server.memory._get_conn", return_value=mock_conn):
            session_id_1 = create_session("dsn")
            session_id_2 = create_session("dsn")
        assert session_id_1 != session_id_2

    def test_returned_uuid_is_valid(self):
        """The returned UUID should be valid and convertible to string."""
        mock_conn, _ = _make_mock_conn()
        with patch("server.memory._get_conn", return_value=mock_conn):
            session_id = create_session("dsn")
        str_id = str(session_id)
        assert len(str_id) == 36
        assert str_id.count("-") == 4

    def test_inserts_row_into_chat_sessions(self):
        """create_session should INSERT the new UUID into chat_sessions."""
        mock_conn, mock_cur = _make_mock_conn()
        with patch("server.memory._get_conn", return_value=mock_conn):
            session_id = create_session("dsn")
        mock_cur.execute.assert_called_once()
        sql, params = mock_cur.execute.call_args[0]
        assert "INSERT INTO chat_sessions" in sql
        assert str(session_id) in params


class TestSessionExists:
    def test_returns_true_when_row_exists(self):
        """session_exists should return True when chat_sessions has a matching row."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = (1,)

        session_id = uuid.uuid4()
        with patch("server.memory._get_conn", return_value=mock_conn):
            result = session_exists("dsn", session_id)

        assert result is True

    def test_returns_false_when_no_row(self):
        """session_exists should return False when chat_sessions has no matching row."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = None

        session_id = uuid.uuid4()
        with patch("server.memory._get_conn", return_value=mock_conn):
            result = session_exists("dsn", session_id)

        assert result is False

    def test_returns_false_when_get_all_returns_dict_with_empty_results(self):
        """session_exists should return False when no row is found (fetchone returns None)."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = None

        session_id = uuid.uuid4()
        with patch("server.memory._get_conn", return_value=mock_conn):
            result = session_exists("dsn", session_id)

        assert result is False

    def test_returns_true_when_get_all_returns_dict_with_populated_results(self):
        """session_exists should return True when a row is found (fetchone returns a row)."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = (1,)

        session_id = uuid.uuid4()
        with patch("server.memory._get_conn", return_value=mock_conn):
            result = session_exists("dsn", session_id)

        assert result is True

    def test_handles_exception_gracefully(self):
        """session_exists should return False if the DB raises an exception."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.side_effect = Exception("Connection error")

        session_id = uuid.uuid4()
        with patch("server.memory._get_conn", return_value=mock_conn):
            result = session_exists("dsn", session_id)

        assert result is False

    def test_converts_session_id_to_string(self):
        """session_exists should pass the UUID as a string to the SQL query."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = None

        session_id = uuid.uuid4()
        with patch("server.memory._get_conn", return_value=mock_conn):
            session_exists("dsn", session_id)

        _, params = mock_cur.execute.call_args[0]
        assert str(session_id) in params


class TestSaveMessage:
    def test_calls_mem0_add_with_correct_parameters(self):
        """save_message should call mem0.add with the message in correct format."""
        mem0 = MagicMock()
        session_id = uuid.uuid4()
        role = "user"
        content = "Hello, world!"

        save_message(mem0, session_id, role, content)

        mem0.add.assert_called_once_with(
            [{"role": role, "content": content}],
            user_id=str(session_id),
        )

    def test_saves_user_messages(self):
        """save_message should correctly save user messages."""
        mem0 = MagicMock()
        session_id = uuid.uuid4()

        save_message(mem0, session_id, "user", "What is Paris?")

        mem0.add.assert_called_once()
        call_args = mem0.add.call_args
        assert call_args[0][0][0]["role"] == "user"
        assert call_args[0][0][0]["content"] == "What is Paris?"

    def test_saves_assistant_messages(self):
        """save_message should correctly save assistant messages."""
        mem0 = MagicMock()
        session_id = uuid.uuid4()

        save_message(mem0, session_id, "assistant", "Paris is the capital of France.")

        mem0.add.assert_called_once()
        call_args = mem0.add.call_args
        assert call_args[0][0][0]["role"] == "assistant"
        assert call_args[0][0][0]["content"] == "Paris is the capital of France."

    def test_converts_session_id_to_string(self):
        """save_message should convert UUID to string when calling mem0.add."""
        mem0 = MagicMock()
        session_id = uuid.uuid4()

        save_message(mem0, session_id, "user", "test")

        call_args = mem0.add.call_args
        assert call_args[1]["user_id"] == str(session_id)

    def test_raises_exception_on_mem0_failure(self):
        """save_message should propagate exceptions from mem0.add."""
        mem0 = MagicMock()
        mem0.add.side_effect = Exception("Storage error")
        session_id = uuid.uuid4()

        with pytest.raises(Exception, match="Storage error"):
            save_message(mem0, session_id, "user", "test")

    def test_handles_long_content(self):
        """save_message should handle long message content."""
        mem0 = MagicMock()
        session_id = uuid.uuid4()
        long_content = "x" * 10000

        save_message(mem0, session_id, "user", long_content)

        call_args = mem0.add.call_args
        assert call_args[0][0][0]["content"] == long_content

    def test_handles_special_characters(self):
        """save_message should handle special characters in content."""
        mem0 = MagicMock()
        session_id = uuid.uuid4()
        content = "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?\n\t"

        save_message(mem0, session_id, "user", content)

        call_args = mem0.add.call_args
        assert call_args[0][0][0]["content"] == content


class TestLoadMessages:
    def test_returns_empty_list_for_empty_results(self):
        """load_messages should return empty list when no memories exist."""
        mem0 = MagicMock()
        mem0.get_all.return_value = []

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="")

        assert messages == []

    def test_calls_get_all_when_query_is_empty(self):
        """load_messages should call get_all when query is empty string."""
        mem0 = MagicMock()
        mem0.get_all.return_value = []

        session_id = uuid.uuid4()
        load_messages(mem0, session_id, query="")

        mem0.get_all.assert_called_once_with(user_id=str(session_id))
        mem0.search.assert_not_called()

    def test_calls_search_when_query_is_provided(self):
        """load_messages should call search when query is non-empty."""
        mem0 = MagicMock()
        mem0.search.return_value = []

        session_id = uuid.uuid4()
        load_messages(mem0, session_id, query="What is France?")

        mem0.search.assert_called_once_with("What is France?", user_id=str(session_id))
        mem0.get_all.assert_not_called()

    def test_extracts_memory_field(self):
        """load_messages should extract and return memory field from results."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"memory": "Paris is the capital of France"},
            {"memory": "France is in Western Europe"},
        ]

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Paris is the capital of France"
        assert messages[1]["role"] == "system"
        assert messages[1]["content"] == "France is in Western Europe"

    def test_extracts_content_field_as_fallback(self):
        """load_messages should extract content field if memory field is missing."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"content": "Test message", "role": "user"},
        ]

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="")

        assert len(messages) == 1
        assert messages[0]["content"] == "Test message"

    def test_preserves_role_from_result(self):
        """load_messages should preserve role field from result if present."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"content": "User said this", "role": "user"},
            {"content": "Assistant replied", "role": "assistant"},
        ]

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="")

        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_defaults_to_system_role_when_missing(self):
        """load_messages should default to system role if role field is absent."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"memory": "A fact about Paris"},
        ]

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="")

        assert messages[0]["role"] == "system"

    def test_handles_dict_with_results_key(self):
        """load_messages should extract results from dict if it has 'results' key."""
        mem0 = MagicMock()
        mem0.get_all.return_value = {
            "results": [{"memory": "Test fact"}],
        }

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="")

        assert len(messages) == 1
        assert messages[0]["content"] == "Test fact"

    def test_handles_dict_with_data_key(self):
        """load_messages should extract data from dict if it has 'data' key."""
        mem0 = MagicMock()
        mem0.get_all.return_value = {
            "data": [{"memory": "Test fact"}],
        }

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="")

        assert len(messages) == 1
        assert messages[0]["content"] == "Test fact"

    def test_filters_out_empty_memories(self):
        """load_messages should skip items without content."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"memory": "Valid memory"},
            {"other_field": "value"},
            {"memory": "Another valid memory"},
        ]

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="")

        assert len(messages) == 2
        assert messages[0]["content"] == "Valid memory"
        assert messages[1]["content"] == "Another valid memory"

    def test_handles_non_dict_items_gracefully(self):
        """load_messages should skip non-dict items in results."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"memory": "Valid memory"},
            "invalid string",
            {"memory": "Another valid memory"},
        ]

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="")

        assert len(messages) == 2

    def test_returns_empty_list_on_exception(self):
        """load_messages should return empty list if an exception occurs."""
        mem0 = MagicMock()
        mem0.get_all.side_effect = Exception("Connection error")

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="")

        assert messages == []

    def test_search_with_custom_query(self):
        """load_messages should pass the query correctly to mem0.search."""
        mem0 = MagicMock()
        mem0.search.return_value = [
            {"memory": "Relevant memory about Paris"},
        ]

        session_id = uuid.uuid4()
        query = "Tell me about Paris"
        messages = load_messages(mem0, session_id, query=query)

        mem0.search.assert_called_once_with(query, user_id=str(session_id))
        assert len(messages) == 1
        assert messages[0]["content"] == "Relevant memory about Paris"

    def test_converts_session_id_to_string(self):
        """load_messages should convert session UUID to string."""
        mem0 = MagicMock()
        mem0.get_all.return_value = []

        session_id = uuid.uuid4()
        load_messages(mem0, session_id, query="")

        call_args = mem0.get_all.call_args
        assert call_args[1]["user_id"] == str(session_id)

    def test_handles_search_returning_dict(self):
        """load_messages should handle search returning a dict with results."""
        mem0 = MagicMock()
        mem0.search.return_value = {
            "results": [{"memory": "Search result"}],
        }

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="test")

        assert len(messages) == 1
        assert messages[0]["content"] == "Search result"

    def test_tries_multiple_content_field_names(self):
        """load_messages should try 'memory', 'content', 'data', 'text' in order."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"data": "From data field"},
            {"text": "From text field"},
        ]

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="")

        assert len(messages) == 2
        assert messages[0]["content"] == "From data field"
        assert messages[1]["content"] == "From text field"

    def test_memory_field_takes_priority(self):
        """load_messages should prefer 'memory' field over others."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {
                "memory": "From memory",
                "content": "From content",
                "data": "From data",
            },
        ]

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="")

        assert messages[0]["content"] == "From memory"

    def test_respects_max_messages_cap(self):
        """load_messages should return at most max_messages entries."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [{"memory": f"fact {i}"} for i in range(20)]

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="", max_messages=5)

        assert len(messages) == 5

    def test_cap_keeps_first_entries(self):
        """load_messages should keep the first (most-relevant) entries when capping."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [{"memory": f"fact {i}"} for i in range(20)]

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="", max_messages=3)

        assert messages[0]["content"] == "fact 0"
        assert messages[1]["content"] == "fact 1"
        assert messages[2]["content"] == "fact 2"

    def test_does_not_cap_when_results_within_limit(self):
        """load_messages should return all entries when count is below the cap."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"memory": "fact a"},
            {"memory": "fact b"},
        ]

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="", max_messages=10)

        assert len(messages) == 2

    def test_default_max_messages_is_ten(self):
        """The default cap should be 10 so callers do not need to pass max_messages."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [{"memory": f"fact {i}"} for i in range(15)]

        session_id = uuid.uuid4()
        messages = load_messages(mem0, session_id, query="")

        assert len(messages) == 10


class TestEnsureTurnsTable:
    def test_executes_create_table_and_index(self):
        """ensure_turns_table should execute CREATE TABLE (sessions), CREATE TABLE (turns) and CREATE INDEX statements."""
        mock_conn, mock_cur = _make_mock_conn()

        with patch("server.memory._get_conn", return_value=mock_conn):
            ensure_turns_table("host=localhost dbname=test")

        assert mock_cur.execute.call_count == 3

    def test_raises_on_db_error(self):
        """ensure_turns_table should propagate database errors."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.side_effect = Exception("DB error")

        with patch("server.memory._get_conn", return_value=mock_conn):
            with pytest.raises(Exception, match="DB error"):
                ensure_turns_table("host=localhost dbname=test")


class TestAppendTurn:
    def test_inserts_row_with_correct_values(self):
        """append_turn should INSERT a row with session_id, role, and content."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur

        session_id = uuid.uuid4()
        with patch("server.memory._get_conn", return_value=mock_conn):
            append_turn("dsn", session_id, "user", "hello")

        mock_cur.execute.assert_called_once()
        call_args = mock_cur.execute.call_args
        sql, params = call_args[0]
        assert "INSERT INTO conversation_turns" in sql
        assert params == (str(session_id), "user", "hello")

    def test_raises_on_db_error(self):
        """append_turn should propagate database errors."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.side_effect = Exception("write error")

        with patch("server.memory._get_conn", return_value=mock_conn):
            with pytest.raises(Exception, match="write error"):
                append_turn("dsn", uuid.uuid4(), "user", "hello")

    def test_converts_session_id_to_string(self):
        """append_turn should pass the session UUID as a string to the DB."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur

        session_id = uuid.uuid4()
        with patch("server.memory._get_conn", return_value=mock_conn):
            append_turn("dsn", session_id, "assistant", "reply")

        _, params = mock_cur.execute.call_args[0]
        assert params[0] == str(session_id)
        assert isinstance(params[0], str)


class TestLoadWindow:
    def _make_mock_conn(self, rows: list[dict]) -> MagicMock:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = rows
        mock_conn.cursor.return_value = mock_cur
        return mock_conn

    def test_returns_rows_as_role_content_dicts(self):
        """load_window should map DB rows to {role, content} dicts."""
        rows = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        mock_conn = self._make_mock_conn(rows)
        session_id = uuid.uuid4()

        with patch("server.memory._get_conn", return_value=mock_conn):
            messages = load_window("dsn", session_id, window_size=10)

        assert messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

    def test_returns_empty_list_when_no_rows(self):
        """load_window should return an empty list when there are no turns."""
        mock_conn = self._make_mock_conn([])
        session_id = uuid.uuid4()

        with patch("server.memory._get_conn", return_value=mock_conn):
            messages = load_window("dsn", session_id, window_size=10)

        assert messages == []

    def test_passes_window_size_and_session_id_to_query(self):
        """load_window should pass window_size (LIMIT) and session_id to the query."""
        mock_conn = self._make_mock_conn([])
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value
        session_id = uuid.uuid4()

        with patch("server.memory._get_conn", return_value=mock_conn):
            load_window("dsn", session_id, window_size=5)

        call_args = mock_cur.execute.call_args
        _, params = call_args[0]
        assert params == (str(session_id), 5)

    def test_returns_empty_list_on_db_error(self):
        """load_window should return empty list when a DB error occurs."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.side_effect = Exception("connection lost")

        with patch("server.memory._get_conn", return_value=mock_conn):
            messages = load_window("dsn", uuid.uuid4(), window_size=10)

        assert messages == []

    def test_default_window_size_is_ten(self):
        """The default window_size should be 10."""
        mock_conn = self._make_mock_conn([])
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value
        session_id = uuid.uuid4()

        with patch("server.memory._get_conn", return_value=mock_conn):
            load_window("dsn", session_id)

        _, params = mock_cur.execute.call_args[0]
        assert params[1] == 10


class TestLoadLongTermMemories:
    def test_returns_single_consolidated_system_message(self):
        """load_long_term_memories should return a single system message containing
        all facts as bullet points under a descriptive header."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"memory": "User likes Python"},
            {"memory": "User is a senior engineer"},
        ]

        session_id = uuid.uuid4()
        messages = load_long_term_memories(mem0, session_id)

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "User likes Python" in messages[0]["content"]
        assert "User is a senior engineer" in messages[0]["content"]
        assert "- User likes Python" in messages[0]["content"]
        assert "- User is a senior engineer" in messages[0]["content"]

    def test_consolidated_message_has_descriptive_header(self):
        """The consolidated message must have a header explaining the source."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [{"memory": "A fact"}]

        messages = load_long_term_memories(mem0, uuid.uuid4())

        assert len(messages) == 1
        content = messages[0]["content"]
        assert content.index("A fact") > 0

    def test_respects_long_term_max_cap(self):
        """load_long_term_memories should include at most long_term_max facts."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [{"memory": f"fact {i}"} for i in range(10)]

        session_id = uuid.uuid4()
        messages = load_long_term_memories(mem0, session_id, long_term_max=3)

        assert len(messages) == 1
        content = messages[0]["content"]
        assert content.count("- fact") == 3

    def test_default_long_term_max_is_three(self):
        """Default long_term_max should be 3."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [{"memory": f"fact {i}"} for i in range(10)]

        session_id = uuid.uuid4()
        messages = load_long_term_memories(mem0, session_id)

        assert len(messages) == 1
        assert messages[0]["content"].count("- fact") == 3

    def test_returns_empty_list_when_no_memories(self):
        """load_long_term_memories should return empty list when no memories exist."""
        mem0 = MagicMock()
        mem0.get_all.return_value = []

        messages = load_long_term_memories(mem0, uuid.uuid4())

        assert messages == []

    def test_handles_dict_with_results_key(self):
        """load_long_term_memories should unwrap dict with 'results' key."""
        mem0 = MagicMock()
        mem0.get_all.return_value = {"results": [{"memory": "A fact"}]}

        messages = load_long_term_memories(mem0, uuid.uuid4())

        assert len(messages) == 1
        assert "A fact" in messages[0]["content"]

    def test_returns_empty_list_on_exception(self):
        """load_long_term_memories should return empty list if get_all raises."""
        mem0 = MagicMock()
        mem0.get_all.side_effect = Exception("connection error")

        messages = load_long_term_memories(mem0, uuid.uuid4())

        assert messages == []

    def test_skips_entries_without_content(self):
        """load_long_term_memories should skip items with no extractable content."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"memory": "Valid fact"},
            {"other_field": "no content"},
            {"memory": "Another valid fact"},
        ]

        messages = load_long_term_memories(mem0, uuid.uuid4(), long_term_max=10)

        assert len(messages) == 1
        assert "Valid fact" in messages[0]["content"]
        assert "Another valid fact" in messages[0]["content"]
        assert "no content" not in messages[0]["content"]

    def test_always_uses_system_role_regardless_of_source_role(self):
        """load_long_term_memories should always assign role=system."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"memory": "A fact", "role": "user"},
        ]

        messages = load_long_term_memories(mem0, uuid.uuid4())

        assert messages[0]["role"] == "system"
