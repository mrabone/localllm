import uuid
from unittest.mock import MagicMock, patch

import pytest

from common.session_store import create_session, session_exists


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
        with patch("common.session_store.get_conn", return_value=mock_conn):
            session_id = create_session("dsn")
        assert isinstance(session_id, uuid.UUID)

    def test_each_call_returns_unique_uuid(self):
        """create_session should return a different UUID each time."""
        mock_conn, _ = _make_mock_conn()
        with patch("common.session_store.get_conn", return_value=mock_conn):
            session_id_1 = create_session("dsn")
            session_id_2 = create_session("dsn")
        assert session_id_1 != session_id_2

    def test_returned_uuid_is_valid(self):
        """The returned UUID should be valid and convertible to string."""
        mock_conn, _ = _make_mock_conn()
        with patch("common.session_store.get_conn", return_value=mock_conn):
            session_id = create_session("dsn")
        str_id = str(session_id)
        assert len(str_id) == 36
        assert str_id.count("-") == 4

    def test_inserts_row_into_chat_sessions(self):
        """create_session should INSERT the new UUID into chat_sessions."""
        mock_conn, mock_cur = _make_mock_conn()
        with patch("common.session_store.get_conn", return_value=mock_conn):
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
        with patch("common.session_store.get_conn", return_value=mock_conn):
            result = session_exists("dsn", session_id)

        assert result is True

    def test_returns_false_when_no_row(self):
        """session_exists should return False when chat_sessions has no matching row."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = None

        session_id = uuid.uuid4()
        with patch("common.session_store.get_conn", return_value=mock_conn):
            result = session_exists("dsn", session_id)

        assert result is False

    def test_raises_on_db_error(self):
        """session_exists should propagate DB exceptions to the caller."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.side_effect = Exception("Connection error")

        session_id = uuid.uuid4()
        with patch("common.session_store.get_conn", return_value=mock_conn):
            with pytest.raises(Exception, match="Connection error"):
                session_exists("dsn", session_id)

    def test_converts_session_id_to_string(self):
        """session_exists should pass the UUID as a string to the SQL query."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = None

        session_id = uuid.uuid4()
        with patch("common.session_store.get_conn", return_value=mock_conn):
            session_exists("dsn", session_id)

        _, params = mock_cur.execute.call_args[0]
        assert str(session_id) in params
