import sqlite3
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
from terrance_teacher.memory.repo import MemoryRepository


def test_save_attempt():
    """Verify attempt is saved with correct fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test.db")
        
        def mock_get_connection():
            return sqlite3.connect(test_db)
        
        def mock_ensure_data_dir():
            pass
        
        with patch("terrance_teacher.memory.repo.get_connection", side_effect=mock_get_connection):
            with patch("terrance_teacher.memory.db.get_connection", side_effect=mock_get_connection):
                with patch("terrance_teacher.memory.db.ensure_data_dir", side_effect=mock_ensure_data_dir):
                    repo = MemoryRepository()
                    repo.save_attempt("tokens", "test answer", 85, "Good job!")
        
        # Verify data was saved
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT topic, answer, score, feedback FROM lesson_attempts")
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == "tokens"
        assert row[1] == "test answer"
        assert row[2] == 85
        assert row[3] == "Good job!"


def test_increment_weakness_new_topic():
    """Insert new weakness when topic doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test.db")
        
        def mock_get_connection():
            return sqlite3.connect(test_db)
        
        def mock_ensure_data_dir():
            pass
        
        with patch("terrance_teacher.memory.repo.get_connection", side_effect=mock_get_connection):
            with patch("terrance_teacher.memory.db.get_connection", side_effect=mock_get_connection):
                with patch("terrance_teacher.memory.db.ensure_data_dir", side_effect=mock_ensure_data_dir):
                    repo = MemoryRepository()
                    repo.increment_weakness("tokens")
        
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT topic, count FROM weaknesses WHERE topic = ?", ("tokens",))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == "tokens"
        assert row[1] == 1


def test_increment_weakness_existing_topic():
    """Increment existing weakness count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test.db")
        
        def mock_get_connection():
            return sqlite3.connect(test_db)
        
        def mock_ensure_data_dir():
            pass
        
        with patch("terrance_teacher.memory.repo.get_connection", side_effect=mock_get_connection):
            with patch("terrance_teacher.memory.db.get_connection", side_effect=mock_get_connection):
                with patch("terrance_teacher.memory.db.ensure_data_dir", side_effect=mock_ensure_data_dir):
                    repo = MemoryRepository()
                    # First increment
                    repo.increment_weakness("tokens")
                    # Second increment
                    repo.increment_weakness("tokens")
        
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT count FROM weaknesses WHERE topic = ?", ("tokens",))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == 2


def test_get_status_summary_empty():
    """Handle empty database gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test.db")
        
        def mock_get_connection():
            return sqlite3.connect(test_db)
        
        def mock_ensure_data_dir():
            pass
        
        with patch("terrance_teacher.memory.repo.get_connection", side_effect=mock_get_connection):
            with patch("terrance_teacher.memory.db.get_connection", side_effect=mock_get_connection):
                with patch("terrance_teacher.memory.db.ensure_data_dir", side_effect=mock_ensure_data_dir):
                    repo = MemoryRepository()
                    summary = repo.get_status_summary()
        
        assert summary["total_attempts"] == 0
        assert summary["average_score"] == 0.0
        assert summary["weakest_topics"] == []


def test_get_status_summary_with_data():
    """Calculate stats correctly with data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test.db")
        
        def mock_get_connection():
            return sqlite3.connect(test_db)
        
        def mock_ensure_data_dir():
            pass
        
        with patch("terrance_teacher.memory.repo.get_connection", side_effect=mock_get_connection):
            with patch("terrance_teacher.memory.db.get_connection", side_effect=mock_get_connection):
                with patch("terrance_teacher.memory.db.ensure_data_dir", side_effect=mock_ensure_data_dir):
                    repo = MemoryRepository()
                    repo.save_attempt("tokens", "answer1", 80, "feedback1")
                    repo.save_attempt("tokens", "answer2", 60, "feedback2")
                    repo.save_attempt("rag", "answer3", 90, "feedback3")
                    summary = repo.get_status_summary()
        
        assert summary["total_attempts"] == 3
        assert summary["average_score"] == (80 + 60 + 90) / 3
        assert len(summary["weakest_topics"]) == 0  # No weaknesses tracked yet


def test_get_status_summary_weakest_topics():
    """Top 5 sorting and limiting works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test.db")
        
        def mock_get_connection():
            return sqlite3.connect(test_db)
        
        def mock_ensure_data_dir():
            pass
        
        with patch("terrance_teacher.memory.repo.get_connection", side_effect=mock_get_connection):
            with patch("terrance_teacher.memory.db.get_connection", side_effect=mock_get_connection):
                with patch("terrance_teacher.memory.db.ensure_data_dir", side_effect=mock_ensure_data_dir):
                    repo = MemoryRepository()
                    # Create multiple weaknesses with different counts
                    repo.increment_weakness("tokens")
                    repo.increment_weakness("tokens")
                    repo.increment_weakness("tokens")  # count = 3
                    repo.increment_weakness("rag")  # count = 1
                    repo.increment_weakness("temperature")
                    repo.increment_weakness("temperature")  # count = 2
                    repo.increment_weakness("prompts")  # count = 1
                    repo.increment_weakness("hallucination")  # count = 1
                    repo.increment_weakness("agents")  # count = 1
                    summary = repo.get_status_summary()
        
        weakest = summary["weakest_topics"]
        assert len(weakest) == 5  # Should limit to top 5
        assert weakest[0][0] == "tokens"  # Highest count first
        assert weakest[0][1] == 3
        assert weakest[1][0] == "temperature"
        assert weakest[1][1] == 2

