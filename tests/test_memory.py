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


def test_get_top_weak_topic():
    """Get topic with highest weakness count, alphabetical tie-breaker."""
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
                    # Create weaknesses with same count to test alphabetical tie-breaker
                    repo.increment_weakness("rag")  # count = 1
                    repo.increment_weakness("prompting")  # count = 1
                    repo.increment_weakness("tokens")  # count = 1
                    # "prompting" should win alphabetically
                    topic = repo.get_top_weak_topic()
        
        assert topic == "prompting"  # Alphabetically first


def test_get_top_weak_topic_none():
    """Return None when no weaknesses exist."""
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
                    topic = repo.get_top_weak_topic()
        
        assert topic is None


def test_get_last_attempt_topic():
    """Get topic from most recent attempt."""
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
                    repo.save_attempt("temperature", "answer2", 70, "feedback2")
                    # Most recent should be "temperature"
                    topic = repo.get_last_attempt_topic()
        
        assert topic == "temperature"


def test_get_last_attempt_topic_none():
    """Return None when no attempts exist."""
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
                    topic = repo.get_last_attempt_topic()
        
        assert topic is None


def test_db_upgrade_adds_llm_columns():
    """Verify DB upgrade adds llm_feedback and llm_score columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test.db")
        
        def mock_get_connection():
            return sqlite3.connect(test_db)
        
        def mock_ensure_data_dir():
            pass
        
        # Create table without llm columns (simulating old DB)
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE lesson_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                answer TEXT NOT NULL,
                score INTEGER NOT NULL,
                feedback TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        
        # Now init_db should add the columns
        with patch("terrance_teacher.memory.repo.get_connection", side_effect=mock_get_connection):
            with patch("terrance_teacher.memory.db.get_connection", side_effect=mock_get_connection):
                with patch("terrance_teacher.memory.db.ensure_data_dir", side_effect=mock_ensure_data_dir):
                    from terrance_teacher.memory.db import init_db
                    init_db()
        
        # Verify columns exist
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(lesson_attempts)")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()
        
        assert "llm_feedback" in columns
        assert "llm_score" in columns


def test_save_attempt_with_llm_fields():
    """Verify save_attempt works with llm_feedback and llm_score."""
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
                    repo.save_attempt(
                        "tokens",
                        "test answer",
                        85,
                        "Good job!",
                        llm_feedback="This is LLM feedback",
                        llm_score=90,
                    )
        
        # Verify data was saved
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT llm_feedback, llm_score FROM lesson_attempts WHERE topic = ?",
            ("tokens",)
        )
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == "This is LLM feedback"
        assert row[1] == 90


def test_save_attempt_without_llm_fields():
    """Verify save_attempt works without llm fields (backward compatibility)."""
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
        
        # Verify data was saved with NULL llm fields
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT llm_feedback, llm_score FROM lesson_attempts WHERE topic = ?",
            ("tokens",)
        )
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] is None
        assert row[1] is None


def test_get_history_empty_returns_empty_list():
    """Verify get_history returns empty list when no attempts exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test.db")
        
        def mock_get_connection():
            return sqlite3.connect(test_db)
        
        def mock_ensure_data_dir():
            pass
        
        with patch("terrance_teacher.memory.repo.get_connection", side_effect=mock_get_connection):
            with patch("terrance_teacher.memory.db.get_connection", side_effect=mock_get_connection):
                with patch("terrance_teacher.memory.db.ensure_data_dir", side_effect=mock_ensure_data_dir):
                    from terrance_teacher.memory.db import init_db
                    init_db()
                    
                    repo = MemoryRepository()
                    history = repo.get_history()
        
        assert history == []


def test_get_history_returns_most_recent_first():
    """Verify get_history returns attempts ordered by most recent first."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test.db")
        
        def mock_get_connection():
            return sqlite3.connect(test_db)
        
        def mock_ensure_data_dir():
            pass
        
        with patch("terrance_teacher.memory.repo.get_connection", side_effect=mock_get_connection):
            with patch("terrance_teacher.memory.db.get_connection", side_effect=mock_get_connection):
                with patch("terrance_teacher.memory.db.ensure_data_dir", side_effect=mock_ensure_data_dir):
                    from terrance_teacher.memory.db import init_db
                    init_db()
                    
                    repo = MemoryRepository()
                    # Insert 3 attempts with different timestamps (using direct SQL to control timestamps)
                    conn = sqlite3.connect(test_db)
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO lesson_attempts (topic, answer, score, feedback, created_at, llm_feedback, llm_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        ("tokens", "answer1", 85, "feedback1", "2024-01-01T00:00:00", None, None)
                    )
                    cursor.execute(
                        "INSERT INTO lesson_attempts (topic, answer, score, feedback, created_at, llm_feedback, llm_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        ("temperature", "answer2", 70, "feedback2", "2024-01-02T00:00:00", None, None)
                    )
                    cursor.execute(
                        "INSERT INTO lesson_attempts (topic, answer, score, feedback, created_at, llm_feedback, llm_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        ("tokens", "answer3", 90, "feedback3", "2024-01-03T00:00:00", "LLM feedback", None)
                    )
                    conn.commit()
                    conn.close()
        
        # Check that history returns in correct order
        with patch("terrance_teacher.memory.repo.get_connection", side_effect=mock_get_connection):
            with patch("terrance_teacher.memory.db.get_connection", side_effect=mock_get_connection):
                with patch("terrance_teacher.memory.db.ensure_data_dir", side_effect=mock_ensure_data_dir):
                    repo = MemoryRepository()
                    history = repo.get_history(limit=3)
        
        assert len(history) == 3
        assert history[0]["topic"] == "tokens"
        assert history[0]["created_at"] == "2024-01-03T00:00:00"
        assert history[0]["has_llm_feedback"] is True
        assert history[1]["topic"] == "temperature"
        assert history[1]["created_at"] == "2024-01-02T00:00:00"
        assert history[1]["has_llm_feedback"] is False
        assert history[2]["topic"] == "tokens"
        assert history[2]["created_at"] == "2024-01-01T00:00:00"
        assert history[2]["has_llm_feedback"] is False


def test_get_topic_summary_no_attempts_defaults():
    """Verify get_topic_summary returns safe defaults when no attempts exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test.db")
        
        def mock_get_connection():
            return sqlite3.connect(test_db)
        
        def mock_ensure_data_dir():
            pass
        
        with patch("terrance_teacher.memory.repo.get_connection", side_effect=mock_get_connection):
            with patch("terrance_teacher.memory.db.get_connection", side_effect=mock_get_connection):
                with patch("terrance_teacher.memory.db.ensure_data_dir", side_effect=mock_ensure_data_dir):
                    from terrance_teacher.memory.db import init_db
                    init_db()
                    
                    repo = MemoryRepository()
                    summary = repo.get_topic_summary("nonexistent_topic")
        
        assert summary["topic"] == "nonexistent_topic"
        assert summary["total_attempts"] == 0
        assert summary["average_score"] == 0.0
        assert summary["last_attempt_at"] is None
        assert summary["last_score"] is None
        assert summary["weakness_count"] == 0
        assert summary["llm_feedback_rate"] == 0.0


def test_get_topic_summary_with_attempts_and_llm_rate():
    """Verify get_topic_summary calculates all fields correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test.db")
        
        def mock_get_connection():
            return sqlite3.connect(test_db)
        
        def mock_ensure_data_dir():
            pass
        
        with patch("terrance_teacher.memory.repo.get_connection", side_effect=mock_get_connection):
            with patch("terrance_teacher.memory.db.get_connection", side_effect=mock_get_connection):
                with patch("terrance_teacher.memory.db.ensure_data_dir", side_effect=mock_ensure_data_dir):
                    from terrance_teacher.memory.db import init_db
                    init_db()
                    
                    repo = MemoryRepository()
                    # Insert attempts: 2 with LLM feedback, 1 without
                    repo.save_attempt("tokens", "answer1", 85, "feedback1", llm_feedback="LLM1")
                    repo.save_attempt("tokens", "answer2", 70, "feedback2", llm_feedback="LLM2")
                    repo.save_attempt("tokens", "answer3", 90, "feedback3")
                    # Add weakness
                    repo.increment_weakness("tokens")
        
        # Verify summary
        with patch("terrance_teacher.memory.repo.get_connection", side_effect=mock_get_connection):
            with patch("terrance_teacher.memory.db.get_connection", side_effect=mock_get_connection):
                with patch("terrance_teacher.memory.db.ensure_data_dir", side_effect=mock_ensure_data_dir):
                    repo = MemoryRepository()
                    summary = repo.get_topic_summary("tokens")
        
        assert summary["topic"] == "tokens"
        assert summary["total_attempts"] == 3
        assert summary["average_score"] == (85 + 70 + 90) / 3.0
        assert summary["last_attempt_at"] is not None
        assert summary["last_score"] == 90
        assert summary["weakness_count"] == 1
        # 2 out of 3 have LLM feedback
        assert abs(summary["llm_feedback_rate"] - (2.0 / 3.0)) < 0.001

