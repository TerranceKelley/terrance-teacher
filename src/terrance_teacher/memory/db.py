import os
import sqlite3
from pathlib import Path


def ensure_data_dir() -> None:
    """Create data directory if it doesn't exist."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Get SQLite connection to data/teacher.db."""
    ensure_data_dir()
    db_path = Path("data") / "teacher.db"
    return sqlite3.connect(str(db_path))


def init_db() -> None:
    """Initialize database tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lesson_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            answer TEXT NOT NULL,
            score INTEGER NOT NULL,
            feedback TEXT NOT NULL,
            created_at TEXT NOT NULL,
            llm_feedback TEXT,
            llm_score INTEGER
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weaknesses (
            topic TEXT PRIMARY KEY,
            count INTEGER NOT NULL DEFAULT 1
        )
    """)
    
    # Backward-compatible schema upgrade: add llm columns if missing
    cursor.execute("PRAGMA table_info(lesson_attempts)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "llm_feedback" not in columns:
        cursor.execute("ALTER TABLE lesson_attempts ADD COLUMN llm_feedback TEXT")
    
    if "llm_score" not in columns:
        cursor.execute("ALTER TABLE lesson_attempts ADD COLUMN llm_score INTEGER")
    
    conn.commit()
    conn.close()

