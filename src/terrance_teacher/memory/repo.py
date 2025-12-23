from datetime import datetime, timezone
from terrance_teacher.memory.db import get_connection, init_db


class MemoryRepository:
    def __init__(self):
        init_db()
    
    def save_attempt(self, topic: str, answer: str, score: int, feedback: str) -> None:
        """Save a lesson attempt to the database."""
        conn = get_connection()
        cursor = conn.cursor()
        
        created_at = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "INSERT INTO lesson_attempts (topic, answer, score, feedback, created_at) VALUES (?, ?, ?, ?, ?)",
            (topic, answer, score, feedback, created_at)
        )
        
        conn.commit()
        conn.close()
    
    def increment_weakness(self, topic: str) -> None:
        """Increment weakness count for a topic, or insert if not exists."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT count FROM weaknesses WHERE topic = ?", (topic,))
        row = cursor.fetchone()
        
        if row:
            cursor.execute("UPDATE weaknesses SET count = count + 1 WHERE topic = ?", (topic,))
        else:
            cursor.execute("INSERT INTO weaknesses (topic, count) VALUES (?, ?)", (topic, 1))
        
        conn.commit()
        conn.close()
    
    def get_status_summary(self) -> dict:
        """Get status summary: total attempts, average score, weakest topics."""
        conn = get_connection()
        cursor = conn.cursor()
        
        # Total attempts and average score
        cursor.execute("SELECT COUNT(*), AVG(score) FROM lesson_attempts")
        row = cursor.fetchone()
        total_attempts = row[0] if row[0] else 0
        average_score = float(row[1]) if row[1] is not None else 0.0
        
        # Weakest topics (top 5, sorted by count descending)
        cursor.execute("SELECT topic, count FROM weaknesses ORDER BY count DESC LIMIT 5")
        weakest_topics = [(row[0], row[1]) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "total_attempts": total_attempts,
            "average_score": average_score,
            "weakest_topics": weakest_topics,
        }
    
    def get_top_weak_topic(self) -> str | None:
        """Get topic with highest weakness count, with alphabetical tie-breaker."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT topic FROM weaknesses ORDER BY count DESC, topic ASC LIMIT 1"
        )
        row = cursor.fetchone()
        
        conn.close()
        
        return row[0] if row else None
    
    def get_last_attempt_topic(self) -> str | None:
        """Get topic from most recent attempt."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT topic FROM lesson_attempts ORDER BY created_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        
        conn.close()
        
        return row[0] if row else None

