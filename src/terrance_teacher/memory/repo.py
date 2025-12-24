from datetime import datetime, timezone
from terrance_teacher.memory.db import get_connection, init_db


class MemoryRepository:
    def __init__(self):
        init_db()
    
    def save_attempt(
        self,
        topic: str,
        answer: str,
        score: int,
        feedback: str,
        llm_feedback: str | None = None,
        llm_score: int | None = None,
    ) -> None:
        """Save a lesson attempt to the database."""
        conn = get_connection()
        cursor = conn.cursor()
        
        created_at = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "INSERT INTO lesson_attempts (topic, answer, score, feedback, created_at, llm_feedback, llm_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (topic, answer, score, feedback, created_at, llm_feedback, llm_score)
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
    
    def get_last_attempt_llm_feedback(self) -> str | None:
        """Get LLM feedback from most recent attempt."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT llm_feedback FROM lesson_attempts ORDER BY created_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        
        conn.close()
        
        return row[0] if row and row[0] else None
    
    def get_weakness_count(self, topic: str) -> int:
        """Get weakness count for a topic, returns 0 if not found."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT count FROM weaknesses WHERE topic = ?", (topic,))
        row = cursor.fetchone()
        
        conn.close()
        
        return row[0] if row else 0
    
    def get_history(self, limit: int = 10) -> list[dict]:
        """Get most recent lesson attempts.
        
        Returns list of dicts with:
        - topic: str
        - score: int
        - created_at: str
        - has_llm_feedback: bool
        """
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT topic, score, created_at, llm_feedback FROM lesson_attempts "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        
        conn.close()
        
        history = []
        for row in rows:
            topic, score, created_at, llm_feedback = row
            has_llm_feedback = bool(llm_feedback and llm_feedback.strip())
            history.append({
                "topic": topic,
                "score": score,
                "created_at": created_at,
                "has_llm_feedback": has_llm_feedback,
            })
        
        return history
    
    def get_topic_summary(self, topic: str) -> dict:
        """Get summary statistics for a specific topic.
        
        Returns dict with:
        - topic: str
        - total_attempts: int
        - average_score: float (0.0 if none)
        - last_attempt_at: str | None
        - last_score: int | None
        - weakness_count: int (0 if none)
        - llm_feedback_rate: float (0.0 if none; fraction of attempts with llm_feedback)
        """
        conn = get_connection()
        cursor = conn.cursor()
        
        # Total attempts and average score
        cursor.execute(
            "SELECT COUNT(*), AVG(score) FROM lesson_attempts WHERE topic = ?",
            (topic,)
        )
        row = cursor.fetchone()
        total_attempts = row[0] if row[0] else 0
        average_score = float(row[1]) if row[1] is not None else 0.0
        
        # Last attempt
        cursor.execute(
            "SELECT created_at, score FROM lesson_attempts WHERE topic = ? "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (topic,)
        )
        last_row = cursor.fetchone()
        last_attempt_at = last_row[0] if last_row else None
        last_score = last_row[1] if last_row else None
        
        # LLM feedback count
        cursor.execute(
            "SELECT COUNT(*) FROM lesson_attempts "
            "WHERE topic = ? AND llm_feedback IS NOT NULL AND TRIM(llm_feedback) != ''",
            (topic,)
        )
        llm_row = cursor.fetchone()
        llm_feedback_count = llm_row[0] if llm_row else 0
        
        llm_feedback_rate = (llm_feedback_count / total_attempts) if total_attempts > 0 else 0.0
        
        # Weakness count
        cursor.execute("SELECT count FROM weaknesses WHERE topic = ?", (topic,))
        weakness_row = cursor.fetchone()
        weakness_count = weakness_row[0] if weakness_row else 0
        
        conn.close()
        
        return {
            "topic": topic,
            "total_attempts": total_attempts,
            "average_score": average_score,
            "last_attempt_at": last_attempt_at,
            "last_score": last_score,
            "weakness_count": weakness_count,
            "llm_feedback_rate": llm_feedback_rate,
        }

