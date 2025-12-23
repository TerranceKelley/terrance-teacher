from unittest.mock import Mock
from terrance_teacher.core.orchestrator import TeacherOrchestrator


def test_generate_lesson_known_topic_tokens():
    orchestrator = TeacherOrchestrator()

    lesson = orchestrator.generate_lesson("tokens")

    assert lesson.topic == "tokens"
    assert "Tokens are the fundamental units that LLMs process" in lesson.explanation
    assert "Why does the same text produce different token counts" in lesson.question
    assert "Write a Python function" in lesson.task


def test_generate_lesson_unknown_topic_placeholder():
    orchestrator = TeacherOrchestrator()
    topic = "quantum"

    lesson = orchestrator.generate_lesson(topic)

    assert lesson.topic == topic
    assert "placeholder lesson" in lesson.explanation
    assert f"Research '{topic}' independently" in lesson.task


def test_grade_answer_tokens_with_keywords():
    orchestrator = TeacherOrchestrator()
    answer = (
        "Token limit constraints lead to truncation when the context window is exceeded,"
        " which is critical for prompt design."
    )

    grade = orchestrator.grade_answer("tokens", answer)

    assert grade.score == 85
    assert "key concepts" in grade.feedback


def test_grade_answer_unknown_topic_placeholder():
    orchestrator = TeacherOrchestrator()

    grade = orchestrator.grade_answer("math", "Some generic answer")

    assert grade.score == 50
    assert "placeholder grade" in grade.feedback


def test_grade_answer_without_memory_repo():
    """Verify grading works without memory repository (backward compatibility)."""
    orchestrator = TeacherOrchestrator(memory_repo=None)
    answer = (
        "Token limit constraints lead to truncation when the context window is exceeded,"
        " which is critical for prompt design."
    )

    grade = orchestrator.grade_answer("tokens", answer)

    assert grade.score == 85
    assert "key concepts" in grade.feedback


def test_grade_answer_persists_attempt():
    """Verify grading saves attempt to memory repository."""
    mock_repo = Mock()
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo)
    answer = "Context window limits affect tokenization."

    grade = orchestrator.grade_answer("tokens", answer)

    assert grade.score == 65  # One keyword found
    mock_repo.save_attempt.assert_called_once_with("tokens", answer, 65, grade.feedback)


def test_grade_answer_tracks_weakness_low_score():
    """Verify weakness is tracked when score < 70."""
    mock_repo = Mock()
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo)
    answer = "I don't know much about tokens."

    grade = orchestrator.grade_answer("tokens", answer)

    assert grade.score == 35  # Low score
    mock_repo.save_attempt.assert_called_once()
    mock_repo.increment_weakness.assert_called_once_with("tokens")


def test_grade_answer_no_weakness_high_score():
    """Verify weakness is NOT tracked when score >= 70."""
    mock_repo = Mock()
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo)
    answer = (
        "Token limit constraints lead to truncation when the context window is exceeded,"
        " which is critical for prompt design."
    )

    grade = orchestrator.grade_answer("tokens", answer)

    assert grade.score == 85  # High score
    mock_repo.save_attempt.assert_called_once()
    mock_repo.increment_weakness.assert_not_called()


def test_recommend_next_topic_no_attempts():
    """Recommend 'tokens' when no attempts exist."""
    mock_repo = Mock()
    mock_repo.get_top_weak_topic.return_value = None
    mock_repo.get_last_attempt_topic.return_value = None
    
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo)
    topic = orchestrator.recommend_next_topic()
    
    assert topic == "tokens"


def test_recommend_next_topic_with_attempts_no_weaknesses():
    """Recommend next topic in curriculum after last attempt."""
    mock_repo = Mock()
    mock_repo.get_top_weak_topic.return_value = None
    mock_repo.get_last_attempt_topic.return_value = "tokens"
    
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo)
    topic = orchestrator.recommend_next_topic()
    
    assert topic == "temperature"


def test_recommend_next_topic_wraps_to_first():
    """Recommend first topic when last attempt was final topic."""
    mock_repo = Mock()
    mock_repo.get_top_weak_topic.return_value = None
    mock_repo.get_last_attempt_topic.return_value = "agents"
    
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo)
    topic = orchestrator.recommend_next_topic()
    
    assert topic == "tokens"


def test_recommend_next_topic_with_weaknesses():
    """Prioritize weaknesses over curriculum progression."""
    mock_repo = Mock()
    mock_repo.get_top_weak_topic.return_value = "rag"
    mock_repo.get_last_attempt_topic.return_value = "tokens"
    
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo)
    topic = orchestrator.recommend_next_topic()
    
    assert topic == "rag"
    mock_repo.get_last_attempt_topic.assert_not_called()


def test_recommend_next_topic_alphabetical_tiebreaker():
    """Verify alphabetical tie-breaker for weaknesses."""
    mock_repo = Mock()
    mock_repo.get_top_weak_topic.return_value = "prompting"  # Alphabetically first among ties
    
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo)
    topic = orchestrator.recommend_next_topic()
    
    assert topic == "prompting"


def test_recommend_next_topic_no_memory_repo():
    """Return default when no memory repository."""
    orchestrator = TeacherOrchestrator(memory_repo=None)
    topic = orchestrator.recommend_next_topic()
    
    assert topic == "tokens"
