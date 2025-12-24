import json
from unittest.mock import Mock, patch
from terrance_teacher.core.orchestrator import TeacherOrchestrator
from terrance_teacher.llm import OllamaClient


def test_generate_lesson_known_topic_tokens():
    orchestrator = TeacherOrchestrator()

    lesson = orchestrator.generate_lesson("tokens")

    assert lesson.topic == "tokens"
    assert "Tokens are the fundamental units that LLMs process" in lesson.explanation
    assert "Why does the same text produce different token counts" in lesson.question
    assert "Write a Python function" in lesson.task


def test_generate_lesson_unknown_topic_placeholder():
    """Verify fallback lesson is generated when Ollama unavailable."""
    mock_ollama = Mock(spec=OllamaClient)
    mock_ollama.generate.return_value = None  # Force fallback
    orchestrator = TeacherOrchestrator(ollama_client=mock_ollama)
    topic = "quantum"

    lesson = orchestrator.generate_lesson(topic)

    assert lesson.topic == topic
    assert "placeholder lesson" in lesson.explanation.lower() or "[Beginner]" in lesson.explanation
    assert f"Research '{topic}' independently" in lesson.task or "brief summary" in lesson.task


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
    mock_ollama = Mock(spec=OllamaClient)
    mock_ollama.generate.return_value = None  # Force no LLM feedback
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo, ollama_client=mock_ollama)
    answer = "Context window limits affect tokenization."

    grade = orchestrator.grade_answer("tokens", answer)

    assert grade.score == 65  # One keyword found
    mock_repo.save_attempt.assert_called_once_with(
        "tokens", answer, 65, grade.feedback, llm_feedback=None, llm_score=None
    )


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


def test_grade_answer_deterministic_unchanged():
    """Verify deterministic grading logic is unchanged."""
    orchestrator = TeacherOrchestrator(memory_repo=None)
    answer = (
        "Token limit constraints lead to truncation when the context window is exceeded,"
        " which is critical for prompt design."
    )
    
    grade = orchestrator.grade_answer("tokens", answer)
    
    # Verify deterministic score and feedback unchanged
    assert grade.score == 85
    assert "key concepts" in grade.feedback
    assert "truncation" in grade.feedback or "context window" in grade.feedback


def test_grade_answer_with_ollama_feedback():
    """Verify LLM feedback is generated and persisted when Ollama available."""
    mock_repo = Mock()
    mock_ollama = Mock(spec=OllamaClient)
    
    valid_json_response = json.dumps({
        "feedback": "Your understanding of tokenization is solid. Consider exploring how different tokenizers handle multilingual text.",
        "score": 88
    })
    mock_ollama.generate.return_value = valid_json_response
    
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo, ollama_client=mock_ollama)
    answer = "Context window limits are important."  # Only one keyword
    
    grade = orchestrator.grade_answer("tokens", answer)
    
    # Verify deterministic grade still computed
    assert grade.score == 65  # One keyword found
    
    # Verify save_attempt called with llm_feedback
    mock_repo.save_attempt.assert_called_once()
    call_args = mock_repo.save_attempt.call_args
    assert call_args[0][0] == "tokens"  # topic
    assert call_args[1]["llm_feedback"] is not None
    assert "tokenization is solid" in call_args[1]["llm_feedback"].lower()
    assert call_args[1]["llm_score"] == 88


def test_grade_answer_falls_back_when_ollama_unavailable():
    """Verify deterministic grading still works when Ollama unavailable."""
    mock_repo = Mock()
    mock_ollama = Mock(spec=OllamaClient)
    mock_ollama.generate.return_value = None
    
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo, ollama_client=mock_ollama)
    answer = "Context window limits are important."  # Only one keyword
    
    grade = orchestrator.grade_answer("tokens", answer)
    
    # Verify deterministic grade computed
    assert grade.score == 65
    
    # Verify save_attempt called without llm fields
    mock_repo.save_attempt.assert_called_once()
    call_args = mock_repo.save_attempt.call_args
    assert call_args[1]["llm_feedback"] is None
    assert call_args[1]["llm_score"] is None


def test_generate_lesson_keeps_hardcoded_tokens():
    """Verify hardcoded tokens lesson is unchanged."""
    orchestrator = TeacherOrchestrator()
    lesson = orchestrator.generate_lesson("tokens")
    
    assert lesson.topic == "tokens"
    assert "Tokens are the fundamental units that LLMs process" in lesson.explanation
    assert "Why does the same text produce different token counts" in lesson.question


def test_generate_lesson_falls_back_when_ollama_unavailable():
    """Verify fallback to scaffold lesson when Ollama is unavailable."""
    mock_ollama = Mock(spec=OllamaClient)
    mock_ollama.generate.return_value = None
    
    orchestrator = TeacherOrchestrator(ollama_client=mock_ollama)
    lesson = orchestrator.generate_lesson("new_topic")
    
    assert lesson.topic == "new_topic"
    assert "placeholder lesson" in lesson.explanation.lower()
    assert "not yet available" in lesson.explanation


def test_generate_lesson_parses_valid_json():
    """Verify successful parsing of Ollama-generated JSON."""
    valid_json_response = json.dumps({
        "topic": "temperature",
        "explanation": "Temperature controls randomness in LLM outputs.",
        "example": "Temperature 0.0 = deterministic, 1.0 = creative",
        "question": "How does temperature affect output diversity?",
        "task": "Experiment with different temperature values."
    })
    
    mock_ollama = Mock(spec=OllamaClient)
    mock_ollama.generate.return_value = valid_json_response
    
    orchestrator = TeacherOrchestrator(ollama_client=mock_ollama)
    lesson = orchestrator.generate_lesson("temperature")
    
    assert lesson.topic == "temperature"
    assert "Temperature controls randomness" in lesson.explanation
    assert "How does temperature affect" in lesson.question
    assert "Experiment with different temperature" in lesson.task


def test_generate_lesson_handles_json_in_code_fences():
    """Verify JSON extraction works when wrapped in markdown code fences."""
    json_with_fences = "```json\n" + json.dumps({
        "topic": "rag",
        "explanation": "RAG combines retrieval with generation.",
        "example": "Query vector DB, then generate answer",
        "question": "What is RAG?",
        "task": "Build a simple RAG system."
    }) + "\n```"
    
    mock_ollama = Mock(spec=OllamaClient)
    mock_ollama.generate.return_value = json_with_fences
    
    orchestrator = TeacherOrchestrator(ollama_client=mock_ollama)
    lesson = orchestrator.generate_lesson("rag")
    
    assert lesson.topic == "rag"
    assert "RAG combines retrieval" in lesson.explanation


def test_generate_lesson_falls_back_on_invalid_json():
    """Verify fallback when Ollama returns invalid JSON."""
    mock_ollama = Mock(spec=OllamaClient)
    mock_ollama.generate.return_value = "This is not valid JSON at all"
    
    orchestrator = TeacherOrchestrator(ollama_client=mock_ollama)
    lesson = orchestrator.generate_lesson("invalid_topic")
    
    assert lesson.topic == "invalid_topic"
    assert "placeholder lesson" in lesson.explanation.lower()


def test_fallback_lesson_beginner_when_no_memory():
    """Verify fallback lesson is beginner tier when no memory repository."""
    mock_ollama_client = Mock(spec=OllamaClient)
    mock_ollama_client.generate.return_value = None  # Force fallback
    orchestrator = TeacherOrchestrator(memory_repo=None, ollama_client=mock_ollama_client)
    
    lesson = orchestrator.generate_lesson("test_topic")
    
    assert lesson.topic == "test_topic"
    assert "[Beginner]" in lesson.explanation
    # Beginner should have shorter explanation
    assert len(lesson.explanation) < 300
    assert "brief summary" in lesson.task or "1-2 paragraphs" in lesson.task


def test_fallback_lesson_intermediate_when_weakness_count_2():
    """Verify fallback lesson is intermediate tier when weakness_count is 2."""
    mock_repo = Mock()
    mock_repo.get_weakness_count.return_value = 2
    mock_ollama_client = Mock(spec=OllamaClient)
    mock_ollama_client.generate.return_value = None  # Force fallback
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo, ollama_client=mock_ollama_client)
    
    lesson = orchestrator.generate_lesson("test_topic")
    
    assert lesson.topic == "test_topic"
    assert "[Intermediate]" in lesson.explanation
    # Intermediate should have longer explanation than beginner
    assert len(lesson.explanation) > 300
    assert "tradeoff" in lesson.explanation.lower() or "edge case" in lesson.explanation.lower()
    assert "reasoning" in lesson.question.lower() or "tradeoff" in lesson.question.lower()


def test_fallback_lesson_advanced_when_weakness_count_4():
    """Verify fallback lesson is advanced tier when weakness_count is 4."""
    mock_repo = Mock()
    mock_repo.get_weakness_count.return_value = 4
    mock_ollama_client = Mock(spec=OllamaClient)
    mock_ollama_client.generate.return_value = None  # Force fallback
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo, ollama_client=mock_ollama_client)
    
    lesson = orchestrator.generate_lesson("test_topic")
    
    assert lesson.topic == "test_topic"
    assert "[Advanced]" in lesson.explanation
    # Advanced should have longest explanation
    assert len(lesson.explanation) > 400
    assert "failure mode" in lesson.explanation.lower() or "operational" in lesson.explanation.lower()
    assert "systems thinking" in lesson.question.lower() or "production" in lesson.question.lower()
    assert "Design" in lesson.task and "implement" in lesson.task.lower()


def test_ollama_success_unchanged():
    """Verify Ollama-generated lessons are returned unchanged (no fallback tiering applied)."""
    mock_ollama_client = Mock(spec=OllamaClient)
    ollama_lesson_json = json.dumps({
        "topic": "test_topic",
        "explanation": "This is an Ollama-generated explanation that should be returned as-is.",
        "example": "This is an Ollama-generated example.",
        "question": "This is an Ollama-generated question?",
        "task": "This is an Ollama-generated task."
    })
    mock_ollama_client.generate.return_value = ollama_lesson_json
    orchestrator = TeacherOrchestrator(ollama_client=mock_ollama_client)
    
    lesson = orchestrator.generate_lesson("test_topic")
    
    # Verify Ollama lesson is returned unchanged
    assert lesson.topic == "test_topic"
    assert "Ollama-generated explanation" in lesson.explanation
    assert "Ollama-generated example" in lesson.example
    assert "Ollama-generated question" in lesson.question
    assert "Ollama-generated task" in lesson.task
    # Verify no tier markers from fallback
    assert "[Beginner]" not in lesson.explanation
    assert "[Intermediate]" not in lesson.explanation
    assert "[Advanced]" not in lesson.explanation


def test_orchestrator_delegates_to_teacher():
    """Verify orchestrator delegates generate_lesson to Teacher."""
    from terrance_teacher.core.teacher import Teacher
    
    mock_teacher = Mock(spec=Teacher)
    mock_lesson = Mock()
    mock_teacher.generate_lesson.return_value = mock_lesson
    
    orchestrator = TeacherOrchestrator()
    orchestrator._teacher = mock_teacher
    
    result = orchestrator.generate_lesson("test_topic")
    
    mock_teacher.generate_lesson.assert_called_once_with("test_topic")
    assert result is mock_lesson


def test_orchestrator_delegates_to_examiner():
    """Verify orchestrator delegates grade_answer to Examiner and handles persistence."""
    from terrance_teacher.core.examiner import Examiner
    from terrance_teacher.core.models import Grade, LlmGrade
    
    mock_examiner = Mock(spec=Examiner)
    mock_grade = Grade(score=85, feedback="Good job!")
    mock_llm_grade = LlmGrade(feedback="LLM feedback", score=90)
    mock_examiner.grade_answer.return_value = (mock_grade, mock_llm_grade)
    
    mock_repo = Mock()
    
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo)
    orchestrator._examiner = mock_examiner
    
    result = orchestrator.grade_answer("tokens", "test answer")
    
    mock_examiner.grade_answer.assert_called_once_with("tokens", "test answer")
    mock_repo.save_attempt.assert_called_once_with(
        "tokens",
        "test answer",
        85,
        "Good job!",
        llm_feedback="LLM feedback",
        llm_score=90,
    )
    assert result is mock_grade
    assert result.score == 85


def test_orchestrator_increments_weakness_on_low_score():
    """Verify orchestrator increments weakness when score < 70."""
    from terrance_teacher.core.examiner import Examiner
    from terrance_teacher.core.models import Grade
    
    mock_examiner = Mock(spec=Examiner)
    mock_grade = Grade(score=35, feedback="Needs improvement")
    mock_examiner.grade_answer.return_value = (mock_grade, None)
    
    mock_repo = Mock()
    
    orchestrator = TeacherOrchestrator(memory_repo=mock_repo)
    orchestrator._examiner = mock_examiner
    
    orchestrator.grade_answer("tokens", "poor answer")
    
    mock_repo.increment_weakness.assert_called_once_with("tokens")
