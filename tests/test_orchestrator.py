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
