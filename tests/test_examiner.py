from terrance_teacher.core.examiner import Examiner
from terrance_teacher.core.models import Grade


def test_tokens_high_score_multiple_concepts():
    """Verify tokens grading gives high score when multiple concepts mentioned."""
    examiner = Examiner(ollama_client=None)
    answer = "Context window limits can cause truncation when token limits are exceeded."
    
    grade, _ = examiner.grade_answer("tokens", answer)
    
    assert 80 <= grade.score <= 100
    assert len(grade.feedback) > 0
    assert "context window" in grade.feedback.lower() or "truncation" in grade.feedback.lower()


def test_temperature_scores_high_when_sampling_and_temperature():
    """Verify temperature grading gives high score with multiple concepts."""
    examiner = Examiner(ollama_client=None)
    answer = "Temperature controls randomness in sampling. Lower values make outputs more deterministic."
    
    grade, _ = examiner.grade_answer("temperature", answer)
    
    assert 80 <= grade.score <= 100
    assert len(grade.feedback) > 0
    assert "temperature" in grade.feedback.lower() or "sampling" in grade.feedback.lower() or "randomness" in grade.feedback.lower()


def test_prompting_scores_medium_with_one_concept():
    """Verify prompting grading gives medium score with one concept."""
    examiner = Examiner(ollama_client=None)
    answer = "System prompts help guide model behavior."
    
    grade, _ = examiner.grade_answer("prompting", answer)
    
    assert 50 <= grade.score <= 79
    assert len(grade.feedback) > 0
    assert "system prompt" in grade.feedback.lower() or "good start" in grade.feedback.lower()


def test_rag_scores_high_with_retrieval_and_embeddings():
    """Verify RAG grading gives high score with multiple concepts."""
    examiner = Examiner(ollama_client=None)
    answer = "Use embeddings in a vector store to retrieve relevant chunks and inject into context."
    
    grade, _ = examiner.grade_answer("rag", answer)
    
    assert 80 <= grade.score <= 100
    assert len(grade.feedback) > 0
    assert "retrieval" in grade.feedback.lower() or "embeddings" in grade.feedback.lower() or "vector" in grade.feedback.lower()


def test_hallucinations_scores_high_with_grounding_and_verification():
    """Verify hallucinations grading gives high score with multiple concepts."""
    examiner = Examiner(ollama_client=None)
    answer = "Hallucinations occur when models fabricate information. Use grounding and verification to reduce them."
    
    grade, _ = examiner.grade_answer("hallucinations", answer)
    
    assert 80 <= grade.score <= 100
    assert len(grade.feedback) > 0
    assert "hallucination" in grade.feedback.lower() or "grounding" in grade.feedback.lower() or "verification" in grade.feedback.lower()


def test_agents_scores_high_with_tools_and_planning():
    """Verify agents grading gives high score with multiple concepts."""
    examiner = Examiner(ollama_client=None)
    answer = "Agents plan steps, call tools, and iterate with memory."
    
    grade, _ = examiner.grade_answer("agents", answer)
    
    assert 80 <= grade.score <= 100
    assert len(grade.feedback) > 0
    assert "agent" in grade.feedback.lower() or "tool" in grade.feedback.lower() or "planning" in grade.feedback.lower()


def test_unknown_topic_neutral_score():
    """Verify unknown topic returns neutral score (50)."""
    examiner = Examiner(ollama_client=None)
    answer = "Some answer about an unknown topic."
    
    grade, _ = examiner.grade_answer("unknown_topic", answer)
    
    assert grade.score == 50
    assert len(grade.feedback) > 0
    assert "not yet implemented" in grade.feedback.lower() or "placeholder" in grade.feedback.lower()


def test_empty_answer_very_low_score():
    """Verify empty answer gets very low score (0-19)."""
    examiner = Examiner(ollama_client=None)
    answer = "   "
    
    grade, _ = examiner.grade_answer("tokens", answer)
    
    assert 0 <= grade.score <= 19
    assert len(grade.feedback) > 0
    assert "empty" in grade.feedback.lower()

