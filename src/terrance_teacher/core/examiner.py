import json
import re
from terrance_teacher.core.models import Grade, LlmGrade
from terrance_teacher.llm import OllamaClient
from pydantic import ValidationError


class Examiner:
    """Handles answer grading: deterministic grading and optional Ollama qualitative feedback."""
    
    def __init__(self, ollama_client=None):
        self.ollama_client = ollama_client
    
    def grade_answer(self, topic: str, answer: str) -> tuple[Grade, LlmGrade | None]:
        """
        Grade an answer deterministically and optionally get LLM feedback.
        
        Returns:
            tuple[Grade, LlmGrade | None]: Deterministic grade and optional LLM grade
        """
        # Handle empty answer
        if not answer.strip():
            grade = Grade(
                score=10,
                feedback="Your answer is empty. Please provide a response to be graded.",
            )
        else:
            # Deterministic grading
            topic_lower = topic.lower()
            answer_lower = answer.lower()
            
            grade = self._grade_deterministic(topic_lower, answer_lower)
        
        # Attempt optional Ollama qualitative feedback (best-effort)
        llm_grade = None
        
        ollama_client = self.ollama_client
        if ollama_client is None:
            ollama_client = OllamaClient()
        
        grading_prompt = (
            f"Grade this answer on '{topic}' for an experienced engineer learning LLM systems.\n\n"
            f"Student answer: \"{answer}\"\n\n"
            f"Deterministic grade: {grade.score}/100\n"
            f"Deterministic feedback: {grade.feedback}\n\n"
            "Provide qualitative feedback as an experienced teacher would. Output ONLY valid JSON:\n"
            "- feedback: detailed qualitative feedback (2-3 sentences)\n"
            "- score: optional integer 0-100 (your assessment, not required)\n\n"
            "No markdown. JSON only."
        )
        
        ollama_response = ollama_client.generate(grading_prompt)
        
        if ollama_response is not None:
            try:
                # Extract JSON from response (handle code fences if present)
                json_text = ollama_response.strip()
                json_text = re.sub(r"^```(?:json)?\s*\n?", "", json_text, flags=re.MULTILINE)
                json_text = re.sub(r"\n?```\s*$", "", json_text, flags=re.MULTILINE)
                json_text = json_text.strip()
                
                # Parse and validate
                data = json.loads(json_text)
                llm_grade = LlmGrade.model_validate(data)
            except (json.JSONDecodeError, ValidationError, KeyError):
                # Fall through - llm_grade remains None
                pass
        
        return (grade, llm_grade)
    
    def _grade_deterministic(self, topic_lower: str, answer_lower: str) -> Grade:
        """Grade answer deterministically based on topic and keyword matching."""
        
        # Define keyword sets per topic
        topic_keywords = {
            "tokens": ["truncation", "context limit", "constraint", "context window", "token limit"],
            "tokenization": ["truncation", "context limit", "constraint", "context window", "token limit"],
            "temperature": ["temperature", "sampling", "randomness", "deterministic", "top-p", "nucleus", "entropy"],
            "prompting": ["system prompt", "instructions", "constraints", "few-shot", "zero-shot", "role", "persona"],
            "rag": ["retrieval", "embeddings", "vector database", "vector store", "context injection", "chunks", "chunking", "citation", "grounding"],
            "hallucinations": ["hallucination", "fabricate", "grounding", "verification", "confidence", "retrieval"],
            "agents": ["agent", "tool use", "tools", "planning", "loop", "iteration", "memory", "reflection"],
        }
        
        if topic_lower not in topic_keywords:
            # Unknown topic fallback
            return Grade(
                score=50,
                feedback=(
                    f"Grading for '{topic_lower}' is not yet implemented. This is a placeholder grade. "
                    "In future versions, answers will be evaluated using LLM-based grading."
                ),
            )
        
        keywords = topic_keywords[topic_lower]
        found_keywords = [kw for kw in keywords if kw in answer_lower]
        num_found = len(found_keywords)
        
        # Scoring bands: High (80-100), Medium (50-79), Low (20-49)
        if num_found >= 2:
            # High score: 2+ key concepts
            score = 85
            feedback = (
                f"Excellent! You identified multiple key concepts: {', '.join(found_keywords)}. "
                f"Your understanding of {topic_lower} demonstrates good grasp of the core principles."
            )
        elif num_found == 1:
            # Medium score: 1 key concept
            score = 65
            # Suggest missing concepts
            missing = [kw for kw in keywords[:3] if kw not in found_keywords]
            suggestions = ", ".join(missing[:2])
            feedback = (
                f"Good start! You mentioned '{found_keywords[0]}', which is relevant. "
                f"To improve, consider discussing: {suggestions}."
            )
        else:
            # Low score: no key concepts
            score = 35
            # Suggest key concepts
            suggestions = ", ".join(keywords[:2])
            feedback = (
                f"Your answer touches on {topic_lower} but misses key concepts. "
                f"Consider discussing: {suggestions}."
            )
        
        return Grade(score=score, feedback=feedback)

