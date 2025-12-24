from terrance_teacher.core.models import Lesson, Grade
from terrance_teacher.core.teacher import Teacher
from terrance_teacher.core.examiner import Examiner


class TeacherOrchestrator:
    """Coordinates Teacher (lesson generation) and Examiner (grading) components."""
    
    CURRICULUM = [
        "tokens",
        "temperature",
        "prompting",
        "rag",
        "hallucinations",
        "agents",
    ]
    
    def __init__(self, memory_repo=None, ollama_client=None):
        self.memory_repo = memory_repo
        self.ollama_client = ollama_client
        self._teacher = Teacher(memory_repo=memory_repo, ollama_client=ollama_client)
        self._examiner = Examiner(ollama_client=ollama_client)

    def generate_lesson(self, topic: str) -> Lesson:
        """Generate a lesson by delegating to Teacher."""
        return self._teacher.generate_lesson(topic)

    def grade_answer(self, topic: str, answer: str) -> Grade:
        """
        Grade an answer by delegating to Examiner, then persist results.
        
        Returns deterministic Grade (unchanged contract).
        """
        # Get grade and optional LLM feedback from Examiner
        grade, llm_grade = self._examiner.grade_answer(topic, answer)
        
        # Extract LLM feedback if available
        llm_feedback = None
        llm_score = None
        if llm_grade is not None:
            llm_feedback = llm_grade.feedback
            llm_score = llm_grade.score
        
        # Persist attempt if memory repository is available
        if self.memory_repo:
            self.memory_repo.save_attempt(
                topic,
                answer,
                grade.score,
                grade.feedback,
                llm_feedback=llm_feedback,
                llm_score=llm_score,
            )
            if grade.score < 70:
                self.memory_repo.increment_weakness(topic)
        
        return grade
    
    def recommend_next_topic(self) -> str:
        """Recommend the next topic based on weaknesses, attempts, or default."""
        # If no memory repository, return default
        if self.memory_repo is None:
            return "tokens"
        
        # Priority 1: Weaknesses
        weak_topic = self.memory_repo.get_top_weak_topic()
        if weak_topic is not None:
            return weak_topic
        
        # Priority 2: Next topic after last attempt
        last_topic = self.memory_repo.get_last_attempt_topic()
        if last_topic is not None:
            try:
                last_index = self.CURRICULUM.index(last_topic.lower())
                next_index = (last_index + 1) % len(self.CURRICULUM)
                return self.CURRICULUM[next_index]
            except ValueError:
                # Topic not in curriculum, return default
                return "tokens"
        
        # Priority 3: No attempts, return default
        return "tokens"

