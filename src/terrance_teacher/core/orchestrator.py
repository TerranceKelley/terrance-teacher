from terrance_teacher.core.models import Lesson, Grade


class TeacherOrchestrator:
    def __init__(self, memory_repo=None):
        self.memory_repo = memory_repo
        self._lessons = {
            "tokens": Lesson(
                topic="tokens",
                explanation=(
                    "Tokens are the fundamental units that LLMs process. A token is roughly 0.75 words "
                    "on average, but can be as short as a character or as long as a word. When you send "
                    "text to an LLM, it's first tokenized (split into tokens) using a tokenizer specific "
                    "to that model. Understanding tokens is critical because:\n"
                    "- Context windows are measured in tokens, not words\n"
                    "- Pricing is often per token (input + output)\n"
                    "- Tokenization affects prompt engineering (some words become multiple tokens)\n"
                    "- Different models have different tokenizers (GPT-4 vs Claude vs Llama)"
                ),
                example=(
                    "Consider the phrase 'Hello, world!'\n"
                    "- GPT-4 tokenizer: ['Hello', ',', ' world', '!'] → 4 tokens\n"
                    "- Some tokenizers might split 'Hello' into ['Hel', 'lo'] → 5 tokens\n"
                    "- Token counts vary by model and language\n\n"
                    "In practice, you can estimate: 1 token ≈ 4 characters for English text, "
                    "but always use the model's tokenizer for accurate counts."
                ),
                question=(
                    "Why does the same text produce different token counts across models, "
                    "and how does this impact prompt engineering strategies?"
                ),
                task=(
                    "Write a Python function that estimates token count for a given string using "
                    "the rule of thumb (1 token ≈ 4 characters). Then, use the tiktoken library "
                    "to get the actual token count for GPT-4. Compare the results for the text: "
                    "'The quick brown fox jumps over the lazy dog.'"
                ),
            ),
        }

    def generate_lesson(self, topic: str) -> Lesson:
        topic_lower = topic.lower()
        if topic_lower in self._lessons:
            return self._lessons[topic_lower]
        
        # Fallback lesson for unknown topics
        return Lesson(
            topic=topic,
            explanation=(
                f"Lesson on '{topic}' is not yet available. This is a placeholder lesson. "
                "In future versions, lessons will be generated dynamically using LLMs."
            ),
            example=(
                "Example content will be generated when this lesson is fully implemented."
            ),
            question=(
                f"What would you like to learn about '{topic}'? "
                "This question will be customized in future versions."
            ),
            task=(
                f"Research '{topic}' independently and document your findings. "
                "In future versions, you'll receive structured tasks and automated grading."
            ),
        )

    def grade_answer(self, topic: str, answer: str) -> Grade:
        topic_lower = topic.lower()
        answer_lower = answer.lower()
        
        if topic_lower in ("tokens", "tokenization"):
            keywords = [
                "truncation",
                "context limit",
                "constraint",
                "context window",
                "token limit",
            ]
            
            found_keywords = [kw for kw in keywords if kw in answer_lower]
            num_found = len(found_keywords)
            
            if num_found >= 2:
                score = 85
                feedback = (
                    f"Excellent! You identified multiple key concepts: {', '.join(found_keywords)}. "
                    "You understand that different tokenizers affect context windows and can cause truncation, "
                    "which directly impacts prompt engineering strategies."
                )
            elif num_found == 1:
                score = 65
                feedback = (
                    f"Good start! You mentioned '{found_keywords[0]}', which is relevant. "
                    "To improve, consider discussing how tokenization differences across models affect "
                    "context window limits and can lead to truncation or constraint loss."
                )
            else:
                score = 35
                feedback = (
                    "Your answer touches on tokenization but misses key impacts. Consider discussing: "
                    "how different tokenizers affect context window limits, the risk of truncation "
                    "when text exceeds limits, and how this constrains prompt engineering strategies."
                )
            
            grade = Grade(score=score, feedback=feedback)
        else:
            # Unknown topic fallback
            grade = Grade(
                score=50,
                feedback=(
                    f"Grading for '{topic}' is not yet implemented. This is a placeholder grade. "
                    "In future versions, answers will be evaluated using LLM-based grading."
                ),
            )
        
        # Persist attempt if memory repository is available
        if self.memory_repo:
            self.memory_repo.save_attempt(topic, answer, grade.score, grade.feedback)
            if grade.score < 70:
                self.memory_repo.increment_weakness(topic)
        
        return grade

