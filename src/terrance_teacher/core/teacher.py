import json
import re
from terrance_teacher.core.models import Lesson
from terrance_teacher.llm import OllamaClient
from pydantic import ValidationError


class Teacher:
    """Handles lesson generation: hardcoded lessons, Ollama generation, and fallback scaffolding."""
    
    def __init__(self, memory_repo=None, ollama_client=None):
        self.memory_repo = memory_repo
        self.ollama_client = ollama_client
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
        
        # Priority 1: Return hardcoded lesson for "tokens" (unchanged)
        if topic_lower in self._lessons:
            return self._lessons[topic_lower]
        
        # Priority 2: Attempt Ollama generation
        ollama_client = self.ollama_client
        if ollama_client is None:
            ollama_client = OllamaClient()
        
        prompt = (
            f"Generate a lesson on '{topic}' for experienced engineers learning LLM systems.\n\n"
            "Output ONLY valid JSON with these exact fields:\n"
            "- topic\n"
            "- explanation (2-3 paragraphs)\n"
            "- example (concrete example or scenario)\n"
            "- question (check-for-understanding)\n"
            "- task (hands-on task)\n\n"
            "No markdown. No extra text. JSON only."
        )
        
        response = ollama_client.generate(prompt)
        
        if response is not None:
            try:
                # Extract JSON from response (handle code fences if present)
                json_text = response.strip()
                # Remove markdown code fences if present
                json_text = re.sub(r"^```(?:json)?\s*\n?", "", json_text, flags=re.MULTILINE)
                json_text = re.sub(r"\n?```\s*$", "", json_text, flags=re.MULTILINE)
                json_text = json_text.strip()
                
                # Parse and validate
                data = json.loads(json_text)
                lesson = Lesson.model_validate(data)
                # Ensure topic matches requested topic
                lesson.topic = topic
                return lesson
            except (json.JSONDecodeError, ValidationError, KeyError):
                # Fall through to fallback
                pass
        
        # Priority 3: Fallback to scaffold lesson with adaptive difficulty
        weakness_count = 0
        if self.memory_repo is not None:
            weakness_count = self.memory_repo.get_weakness_count(topic)
        
        return self._build_fallback_lesson(topic, weakness_count)
    
    def _build_fallback_lesson(self, topic: str, weakness_count: int) -> Lesson:
        """Build a fallback lesson with adaptive difficulty based on weakness count.
        
        Tiers:
        - Beginner (weakness_count <= 1): Short explanation, simple example, straightforward question, single-step task
        - Intermediate (weakness_count in [2, 3]): Medium explanation, example with tradeoff, reasoning question, 2-3 step task
        - Advanced (weakness_count >= 4): Deep explanation, example with failure modes, systems thinking question, multi-step task
        """
        if weakness_count <= 1:
            # Beginner tier
            return Lesson(
                topic=topic,
                explanation=(
                    f"[Beginner] Lesson on '{topic}' is not yet available. This is a placeholder lesson. "
                    f"'{topic}' is an important concept in LLM systems. "
                    "In future versions, lessons will be generated dynamically using LLMs. "
                    "For now, this provides a basic introduction to get you started."
                ),
                example=(
                    f"A simple example of '{topic}' would demonstrate the core concept. "
                    "Example content will be generated when this lesson is fully implemented."
                ),
                question=(
                    f"What is the basic definition of '{topic}'? "
                    "This question will be customized in future versions."
                ),
                task=(
                    f"Research '{topic}' independently and write a brief summary (1-2 paragraphs). "
                    "In future versions, you'll receive structured tasks and automated grading."
                ),
            )
        elif weakness_count in [2, 3]:
            # Intermediate tier
            return Lesson(
                topic=topic,
                explanation=(
                    f"[Intermediate] Lesson on '{topic}' is not yet available. This is a placeholder lesson. "
                    f"'{topic}' is a nuanced concept in LLM systems that requires understanding of tradeoffs and edge cases. "
                    "In future versions, lessons will be generated dynamically using LLMs. "
                    "This intermediate-level content explores the concept more deeply, including considerations "
                    "of when and how to apply it effectively, as well as potential limitations."
                ),
                example=(
                    f"An intermediate example of '{topic}' would illustrate both the benefits and tradeoffs. "
                    "For instance, you might see how it works in ideal conditions, but also encounter "
                    "edge cases where it fails or requires careful handling. "
                    "Example content will be generated when this lesson is fully implemented."
                ),
                question=(
                    f"How does '{topic}' interact with other LLM concepts, and what are the key tradeoffs? "
                    "This question requires reasoning about relationships and implications."
                ),
                task=(
                    f"Research '{topic}' and create a short analysis: (1) Define the concept, "
                    "(2) Identify one key tradeoff or limitation, (3) Propose a scenario where it would be most effective. "
                    "In future versions, you'll receive structured tasks and automated grading."
                ),
            )
        else:
            # Advanced tier (weakness_count >= 4)
            return Lesson(
                topic=topic,
                explanation=(
                    f"[Advanced] Lesson on '{topic}' is not yet available. This is a placeholder lesson. "
                    f"'{topic}' is a complex concept in LLM systems that requires systems thinking and "
                    "understanding of failure modes, operational constraints, and production considerations. "
                    "In future versions, lessons will be generated dynamically using LLMs. "
                    "This advanced-level content delves into the deeper implications, including how it fits "
                    "into larger system architectures, what can go wrong in production, and how to design "
                    "robust implementations that account for real-world constraints and edge cases."
                ),
                example=(
                    f"An advanced example of '{topic}' would demonstrate failure modes and operational considerations. "
                    "This might include: how it behaves under load, what happens when assumptions break down, "
                    "how to monitor and detect issues, and how to design for resilience. "
                    "Example content will be generated when this lesson is fully implemented."
                ),
                question=(
                    f"How would you design a production system that uses '{topic}' while accounting for "
                    "failure modes, scalability constraints, and operational observability? "
                    "This question requires systems thinking about real-world deployment."
                ),
                task=(
                    f"Design and document a production-ready approach to '{topic}': "
                    "(1) Design the architecture and data flow, (2) Identify potential failure modes and mitigation strategies, "
                    "(3) Implement a proof-of-concept or detailed design, (4) Define metrics and observability requirements. "
                    "In future versions, you'll receive structured tasks and automated grading."
                ),
            )

