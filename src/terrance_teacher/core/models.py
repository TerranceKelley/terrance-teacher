from pydantic import BaseModel


class Lesson(BaseModel):
    topic: str
    explanation: str
    example: str
    question: str
    task: str


class Grade(BaseModel):
    score: int
    feedback: str


class StatusSummary(BaseModel):
    total_attempts: int
    average_score: float
    weakest_topics: list[tuple[str, int]]


class LlmGrade(BaseModel):
    feedback: str
    score: int | None = None

