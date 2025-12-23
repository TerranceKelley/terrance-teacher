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

