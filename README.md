# Terrance Teacher

Terrance Teacher is an **agentic AI-powered personal teacher** designed to train experienced DevOps and platform engineers in **LLM engineering, agentic AI systems, and AI infrastructure**.

Unlike traditional tutorials or chatbots, this system:
- Assesses existing knowledge
- Teaches concepts incrementally
- Assigns hands-on tasks
- Grades responses and code
- Tracks weaknesses over time
- Adapts the curriculum automatically

This project is both a **learning engine** and a **demonstration of real-world agentic AI architecture**.

---

## Why This Exists

Most AI learning resources are:
- Passive (videos, blogs)
- Beginner-focused
- Not system-oriented
- Not verifiable

Terrance Teacher is built to:
- Teach **how AI systems actually fail**
- Emphasize production tradeoffs
- Require validation before advancing
- Mirror how senior engineers learn on the job

---

## Core Capabilities (v1)

- CLI-based interactive teaching (`teach "topic"`)
- Structured lessons (explain → example → test)
- Automatic grading and feedback
- Persistent learner memory (SQLite)
- Adaptive topic progression
- Local-first LLM support (Ollama)

---

## Architecture (v1)

```text
CLI
  ↓
Teacher Orchestrator
  ↓
LLM Adapter (Ollama / OpenAI)
  ↓
Lesson + Quiz + Task
  ↓
Grading Rubric
  ↓
Memory Store (SQLite)
```

---

## Technologies

- Python
- Typer (CLI)
- Pydantic (schemas)
- SQLite (learner memory)
- Ollama (local LLM inference)
- Rich (console output)

---

## Initial Curriculum

- Tokens & context windows
- Temperature, determinism, and sampling
- Prompt architecture & guardrails
- Retrieval-Augmented Generation (RAG)
- Hallucination causes and mitigation
- Agent loops and planning strategies

---

## Roadmap

- Multi-agent teaching system
- Code evaluation and critique
- Curriculum graph with prerequisites
- Cloud deployment (AWS)
- Cost-aware model selection
- Certification-aligned learning paths

---

## Status

🚧 Active development  
This project evolves alongside the author’s transition into AI engineering and platform leadership.

---

## License

MIT
