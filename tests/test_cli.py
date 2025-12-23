import sys

from typer.testing import CliRunner

from terrance_teacher.cli import app


runner = CliRunner()


def test_teach_tokens_outputs_sections(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["teach", "tokens"])

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    output = result.stdout
    assert "=== Lesson ===" in output
    assert "Tokens are the fundamental units that LLMs process" in output
    assert "=== Example ===" in output
    assert "=== Check for Understanding ===" in output
    assert "=== Hands-on Task ===" in output
    assert "Write a Python function that estimates token count" in output


def test_answer_tokens_grades_keywords():
    response = runner.invoke(
        app,
        [
            "answer",
            "tokens",
            "Context window constraints can cause truncation when hitting the token limit.",
        ],
    )

    assert response.exit_code == 0
    output = response.stdout
    assert "=== Grade ===" in output
    assert "Score: 85/100" in output
    assert "Feedback:" in output
    assert "key concepts" in output
