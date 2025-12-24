import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

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
    assert "Next:" in output
    # LLM Feedback section may or may not appear (depends on Ollama availability)


def test_next_command_output_format():
    """Verify 'teach next' command outputs correct format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            response = runner.invoke(app, ["next"])
            
            assert response.exit_code == 0
            output = response.stdout
            assert "Next recommended topic:" in output
            assert "tokens" in output  # Default when no attempts
        finally:
            os.chdir(original_cwd)


def test_status_command_empty():
    """Status command shows empty state when no attempts exist."""
    # Use temporary directory for test database
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            response = runner.invoke(app, ["status"])
            
            assert response.exit_code == 0
            output = response.stdout
            assert "=== Learning Status ===" in output
            assert "Total attempts: 0" in output
            assert "Average score: 0.0" in output
        finally:
            os.chdir(original_cwd)


def test_status_command_with_attempts():
    """Status command shows correct stats with attempts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            # Create some attempts
            runner.invoke(app, ["answer", "tokens", "Context window limits cause truncation."])
            runner.invoke(app, ["answer", "tokens", "I don't know much about tokens."])
            
            response = runner.invoke(app, ["status"])
            
            assert response.exit_code == 0
            output = response.stdout
            assert "=== Learning Status ===" in output
            assert "Total attempts: 2" in output
            assert "Average score:" in output
        finally:
            os.chdir(original_cwd)


def test_status_command_weakest_topics():
    """Status command displays weakest topics correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            # Create low-score attempts to trigger weakness tracking
            runner.invoke(app, ["answer", "tokens", "I don't know."])  # Score < 70
            runner.invoke(app, ["answer", "rag", "I don't know."])  # Score < 70
            
            response = runner.invoke(app, ["status"])
            
            assert response.exit_code == 0
            output = response.stdout
            assert "Weakest topics" in output or "weak topics" in output.lower()
        finally:
            os.chdir(original_cwd)


def test_answer_persists_to_database():
    """Verify CLI answer command persists data to database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            # Answer a question
            runner.invoke(app, ["answer", "tokens", "Context window limits cause truncation."])
            
            # Check that status shows the attempt
            response = runner.invoke(app, ["status"])
            
            assert response.exit_code == 0
            output = response.stdout
            assert "Total attempts: 1" in output
        finally:
            os.chdir(original_cwd)


def test_exam_command_grades_and_prints_exam_header():
    """Verify exam command shows exam header and grades answer."""
    response = runner.invoke(
        app,
        [
            "exam",
            "tokens",
            "--answer",
            "context window limit and truncation",
        ],
    )
    
    assert response.exit_code == 0
    output = response.stdout
    assert "=== Exam: tokens ===" in output
    assert "Q:" in output
    assert "Task:" in output
    assert "=== Grade ===" in output
    assert "Score:" in output
    assert "Next:" in output


def test_exam_command_reveal_flag_prints_lesson_reveal():
    """Verify exam command with --reveal flag shows lesson after grading."""
    response = runner.invoke(
        app,
        [
            "exam",
            "tokens",
            "--answer",
            "context window causes truncation",
            "--reveal",
        ],
    )
    
    assert response.exit_code == 0
    output = response.stdout
    assert "=== Exam: tokens ===" in output
    assert "=== Grade ===" in output
    assert "=== Lesson Reveal ===" in output
    assert "Tokens are the fundamental units" in output or "explanation" in output.lower()
    assert "Example" in output or "example" in output.lower()


def test_exam_is_registered_subcommand():
    """Verify exam is treated as a subcommand, not a topic."""
    # Test that exam --help works (proves it's a registered command)
    response = runner.invoke(app, ["exam", "--help"])
    
    assert response.exit_code == 0
    # Should show exam command help, not treat "exam" as a topic
    assert "Topic to be tested on" in response.stdout or "exam" in response.stdout.lower()
