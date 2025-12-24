import sys
import typer
from terrance_teacher.core.orchestrator import TeacherOrchestrator
from terrance_teacher.memory import MemoryRepository

app = typer.Typer(
    help="Terrance Teacher — an agentic AI-powered personal teacher.",
    no_args_is_help=False,
    chain=False,  # Disable chaining to allow proper option parsing
)


def _get_topic_from_argv() -> str:
    """Extract topic from sys.argv, handling both 'teach' command and script execution."""
    # When installed as command: sys.argv = ['teach', 'tokens', ...]
    # When run as script: sys.argv = ['/path/to/cli.py', 'tokens', ...]
    if len(sys.argv) < 2:
        raise typer.BadParameter("Missing topic. Example: teach tokens")
    
    first_arg = sys.argv[1]
    if first_arg in ("answer", "status", "next", "history", "topic", "exam"):
        raise typer.BadParameter("Missing topic. Example: teach tokens")
    
    return first_arg


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Default usage: teach <topic> | Subcommands: teach answer <topic> '<answer>'"""
    if ctx.invoked_subcommand is not None:
        return
    
    topic = _get_topic_from_argv()
    memory_repo = MemoryRepository()
    orch = TeacherOrchestrator(memory_repo=memory_repo)
    lesson = orch.generate_lesson(topic)

    typer.echo("\n=== Lesson ===\n")
    typer.echo(lesson.explanation)

    typer.echo("\n=== Example ===\n")
    typer.echo(lesson.example)

    typer.echo("\n=== Check for Understanding ===\n")
    typer.echo(f"Q: {lesson.question}")

    typer.echo("\n=== Hands-on Task ===\n")
    typer.echo(f"Task: {lesson.task}\n")


@app.command()
def answer(
    topic: str = typer.Argument(..., help="Topic being answered (e.g., 'tokens')"),
    answer: str = typer.Argument(..., help="Your answer (wrap in quotes)"),
) -> None:
    memory_repo = MemoryRepository()
    orch = TeacherOrchestrator(memory_repo=memory_repo)
    grade = orch.grade_answer(topic, answer)

    typer.echo("\n=== Grade ===\n")
    typer.echo(f"Score: {grade.score}/100")
    typer.echo(f"Feedback: {grade.feedback}\n")
    
    # Display LLM feedback if available
    llm_feedback = memory_repo.get_last_attempt_llm_feedback()
    if llm_feedback:
        typer.echo("=== LLM Feedback ===\n")
        typer.echo(f"{llm_feedback}\n")
    
    next_topic = orch.recommend_next_topic()
    typer.echo(f"Next: {next_topic}\n")


@app.command()
def status() -> None:
    """Display learning progress statistics."""
    memory_repo = MemoryRepository()
    summary = memory_repo.get_status_summary()
    
    typer.echo("\n=== Learning Status ===\n")
    typer.echo(f"Total attempts: {summary['total_attempts']}")
    typer.echo(f"Average score: {summary['average_score']:.1f}")
    
    if summary['weakest_topics']:
        typer.echo("\nWeakest topics (top 5):")
        for topic, count in summary['weakest_topics']:
            typer.echo(f"  - {topic}: {count}")
    else:
        typer.echo("\nNo weak topics tracked yet.")
    
    typer.echo()


@app.command()
def next() -> None:
    """Display the next recommended topic."""
    memory_repo = MemoryRepository()
    orch = TeacherOrchestrator(memory_repo=memory_repo)
    topic = orch.recommend_next_topic()
    typer.echo(f"Next recommended topic: {topic}\n")


@app.command()
def history(limit: int = typer.Option(10, "--limit", "-n")) -> None:
    """Display recent lesson attempt history."""
    memory_repo = MemoryRepository()
    history_list = memory_repo.get_history(limit=limit)
    
    typer.echo(f"\n=== History (last {limit}) ===\n")
    
    if not history_list:
        typer.echo("No attempts yet.\n")
        return
    
    for i, entry in enumerate(history_list, 1):
        llm_indicator = "yes" if entry["has_llm_feedback"] else "no"
        typer.echo(
            f"{i}. {entry['created_at']} | {entry['topic']} | "
            f"{entry['score']}/100 | llm:{llm_indicator}"
        )
    
    typer.echo()


@app.command()
def topic(topic_name: str = typer.Argument(..., help="Topic to summarize")) -> None:
    """Display summary statistics for a specific topic."""
    memory_repo = MemoryRepository()
    summary = memory_repo.get_topic_summary(topic_name)
    
    typer.echo(f"\n=== Topic Summary: {summary['topic']} ===\n")
    typer.echo(f"Attempts: {summary['total_attempts']}")
    typer.echo(f"Average score: {summary['average_score']:.1f}")
    
    last_attempt = summary['last_attempt_at'] if summary['last_attempt_at'] else "None"
    last_score = summary['last_score'] if summary['last_score'] is not None else "None"
    
    typer.echo(f"Last attempt: {last_attempt}")
    typer.echo(f"Last score: {last_score}")
    typer.echo(f"Weakness count: {summary['weakness_count']}")
    
    llm_rate_percent = int(summary['llm_feedback_rate'] * 100)
    typer.echo(f"LLM feedback rate: {llm_rate_percent}%\n")


@app.command()
def exam(
    topic: str = typer.Argument(..., help="Topic to be tested on"),
    answer: str = typer.Option(
        "",
        "--answer",
        "-a",
        help="Answer (non-interactive mode)"
    ),
    reveal: bool = typer.Option(
        False,
        "--reveal",
        "-r",
        help="Show full lesson after grading"
    ),
) -> None:
    """Take an exam on a topic: see question + task, answer, get graded, optionally see lesson."""
    memory_repo = MemoryRepository()
    orch = TeacherOrchestrator(memory_repo=memory_repo)
    lesson = orch.generate_lesson(topic)
    
    # Display exam header with question and task only
    typer.echo(f"\n=== Exam: {topic} ===\n")
    typer.echo(f"Q: {lesson.question}")
    typer.echo(f"\nTask: {lesson.task}\n")
    
    # Collect answer
    if not answer:
        answer = typer.prompt("Your answer")
    # Otherwise use provided answer (non-interactive mode)
    
    # Grade using existing pipeline
    grade = orch.grade_answer(topic, answer)
    
    typer.echo("\n=== Grade ===\n")
    typer.echo(f"Score: {grade.score}/100")
    typer.echo(f"Feedback: {grade.feedback}\n")
    
    # Display LLM feedback if available
    llm_feedback = memory_repo.get_last_attempt_llm_feedback()
    if llm_feedback:
        typer.echo("=== LLM Feedback ===\n")
        typer.echo(f"{llm_feedback}\n")
    
    next_topic = orch.recommend_next_topic()
    typer.echo(f"Next: {next_topic}\n")
    
    # Optional lesson reveal
    if reveal:
        typer.echo("=== Lesson Reveal ===\n")
        typer.echo(lesson.explanation)
        typer.echo("\n=== Example ===\n")
        typer.echo(lesson.example)
        typer.echo()


def cli_main():
    """Main entry point that handles both command and topic modes."""
    # Check if first arg is a known subcommand
    if len(sys.argv) > 1 and sys.argv[1] in ("answer", "status", "next", "history", "topic", "exam"):
        # Let Typer handle the subcommand
        app()
    elif len(sys.argv) > 1:
        # First arg is a topic, handle it directly
        topic = sys.argv[1]
        memory_repo = MemoryRepository()
        orch = TeacherOrchestrator(memory_repo=memory_repo)
        lesson = orch.generate_lesson(topic)
        
        typer.echo("\n=== Lesson ===\n")
        typer.echo(lesson.explanation)
        typer.echo("\n=== Example ===\n")
        typer.echo(lesson.example)
        typer.echo("\n=== Check for Understanding ===\n")
        typer.echo(f"Q: {lesson.question}")
        typer.echo("\n=== Hands-on Task ===\n")
        typer.echo(f"Task: {lesson.task}\n")
    else:
        # No args, let Typer show help or error
        app()


if __name__ == "__main__":
    cli_main()