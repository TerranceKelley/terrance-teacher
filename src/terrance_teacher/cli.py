import sys
import typer
from terrance_teacher.core.orchestrator import TeacherOrchestrator
from terrance_teacher.memory import MemoryRepository

app = typer.Typer(
    help="Terrance Teacher — an agentic AI-powered personal teacher.",
    no_args_is_help=False,
    chain=True,  # Allow chaining commands
)


def _get_topic_from_argv() -> str:
    """Extract topic from sys.argv, handling both 'teach' command and script execution."""
    # When installed as command: sys.argv = ['teach', 'tokens', ...]
    # When run as script: sys.argv = ['/path/to/cli.py', 'tokens', ...]
    if len(sys.argv) < 2:
        raise typer.BadParameter("Missing topic. Example: teach tokens")
    
    first_arg = sys.argv[1]
    if first_arg == "answer":
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


def cli_main():
    """Main entry point that handles both command and topic modes."""
    # Check if first arg is a known subcommand
    if len(sys.argv) > 1 and sys.argv[1] in ("answer", "status"):
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