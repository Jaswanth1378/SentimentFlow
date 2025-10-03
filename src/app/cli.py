import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
from src.app.dag import SelfHealingDAG
from src.app.config import Config
import sys

app = typer.Typer()
console = Console()

def user_input_callback(question: str) -> str:
    return typer.prompt(f"\n[yellow]{question}[/yellow]")

@app.command()
def run(
    model_path: str = typer.Option(
        str(Config.CHECKPOINTS_DIR / "model"),
        "--model-path",
        "-m",
        help="Path to the trained model"
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--non-interactive",
        "-i/-n",
        help="Enable interactive fallback mode"
    ),
    temperature: float = typer.Option(
        1.0,
        "--temperature",
        "-t",
        help="Temperature for probability calibration"
    )
):
    console.print(Panel.fit(
        "[bold cyan]Self-Healing Classification System[/bold cyan]\n"
        "Using LangGraph DAG with LoRA-tuned DistilBERT",
        border_style="cyan"
    ))
    
    if not Path(model_path).exists():
        console.print(f"[red]Error: Model not found at {model_path}[/red]")
        console.print("[yellow]Please train the model first using: make train[/yellow]")
        raise typer.Exit(1)
    
    console.print(f"\n[green]Loading model from: {model_path}[/green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Initializing DAG pipeline...", total=None)
        dag = SelfHealingDAG(
            model_path=model_path,
            user_input_callback=user_input_callback if interactive else None,
            interactive=interactive
        )
        dag.set_temperature(temperature)
        progress.update(task, completed=True)
    
    console.print(f"[green]✓ Model loaded successfully![/green]")
    console.print(f"[dim]Temperature: {temperature} | Interactive: {interactive}[/dim]\n")
    
    console.print("[bold]Enter text to classify (or 'quit' to exit):[/bold]\n")
    
    while True:
        try:
            text = typer.prompt("\n> ", default="")
            
            if text.lower() in ['quit', 'exit', 'q']:
                console.print("\n[cyan]Thank you for using the Self-Healing Classifier![/cyan]")
                break
            
            if not text.strip():
                continue
            
            console.print(f"\n[dim]Processing...[/dim]")
            result = dag.run(text)
            
            console.print(f"\n[bold]Inference Results:[/bold]")
            console.print(f"  Predicted Label: [yellow]{result['label']}[/yellow]")
            console.print(f"  Confidence: [cyan]{result['confidence']:.1%}[/cyan]")
            console.print(f"  Status: [{'green' if result['status'] == 'HIGH' else 'yellow' if result['status'] == 'MEDIUM' else 'red'}]{result['status']}[/]")
            
            if result.get('fallback_activated'):
                console.print(f"\n[bold]Fallback Activated:[/bold]")
                console.print(f"  Strategy: [magenta]{result.get('fallback_strategy', 'N/A')}[/magenta]")
                
                if result.get('backup_model'):
                    backup = result['backup_model']
                    console.print(f"  Backup Model Prediction: [yellow]{backup['label']}[/yellow] ({backup['confidence']:.1%})")
            
            console.print(f"\n[bold green]Final Decision:[/bold green] {result['final_label']} [dim](via {result['decision_via']})[/dim]")
            console.print(f"[dim]Request ID: {result['request_id']}[/dim]")
        
        except KeyboardInterrupt:
            console.print("\n\n[cyan]Interrupted. Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")

@app.command()
def logs(
    lines: int = typer.Option(10, "--lines", "-n", help="Number of log lines to show"),
    json_format: bool = typer.Option(False, "--json", "-j", help="Show JSON logs")
):
    log_file = Config.LOG_JSONL_FILE if json_format else Config.LOG_FILE
    
    if not log_file.exists():
        console.print(f"[yellow]No logs found at {log_file}[/yellow]")
        return
    
    console.print(f"\n[bold cyan]Last {lines} log entries:[/bold cyan]\n")
    
    with open(log_file, 'r') as f:
        log_lines = f.readlines()
        for line in log_lines[-lines:]:
            console.print(line.strip())

if __name__ == "__main__":
    app()
