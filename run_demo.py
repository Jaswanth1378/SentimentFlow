#!/usr/bin/env python
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from src.app.dag import SelfHealingDAG
from src.app.config import Config
from rich.console import Console
from rich.table import Table

console = Console()

console.print("\n[bold cyan]Self-Healing Classification System - Demo[/bold cyan]\n")

dag = SelfHealingDAG(
    model_path=str(Config.CHECKPOINTS_DIR / "model"),
    interactive=False,
    device="cpu"
)

test_cases = [
    ("This movie was absolutely amazing! Best film I've seen all year!", "High confidence positive"),
    ("The movie was okay, nothing special.", "Medium confidence"),
    ("I'm not sure what to think about this film.", "Low confidence ambiguous"),
    ("Worst movie ever! Complete waste of time!", "High confidence negative"),
]

console.print("[bold]Running test cases:[/bold]\n")

results_table = Table(show_header=True, header_style="bold magenta")
results_table.add_column("Input Text", width=50)
results_table.add_column("Predicted", width=12)
results_table.add_column("Confidence", width=12)
results_table.add_column("Decision Via", width=20)

for text, description in test_cases:
    console.print(f"[dim]{description}:[/dim]")
    console.print(f"  Input: [yellow]{text}[/yellow]")
    
    result = dag.run(text)
    
    console.print(f"  → Prediction: [cyan]{result['final_label']}[/cyan] ({result['confidence']:.1%})")
    console.print(f"  → Decision via: [green]{result['decision_via']}[/green]")
    
    if result.get('fallback_activated'):
        console.print(f"  → Fallback: [magenta]{result.get('fallback_strategy', 'N/A')}[/magenta]")
    
    console.print()
    
    results_table.add_row(
        text[:47] + "..." if len(text) > 50 else text,
        result['final_label'],
        f"{result['confidence']:.1%}",
        result['decision_via']
    )

console.print(results_table)

console.print("\n[bold green]✓ Demo completed successfully![/bold green]")
console.print(f"\n[dim]Logs saved to: {Config.LOG_JSONL_FILE}[/dim]")
