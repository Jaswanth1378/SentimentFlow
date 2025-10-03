#!/usr/bin/env python
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from src.app.dag import SelfHealingDAG
from src.app.config import Config
from rich.console import Console

console = Console()

console.print("\n[bold cyan]Self-Healing Classifier - Fallback Demo[/bold cyan]\n")
console.print("[dim]Demonstrating confidence-based fallback with zero-shot backup model[/dim]\n")

dag = SelfHealingDAG(
    model_path=str(Config.CHECKPOINTS_DIR / "model"),
    interactive=False,
    device="cpu"
)

dag.confidence_node.threshold_accept = 0.99
dag.confidence_node.threshold_clarify = 0.95

console.print(f"[yellow]Adjusted thresholds for demo:[/yellow]")
console.print(f"  Accept: ≥{dag.confidence_node.threshold_accept:.0%}")
console.print(f"  Clarify: ≥{dag.confidence_node.threshold_clarify:.0%}")
console.print(f"  Escalate: <{dag.confidence_node.threshold_clarify:.0%}\n")

test_cases = [
    "This movie was absolutely perfect!",
    "The film had some good moments.",
    "It was an okay movie, not great.",
]

for i, text in enumerate(test_cases, 1):
    console.print(f"[bold]Test {i}:[/bold] [yellow]{text}[/yellow]")
    
    result = dag.run(text)
    
    console.print(f"  Model Prediction: {result['label']} ({result['confidence']:.1%})")
    console.print(f"  Status: [{'green' if result['status'] == 'HIGH' else 'yellow' if result['status'] == 'MEDIUM' else 'red'}]{result['status']}[/]")
    
    if result.get('fallback_activated'):
        console.print(f"  [magenta]✓ Fallback Activated[/magenta]")
        console.print(f"  Strategy: {result.get('fallback_strategy')}")
        
        if result.get('backup_model'):
            backup = result['backup_model']
            console.print(f"  Backup Model: {backup['label']} ({backup['confidence']:.1%})")
        
        console.print(f"  [bold green]Final Decision: {result['final_label']}[/bold green] (via {result['decision_via']})")
    else:
        console.print(f"  [green]✓ Direct Accept[/green]")
        console.print(f"  [bold]Final: {result['final_label']}[/bold]")
    
    console.print()

console.print("[bold green]Demo completed![/bold green]\n")
console.print(f"[dim]View logs: tail -f {Config.LOG_JSONL_FILE}[/dim]")
