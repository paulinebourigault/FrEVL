"""
FrEVL Command Line Interface
Provides CLI commands for training, evaluation, and serving
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import yaml
import json

import torch
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint


console = Console()


def train(args: Optional[List[str]] = None):
    """Train FrEVL model"""
    parser = argparse.ArgumentParser(description="Train FrEVL model")
    parser.add_argument("--config", type=str, default="configs/train_vqa.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--dataset", type=str, default="vqa",
                       help="Dataset to train on")
    parser.add_argument("--data-root", type=str, default="./data",
                       help="Root directory for datasets")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--wandb-project", type=str, default=None,
                       help="Weights & Biases project name")
    parser.add_argument("--distributed", action="store_true",
                       help="Enable distributed training")
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Enable mixed precision training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    parsed_args = parser.parse_args(args)
    
    console.print(f"[bold green]Starting FrEVL training[/bold green]")
    console.print(f"Config: {parsed_args.config}")
    console.print(f"Dataset: {parsed_args.dataset}")
    
    # Load configuration
    if parsed_args.config:
        with open(parsed_args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override config with command line arguments
    if parsed_args.epochs:
        config.setdefault("training", {})["epochs"] = parsed_args.epochs
    if parsed_args.batch_size:
        config.setdefault("training", {})["batch_size"] = parsed_args.batch_size
    if parsed_args.learning_rate:
        config.setdefault("training", {})["learning_rate"] = parsed_args.learning_rate
    
    # Import and run training
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Initializing training...", total=None)
        
        from train import main as train_main
        
        # Convert config to namespace for compatibility
        import types
        args_namespace = types.SimpleNamespace(**parsed_args.__dict__)
        for key, value in config.items():
            if not hasattr(args_namespace, key):
                setattr(args_namespace, key, value)
        
        progress.update(task, description="Training in progress...")
        train_main(args_namespace)
    
    console.print("[bold green]✓ Training complete![/bold green]")


def evaluate(args: Optional[List[str]] = None):
    """Evaluate FrEVL model"""
    parser = argparse.ArgumentParser(description="Evaluate FrEVL model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint or model name")
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["vqa", "snli-ve", "coco", "all"],
                       help="Dataset to evaluate on")
    parser.add_argument("--data-root", type=str, default="./data",
                       help="Root directory for datasets")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for evaluation")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark")
    parser.add_argument("--save-predictions", action="store_true",
                       help="Save model predictions")
    parser.add_argument("--compute-attention", action="store_true",
                       help="Compute and save attention maps")
    
    parsed_args = parser.parse_args(args)
    
    console.print(f"[bold blue]Evaluating FrEVL model[/bold blue]")
    console.print(f"Model: {parsed_args.model}")
    console.print(f"Dataset: {parsed_args.dataset}")
    
    # Import and run evaluation
    from evaluate import main as evaluate_main
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running evaluation...", total=None)
        
        evaluate_main(parsed_args)
        
        progress.update(task, complete=True)
    
    # Display results
    results_file = Path(parsed_args.output_dir) / "evaluation_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create results table
        table = Table(title="Evaluation Results")
        table.add_column("Dataset", style="cyan")
        table.add_column("Accuracy", style="green")
        table.add_column("Precision", style="yellow")
        table.add_column("Recall", style="yellow")
        table.add_column("F1", style="magenta")
        
        for dataset, metrics in results.items():
            table.add_row(
                dataset,
                f"{metrics.get('accuracy', 0):.3f}",
                f"{metrics.get('precision', 0):.3f}",
                f"{metrics.get('recall', 0):.3f}",
                f"{metrics.get('f1', 0):.3f}"
            )
        
        console.print(table)
    
    console.print("[bold green]✓ Evaluation complete![/bold green]")


def demo(args: Optional[List[str]] = None):
    """Launch interactive demo"""
    parser = argparse.ArgumentParser(description="Launch FrEVL demo")
    parser.add_argument("--model", type=str, default="frevl-base",
                       help="Model checkpoint or name")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run demo on")
    parser.add_argument("--share", action="store_true",
                       help="Create public shareable link")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--examples-dir", type=str, default="./examples",
                       help="Directory with example images")
    
    parsed_args = parser.parse_args(args)
    
    console.print(f"[bold cyan]Launching FrEVL demo[/bold cyan]")
    console.print(f"Model: {parsed_args.model}")
    console.print(f"Port: {parsed_args.port}")
    
    # Import and run demo
    from demo import main as demo_main
    
    demo_main(parsed_args)


def serve(args: Optional[List[str]] = None):
    """Start API server"""
    parser = argparse.ArgumentParser(description="Start FrEVL API server")
    parser.add_argument("--model", type=str, default="frevl-base",
                       help="Model checkpoint or name")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to run server on")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of worker processes")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--cache", action="store_true",
                       help="Enable Redis caching")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload for development")
    
    parsed_args = parser.parse_args(args)
    
    console.print(f"[bold yellow]Starting FrEVL API server[/bold yellow]")
    console.print(f"Model: {parsed_args.model}")
    console.print(f"URL: http://{parsed_args.host}:{parsed_args.port}")
    
    # Start server using uvicorn
    import uvicorn
    
    uvicorn.run(
        "serve:app",
        host=parsed_args.host,
        port=parsed_args.port,
        workers=parsed_args.workers if not parsed_args.reload else 1,
        reload=parsed_args.reload
    )


def download(args: Optional[List[str]] = None):
    """Download models and datasets"""
    parser = argparse.ArgumentParser(description="Download FrEVL models and datasets")
    parser.add_argument("--type", type=str, choices=["model", "dataset", "all"],
                       default="all", help="What to download")
    parser.add_argument("--model", type=str, nargs="+", default=["frevl-base"],
                       help="Models to download")
    parser.add_argument("--dataset", type=str, nargs="+", default=["vqa"],
                       help="Datasets to download")
    parser.add_argument("--model-dir", type=str, default="./checkpoints",
                       help="Directory to save models")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Directory to save datasets")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download")
    
    parsed_args = parser.parse_args(args)
    
    console.print(f"[bold magenta]Downloading FrEVL resources[/bold magenta]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        if parsed_args.type in ["model", "all"]:
            task = progress.add_task("Downloading models...", total=None)
            
            from scripts.download_models import ModelDownloader
            downloader = ModelDownloader(parsed_args.model_dir)
            
            for model_name in parsed_args.model:
                progress.update(task, description=f"Downloading {model_name}...")
                downloader.download_model(model_name, force=parsed_args.force)
            
            progress.update(task, complete=True)
        
        if parsed_args.type in ["dataset", "all"]:
            task = progress.add_task("Downloading datasets...", total=None)
            
            from scripts.download_datasets import DatasetDownloader
            downloader = DatasetDownloader(parsed_args.data_dir)
            
            for dataset_name in parsed_args.dataset:
                progress.update(task, description=f"Downloading {dataset_name}...")
                downloader.download_all([dataset_name])
            
            progress.update(task, complete=True)
    
    console.print("[bold green]✓ Downloads complete![/bold green]")


def info(args: Optional[List[str]] = None):
    """Show FrEVL information"""
    parser = argparse.ArgumentParser(description="Show FrEVL information")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed information")
    
    parsed_args = parser.parse_args(args)
    
    import frevl
    
    # Basic info
    console.print(f"[bold]FrEVL v{frevl.__version__}[/bold]")
    console.print(f"Author: {frevl.__author__}")
    console.print(f"License: {frevl.__license__}")
    
    # System info
    console.print("\n[bold]System Information:[/bold]")
    console.print(f"Python: {sys.version.split()[0]}")
    console.print(f"PyTorch: {torch.__version__}")
    console.print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        console.print(f"CUDA version: {torch.version.cuda}")
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")
        console.print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    if parsed_args.verbose:
        # Available models
        console.print("\n[bold]Available Models:[/bold]")
        models_dir = Path("./checkpoints")
        if models_dir.exists():
            for model_path in models_dir.iterdir():
                if model_path.is_dir():
                    console.print(f"  - {model_path.name}")
        
        # Available datasets
        console.print("\n[bold]Available Datasets:[/bold]")
        data_dir = Path("./data")
        if data_dir.exists():
            for dataset_path in data_dir.iterdir():
                if dataset_path.is_dir():
                    console.print(f"  - {dataset_path.name}")
        
        # Package dependencies
        console.print("\n[bold]Key Dependencies:[/bold]")
        try:
            import clip
            console.print(f"  - CLIP: installed")
        except ImportError:
            console.print(f"  - CLIP: [red]not installed[/red]")
        
        try:
            import gradio
            console.print(f"  - Gradio: v{gradio.__version__}")
        except ImportError:
            console.print(f"  - Gradio: [red]not installed[/red]")
        
        try:
            import fastapi
            console.print(f"  - FastAPI: installed")
        except ImportError:
            console.print(f"  - FastAPI: [red]not installed[/red]")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="FrEVL - Frozen Embeddings Vision-Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  frevl-train --config configs/train_vqa.yaml
  frevl-evaluate --model frevl-base --dataset vqa
  frevl-demo --model checkpoints/best_model.pt
  frevl-serve --model frevl-base --port 8000
  frevl-download --type model --model frevl-base frevl-large
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add subcommands
    subparsers.add_parser("train", help="Train model")
    subparsers.add_parser("evaluate", help="Evaluate model")
    subparsers.add_parser("demo", help="Launch interactive demo")
    subparsers.add_parser("serve", help="Start API server")
    subparsers.add_parser("download", help="Download models and datasets")
    subparsers.add_parser("info", help="Show FrEVL information")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "demo":
        demo()
    elif args.command == "serve":
        serve()
    elif args.command == "download":
        download()
    elif args.command == "info":
        info()
    else:
        parser.print_help()
        console.print("\n[bold]Available commands:[/bold]")
        console.print("  train     - Train a FrEVL model")
        console.print("  evaluate  - Evaluate model performance")
        console.print("  demo      - Launch interactive Gradio demo")
        console.print("  serve     - Start FastAPI server")
        console.print("  download  - Download models and datasets")
        console.print("  info      - Show system and package information")


if __name__ == "__main__":
    main()
