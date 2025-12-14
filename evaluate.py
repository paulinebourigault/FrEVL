"""
FrEVL Model Evaluation Script
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
import wandb

from model import FrEVL
from data_loader import VQADataset, SNLIVEDataset, COCODataset
from utils import (
    setup_logger,
    MetricTracker,
    compute_metrics,
    plot_confusion_matrix,
    timer,
    set_random_seed,
    load_checkpoint
)


# ============================================================================
# Evaluation Functions
# ============================================================================

class Evaluator:
    """Comprehensive model evaluator"""
    
    def __init__(
        self,
        model: FrEVL,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        # Metrics storage
        self.results = {}
        self.predictions = []
        self.ground_truth = []
        self.attention_maps = []
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        dataset_name: str,
        task: str = "vqa",
        save_predictions: bool = False,
        compute_attention: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset
        
        Args:
            dataloader: Data loader
            dataset_name: Name of dataset
            task: Task type (vqa, retrieval, etc.)
            save_predictions: Whether to save predictions
            compute_attention: Whether to compute attention maps
        
        Returns:
            Dictionary of metrics
        """
        
        self.logger.info(f"Evaluating on {dataset_name}...")
        
        # Reset storage
        self.predictions = []
        self.ground_truth = []
        self.attention_maps = []
        
        # Metrics tracker
        metrics = MetricTracker([
            'loss', 'accuracy', 'precision', 'recall', 'f1',
            'top1_acc', 'top5_acc', 'mrr', 'inference_time'
        ])
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Evaluating {dataset_name}")
        
        total_time = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            if len(batch) == 3:
                images, questions, labels = batch
                metadata = None
            else:
                images, questions, labels, metadata = batch
            
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Time inference
            start_time = time.time()
            
            # Forward pass
            outputs = self.model(
                images=images,
                text=questions,
                task=task,
                return_attention=compute_attention
            )
            
            inference_time = (time.time() - start_time) * 1000  # ms
            total_time += inference_time
            
            # Compute loss
            if task == "vqa":
                loss = nn.CrossEntropyLoss()(outputs['logits'], labels)
                predictions = outputs['predictions']
                
                # Top-k accuracy
                top1_correct = (predictions == labels).float().mean()
                top5_correct = (labels.unsqueeze(1) == outputs['top_k_predictions']).any(1).float().mean()
                
                metrics.update({
                    'loss': loss.item(),
                    'top1_acc': top1_correct.item(),
                    'top5_acc': top5_correct.item(),
                    'inference_time': inference_time
                })
            
            elif task == "retrieval":
                # Retrieval metrics
                scores = outputs['retrieval_scores']
                predictions = (scores > 0.5).long()
                
                metrics.update({
                    'accuracy': accuracy_score(labels.cpu(), predictions.cpu()),
                    'inference_time': inference_time
                })
            
            # Store predictions
            if save_predictions:
                self.predictions.extend(predictions.cpu().numpy())
                self.ground_truth.extend(labels.cpu().numpy())
                
                if compute_attention and 'attention' in outputs:
                    self.attention_maps.append(outputs['attention'].cpu())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics.get_average('loss'):.4f}",
                'acc': f"{metrics.get_average('top1_acc'):.4f}"
            })
            
            batch_count += 1
        
        # Compute final metrics
        avg_inference_time = total_time / batch_count
        
        # Calculate additional metrics if predictions were saved
        final_metrics = metrics.get_all_averages()
        
        if save_predictions and self.predictions:
            predictions_np = np.array(self.predictions)
            labels_np = np.array(self.ground_truth)
            
            # Detailed metrics
            accuracy = accuracy_score(labels_np, predictions_np)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_np, predictions_np, average='weighted', zero_division=0
            )
            
            final_metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'avg_inference_ms': avg_inference_time,
                'total_samples': len(predictions_np)
            })
        
        # Store results
        self.results[dataset_name] = final_metrics
        
        self.logger.info(f"Evaluation results for {dataset_name}:")
        for metric, value in final_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return final_metrics
    
    def evaluate_all_datasets(
        self,
        datasets: Dict[str, DataLoader],
        tasks: Dict[str, str],
        save_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on multiple datasets
        
        Args:
            datasets: Dictionary of dataset names to dataloaders
            tasks: Dictionary mapping dataset names to task types
            save_dir: Directory to save results
        
        Returns:
            Dictionary of results per dataset
        """
        
        all_results = {}
        
        for dataset_name, dataloader in datasets.items():
            task = tasks.get(dataset_name, "vqa")
            
            # Evaluate
            results = self.evaluate_dataset(
                dataloader,
                dataset_name,
                task=task,
                save_predictions=True,
                compute_attention=False
            )
            
            all_results[dataset_name] = results
            
            # Save confusion matrix if applicable
            if save_dir and self.predictions:
                save_path = Path(save_dir) / f"{dataset_name}_confusion_matrix.png"
                self.plot_confusion_matrix(save_path)
        
        # Save all results
        if save_dir:
            self.save_results(all_results, save_dir)
        
        return all_results
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix for current predictions"""
        
        if not self.predictions:
            return
        
        fig = plot_confusion_matrix(
            np.array(self.predictions),
            np.array(self.ground_truth),
            save_path=save_path
        )
        
        return fig
    
    def save_results(
        self,
        results: Dict[str, Dict[str, float]],
        save_dir: str
    ):
        """Save evaluation results"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(save_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(results).T
        df.to_csv(save_dir / "evaluation_results.csv")
        
        # Create summary plot
        self.plot_results_summary(results, save_dir / "results_summary.png")
        
        self.logger.info(f"Results saved to {save_dir}")
    
    def plot_results_summary(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """Create summary visualization of results"""
        
        # Prepare data
        datasets = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        data = []
        for metric in metrics:
            values = [results[ds].get(metric, 0) for ds in datasets]
            data.append(values)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(datasets))
        width = 0.2
        
        for i, (metric, values) in enumerate(zip(metrics, data)):
            ax.bar(x + i * width, values, width, label=metric)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Score')
        ax.set_title('Evaluation Results Summary')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return fig
    
    def compute_inference_speed(
        self,
        dataloader: DataLoader,
        num_samples: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Measure inference speed
        
        Args:
            dataloader: Data loader
            num_samples: Number of samples to test
            warmup: Number of warmup iterations
        
        Returns:
            Dictionary with timing statistics
        """
        
        times = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_samples + warmup:
                break
            
            images, questions, labels = batch[:3]
            images = images.to(self.device)
            
            # Synchronize for accurate timing
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.model(images=images, text=questions)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = (time.time() - start_time) * 1000  # ms
            
            # Skip warmup iterations
            if i >= warmup:
                times.append(elapsed)
        
        times = np.array(times)
        
        return {
            'mean_ms': times.mean(),
            'std_ms': times.std(),
            'min_ms': times.min(),
            'max_ms': times.max(),
            'p50_ms': np.percentile(times, 50),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'throughput': 1000 / times.mean()  # samples/second
        }
    
    def evaluate_robustness(
        self,
        dataloader: DataLoader,
        perturbations: List[str] = ['gaussian_noise', 'blur', 'brightness']
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model robustness to perturbations
        
        Args:
            dataloader: Data loader
            perturbations: List of perturbation types
        
        Returns:
            Dictionary of results per perturbation
        """
        
        from torchvision import transforms
        import cv2
        
        robustness_results = {}
        
        for perturbation in perturbations:
            self.logger.info(f"Testing robustness to {perturbation}")
            
            correct = 0
            total = 0
            
            for batch in tqdm(dataloader, desc=f"Testing {perturbation}"):
                images, questions, labels = batch[:3]
                
                # Apply perturbation
                if perturbation == 'gaussian_noise':
                    noise = torch.randn_like(images) * 0.1
                    images = images + noise
                elif perturbation == 'blur':
                    # Apply Gaussian blur
                    for i in range(images.shape[0]):
                        img = images[i].numpy().transpose(1, 2, 0)
                        img = cv2.GaussianBlur(img, (5, 5), 1.0)
                        images[i] = torch.from_numpy(img.transpose(2, 0, 1))
                elif perturbation == 'brightness':
                    images = images * 1.5  # Increase brightness
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(images=images, text=questions)
                    predictions = outputs['predictions']
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            accuracy = correct / total
            robustness_results[perturbation] = {'accuracy': accuracy}
            self.logger.info(f"  Accuracy under {perturbation}: {accuracy:.4f}")
        
        return robustness_results


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_performance(
    model: FrEVL,
    dataloader: DataLoader,
    device: torch.device,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive performance benchmark
    
    Args:
        model: Model to benchmark
        dataloader: Data loader
        device: Device to run on
        output_file: Path to save results
    
    Returns:
        Dictionary of benchmark results
    """
    
    evaluator = Evaluator(model, device)
    
    # Speed benchmark
    speed_results = evaluator.compute_inference_speed(dataloader, num_samples=100)
    
    # Memory usage
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
        # Run inference
        for i, batch in enumerate(dataloader):
            if i >= 10:
                break
            images = batch[0].to(device)
            questions = batch[1]
            with torch.no_grad():
                _ = model(images=images, text=questions)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        peak_memory = 0
    
    # Model size
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024**2  # MB
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Compile results
    results = {
        'speed': speed_results,
        'memory': {
            'peak_memory_mb': peak_memory,
            'model_size_mb': model_size
        },
        'parameters': {
            'total': sum(p.numel() for p in model.parameters()),
            'trainable': trainable_params
        }
    }
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


# ============================================================================
# Main Evaluation Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate FrEVL model")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to model config")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="all", 
                       choices=["vqa", "snli-ve", "coco", "all"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    
    # Evaluation arguments
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./evaluation_results")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--compute-attention", action="store_true")
    
    # Additional tests
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--robustness", action="store_true", help="Test robustness")
    
    # Logging
    parser.add_argument("--wandb-project", type=str, help="W&B project name")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Setup logging
    logger = setup_logger("evaluation", args.output_dir)
    logger.info(f"Starting evaluation with args: {args}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = FrEVL.from_pretrained(args.model)
    model.to(device)
    model.eval()
    
    # Setup datasets
    datasets = {}
    tasks = {}
    
    if args.dataset == "all" or args.dataset == "vqa":
        datasets["vqa"] = DataLoader(
            VQADataset(args.data_root, split=args.split),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True
        )
        tasks["vqa"] = "vqa"
    
    if args.dataset == "all" or args.dataset == "snli-ve":
        datasets["snli-ve"] = DataLoader(
            SNLIVEDataset(args.data_root, split=args.split),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True
        )
        tasks["snli-ve"] = "classification"
    
    if args.dataset == "all" or args.dataset == "coco":
        datasets["coco"] = DataLoader(
            COCODataset(args.data_root, split=args.split),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True
        )
        tasks["coco"] = "retrieval"
    
    # Initialize W&B if requested
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name="evaluation", config=vars(args))
    
    # Create evaluator
    evaluator = Evaluator(model, device, logger)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_all_datasets(datasets, tasks, args.output_dir)
    
    # Log to W&B
    if args.wandb_project:
        wandb.log({"evaluation": results})
    
    # Run benchmark if requested
    if args.benchmark:
        logger.info("Running performance benchmark...")
        benchmark_results = benchmark_performance(
            model, 
            next(iter(datasets.values())),
            device,
            Path(args.output_dir) / "benchmark_results.json"
        )
        logger.info(f"Benchmark results: {benchmark_results}")
        
        if args.wandb_project:
            wandb.log({"benchmark": benchmark_results})
    
    # Test robustness if requested
    if args.robustness:
        logger.info("Testing model robustness...")
        robustness_results = evaluator.evaluate_robustness(
            next(iter(datasets.values()))
        )
        logger.info(f"Robustness results: {robustness_results}")
        
        if args.wandb_project:
            wandb.log({"robustness": robustness_results})
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    
    for dataset_name, metrics in results.items():
        logger.info(f"\n{dataset_name.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
    
    logger.info("\nEvaluation complete!")
    
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
