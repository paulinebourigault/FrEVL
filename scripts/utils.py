"""
Utility functions for FrEVL
Helper functions for training, evaluation, and deployment
"""

import os
import json
import random
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
from datetime import datetime
import hashlib
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import clip
from torchvision import transforms
from tqdm import tqdm
import yaml
import wandb


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logger(
    name: str,
    log_dir: str = "./logs",
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Formatter
    if format_string is None:
        format_string = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    formatter = logging.Formatter(format_string)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        Path(log_dir) / f"{name}_{timestamp}.log"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# Image Processing
# ============================================================================

class ImagePreprocessor:
    """Image preprocessing pipeline"""
    
    def __init__(self, model_type: str = "clip"):
        if model_type == "clip":
            _, self.preprocess = clip.load("ViT-B/32", device="cpu")
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __call__(self, image: Union[Image.Image, str, np.ndarray]) -> torch.Tensor:
        """Preprocess image"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        return self.preprocess(image)


def preprocess_image(
    image: Union[Image.Image, str, np.ndarray],
    target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """Simple image preprocessing"""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(image)


# ============================================================================
# Training Utilities
# ============================================================================

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """Check if should stop training"""
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping"""
        self.counter = 0
        self.should_stop = False
        self.best_value = float('inf') if self.mode == "min" else float('-inf')


class ModelCheckpoint:
    """Model checkpoint callback"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best: bool = True,
        save_last: bool = True,
        max_checkpoints: int = 5
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.max_checkpoints = max_checkpoints
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.checkpoints = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ):
        """Save model checkpoint"""
        # Check if best model
        current_value = metrics.get(self.monitor, 0)
        is_best = False
        
        if self.mode == "min":
            is_best = current_value < self.best_value
        else:
            is_best = current_value > self.best_value
        
        if is_best:
            self.best_value = current_value
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_value': self.best_value,
            **kwargs
        }
        
        # Save checkpoint
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_{timestamp}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        # Save best model
        if self.save_best and is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Save last model
        if self.save_last:
            last_path = self.checkpoint_dir / "last_model.pt"
            torch.save(checkpoint, last_path)
        
        # Clean up old checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


# ============================================================================
# Optimization Utilities
# ============================================================================

def create_optimizer(
    model: nn.Module,
    config: Any
) -> torch.optim.Optimizer:
    """Create optimizer based on config"""
    
    # Get parameters to optimize
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if config.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Any,
    steps_per_epoch: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler"""
    
    if config.scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps_per_epoch,
            T_mult=2,
            eta_min=1e-6
        )
    elif config.scheduler.lower() == "linear":
        total_steps = steps_per_epoch * config.epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
    elif config.scheduler.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=steps_per_epoch * 5,
            gamma=0.1
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0
        )
    
    return scheduler


# ============================================================================
# Data Utilities
# ============================================================================

def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """Custom collate function for batching"""
    images, texts, labels = zip(*batch)
    
    # Stack images
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)
    else:
        # Convert to tensors if needed
        preprocessor = ImagePreprocessor()
        images = torch.stack([preprocessor(img) for img in images])
    
    # Process texts
    if isinstance(texts[0], str):
        # Keep as list of strings
        texts = list(texts)
    else:
        # Stack if already tokenized
        texts = torch.stack(texts)
    
    # Stack labels
    if isinstance(labels[0], torch.Tensor):
        labels = torch.stack(labels)
    else:
        labels = torch.tensor(labels)
    
    return images, texts, labels


def worker_init_fn(worker_id: int):
    """Initialize worker with unique random seed"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
# Evaluation Metrics
# ============================================================================

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate accuracy"""
    correct = (predictions == targets).float().sum()
    total = targets.size(0)
    return (correct / total).item()


def calculate_precision_recall_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score"""
    from sklearn.metrics import precision_recall_fscore_support
    
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted'
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_retrieval_metrics(
    similarities: torch.Tensor,
    positive_pairs: torch.Tensor
) -> Dict[str, float]:
    """Calculate retrieval metrics (R@1, R@5, R@10)"""
    batch_size = similarities.size(0)
    
    # Sort similarities
    _, indices = similarities.topk(10, dim=1, largest=True, sorted=True)
    
    # Calculate recall at different K
    metrics = {}
    for k in [1, 5, 10]:
        correct = 0
        for i in range(batch_size):
            if positive_pairs[i] in indices[i, :k]:
                correct += 1
        metrics[f'R@{k}'] = correct / batch_size
    
    return metrics


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def setup_distributed(rank: int, world_size: int):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training"""
    torch.distributed.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Reduce tensor across all processes"""
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt


# ============================================================================
# Visualization Utilities
# ============================================================================

def visualize_attention(
    image: Image.Image,
    attention_weights: np.ndarray,
    save_path: Optional[str] = None
) -> Image.Image:
    """Visualize attention weights on image"""
    import matplotlib.pyplot as plt
    import cv2
    
    # Resize attention to match image size
    img_array = np.array(image)
    attention_resized = cv2.resize(
        attention_weights,
        (img_array.shape[1], img_array.shape[0])
    )
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(img_array)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Attention overlay
    ax2.imshow(img_array)
    ax2.imshow(attention_resized, cmap='jet', alpha=0.5)
    ax2.set_title("Attention Heatmap")
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to PIL Image
    fig.canvas.draw()
    img = Image.frombytes(
        'RGB',
        fig.canvas.get_width_height(),
        fig.canvas.tostring_rgb()
    )
    
    plt.close()
    return img


# ============================================================================
# Cache Utilities
# ============================================================================

class FileCache:
    """Simple file-based cache"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return self._get_cache_path(key).exists()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        cache_path = self._get_cache_path(key)
        with open(cache_path, 'wb') as f:
            pickle.dump(value, f)
    
    def clear(self):
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


# ============================================================================
# Configuration Utilities
# ============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML or JSON file"""
    path = Path(config_path)
    
    if path.suffix in ['.yaml', '.yml']:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unknown config file format: {path.suffix}")
    
    return config


def save_config(config: Dict, save_path: str):
    """Save configuration to file"""
    path = Path(save_path)
    
    if path.suffix in ['.yaml', '.yml']:
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif path.suffix == '.json':
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unknown config file format: {path.suffix}")


# ============================================================================
# Model Export Utilities
# ============================================================================

def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    opset_version: int = 14
):
    """Export model to ONNX format"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {save_path}")


def export_to_torchscript(
    model: nn.Module,
    save_path: str,
    example_input: Optional[torch.Tensor] = None
):
    """Export model to TorchScript"""
    model.eval()
    
    if example_input is None:
        example_input = torch.randn(1, 3, 224, 224)
    
    # Trace model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save
    traced_model.save(save_path)
    
    print(f"Model exported to {save_path}")


# ============================================================================
# Profiling Utilities
# ============================================================================

class Timer:
    """Simple timer for profiling"""
    
    def __init__(self):
        self.times = defaultdict(list)
    
    def __call__(self, name: str):
        return self.TimeContext(self, name)
    
    class TimeContext:
        def __init__(self, timer, name):
            self.timer = timer
            self.name = name
        
        def __enter__(self):
            self.start = time.time()
            return self
        
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            self.timer.times[self.name].append(elapsed)
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary"""
        summary = {}
        for name, times in self.times.items():
            summary[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times),
                'count': len(times)
            }
        return summary
    
    def print_summary(self):
        """Print timing summary"""
        summary = self.summary()
        for name, stats in summary.items():
            print(f"{name}:")
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")


def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    num_iterations: int = 100,
    use_cuda: bool = True
) -> Dict[str, float]:
    """Profile model performance"""
    model.eval()
    
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Warmup
    dummy_input = torch.randn(input_shape).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Profile
    if use_cuda:
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_iterations):
        start = time.time()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        if use_cuda:
            torch.cuda.synchronize()
        
        times.append(time.time() - start)
    
    # Calculate statistics
    times = np.array(times) * 1000  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p50_ms': np.percentile(times, 50),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'throughput': 1000 / np.mean(times)  # images/sec
    }
