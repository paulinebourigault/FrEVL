"""
FrEVL: Frozen Embeddings for Efficient Vision-Language Understanding

A lightweight framework for vision-language understanding that leverages
frozen CLIP embeddings with trainable fusion layers.
"""

__version__ = "1.0.0"
__author__ = "Emmanuelle Bourigault"
__email__ = "emmanuelle.bourigault@research.org"
__license__ = "MIT"

# Core imports
from .model import FrEVL, FrEVLConfig
from .data_loader import (
    VQADataset,
    SNLIVEDataset,
    COCODataset,
    CustomVLDataset,
    create_dataloader
)
from .train import Trainer, TrainingConfig
from .evaluate import Evaluator, benchmark_performance
from .utils import (
    setup_logger,
    AverageMeter,
    MetricTracker,
    EarlyStopping,
    ModelCheckpoint,
    preprocess_image,
    create_attention_map,
    set_random_seed,
    load_checkpoint
)
from .optimizers import (
    create_optimizer,
    create_scheduler,
    LAMB,
    Lookahead,
    WarmupCosineScheduler
)

# Version check
import torch
import sys

if sys.version_info < (3, 8):
    raise RuntimeError("FrEVL requires Python 3.8 or later")

if torch.__version__ < "2.0.0":
    import warnings
    warnings.warn(
        "FrEVL is optimized for PyTorch 2.0+. "
        "Some features may not be available with older versions."
    )

# Module exports
__all__ = [
    # Version
    "__version__",
    
    # Model
    "FrEVL",
    "FrEVLConfig",
    
    # Data
    "VQADataset",
    "SNLIVEDataset",
    "COCODataset",
    "CustomVLDataset",
    "create_dataloader",
    
    # Training
    "Trainer",
    "TrainingConfig",
    
    # Evaluation
    "Evaluator",
    "benchmark_performance",
    
    # Utils
    "setup_logger",
    "AverageMeter",
    "MetricTracker",
    "EarlyStopping",
    "ModelCheckpoint",
    "preprocess_image",
    "create_attention_map",
    "set_random_seed",
    "load_checkpoint",
    
    # Optimizers
    "create_optimizer",
    "create_scheduler",
    "LAMB",
    "Lookahead",
    "WarmupCosineScheduler",
]

# Convenience functions
def load_model(name_or_path: str, device: str = "cuda") -> FrEVL:
    """
    Load a pretrained FrEVL model
    
    Args:
        name_or_path: Model name (e.g., 'frevl-base') or path to checkpoint
        device: Device to load model on
    
    Returns:
        FrEVL model instance
    """
    import os
    from pathlib import Path
    
    # Check if it's a path or a model name
    if os.path.exists(name_or_path):
        model = FrEVL.from_pretrained(name_or_path)
    else:
        # Try to download from HuggingFace
        from huggingface_hub import snapshot_download
        
        model_map = {
            "frevl-base": "EmmanuelleB985/frevl-base",
            "frevl-large": "EmmanuelleB985/frevl-large",
            "frevl-multi": "EmmanuelleB985/frevl-multilingual",
        }
        
        if name_or_path in model_map:
            repo_id = model_map[name_or_path]
            cache_dir = Path.home() / ".cache" / "frevl"
            
            model_dir = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                resume_download=True
            )
            
            model = FrEVL.from_pretrained(model_dir)
        else:
            raise ValueError(f"Unknown model: {name_or_path}")
    
    # Move to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model


def get_default_config(task: str = "vqa") -> dict:
    """
    Get default configuration for a task
    
    Args:
        task: Task name ('vqa', 'retrieval', 'classification')
    
    Returns:
        Configuration dictionary
    """
    base_config = {
        "model": {
            "clip_model": "ViT-B/32",
            "hidden_dim": 768,
            "num_layers": 6,
            "num_heads": 12,
            "dropout": 0.1,
        },
        "training": {
            "batch_size": 128,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "epochs": 20,
            "warmup_ratio": 0.1,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "mixed_precision": True,
            "gradient_checkpointing": False,
        },
        "data": {
            "num_workers": 8,
            "cache_embeddings": True,
            "use_augmentation": True,
        },
        "logging": {
            "log_interval": 50,
            "eval_interval": 500,
            "save_interval": 1000,
        }
    }
    
    # Task-specific modifications
    if task == "vqa":
        base_config["model"]["num_vqa_answers"] = 3129
    elif task == "retrieval":
        base_config["training"]["batch_size"] = 256
        base_config["model"]["temperature"] = 0.07
    elif task == "classification":
        base_config["model"]["num_classes"] = 3
    
    return base_config


# Print package info when imported
if __name__ == "__main__":
    print(f"FrEVL v{__version__}")
    print(f"PyTorch v{torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
