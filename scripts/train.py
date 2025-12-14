"""
FrEVL: Training Script 
"""

import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import numpy as np
from tqdm import tqdm
import wandb
import clip
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import yaml
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict
import pickle

# Custom modules
from model import FrEVL
from data_loader import VQADataset, SNLIVEDataset, COCODataset
from utils import AverageMeter, EarlyStopping, ModelCheckpoint, setup_logger
from optimizers import create_optimizer, create_scheduler


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with type hints and defaults"""
    # Model
    model_name: str = "frevl-base"
    clip_model: str = "ViT-B/32"
    hidden_dim: int = 768
    num_layers: int = 6
    num_heads: int = 12
    dropout: float = 0.1
    
    # Dataset
    dataset: str = "vqa"
    data_root: str = "./data"
    cache_embeddings: bool = True
    num_workers: int = 8
    prefetch_factor: int = 2
    
    # Training
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_prob: float = 0.5
    
    # Distributed
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1
    backend: str = "nccl"
    
    # Logging
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    wandb_project: str = "frevl"
    wandb_entity: Optional[str] = None
    use_tensorboard: bool = True
    
    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Other
    seed: int = 42
    resume_from: Optional[str] = None
    early_stopping_patience: int = 5
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Load config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


# ============================================================================
# Model Architecture
# ============================================================================

class FrEVL(nn.Module):
    """FrEVL: Frozen Embeddings Vision-Language Model"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Load frozen CLIP model
        self.clip_model, self.preprocess = clip.load(config.clip_model, device="cpu")
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Dimensions
        self.vision_dim = self.clip_model.visual.output_dim
        self.text_dim = self.clip_model.token_embedding.embedding_dim
        self.hidden_dim = config.hidden_dim
        
        # Projection layers
        self.vision_proj = nn.Linear(self.vision_dim, self.hidden_dim)
        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim)
        
        # Cross-attention fusion network
        self.fusion_network = nn.ModuleList([
            CrossAttentionLayer(
                hidden_dim=self.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # Task-specific heads
        self.vqa_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim // 2, 3129)  # VQA v2 answer vocab size
        )
        
        self.retrieval_head = nn.Linear(self.hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using frozen CLIP encoder"""
        return self.clip_model.encode_image(images)
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text using frozen CLIP encoder"""
        tokens = clip.tokenize(texts).to(images.device)
        return self.clip_model.encode_text(tokens)
    
    def forward(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor,
        task: str = "vqa"
    ) -> torch.Tensor:
        """Forward pass through fusion network"""
        
        # Project features
        vision_emb = self.vision_proj(image_features)
        text_emb = self.text_proj(text_features)
        
        # Cross-attention fusion
        fused = vision_emb
        for layer in self.fusion_network:
            fused = layer(fused, text_emb)
        
        # Task-specific output
        if task == "vqa":
            return self.vqa_head(fused.mean(dim=1))
        elif task == "retrieval":
            return self.retrieval_head(fused.mean(dim=1))
        else:
            return fused


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for vision-language fusion"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Forward pass with cross-attention"""
        # Cross-attention
        attn_output, _ = self.attention(query, key_value, key_value)
        query = self.norm1(query + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(query)
        output = self.norm2(query + self.dropout(ffn_output))
        
        return output


# ============================================================================
# Data Loading
# ============================================================================

class EmbeddingCache:
    """Cache for precomputed CLIP embeddings"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
    
    def get_cache_path(self, dataset: str, split: str) -> Path:
        return self.cache_dir / f"{dataset}_{split}_embeddings.pkl"
    
    def load(self, dataset: str, split: str) -> Optional[Dict]:
        cache_path = self.get_cache_path(dataset, split)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save(self, embeddings: Dict, dataset: str, split: str):
        cache_path = self.get_cache_path(dataset, split)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)


class CachedDataset(Dataset):
    """Dataset with cached embeddings"""
    
    def __init__(
        self, 
        dataset: Dataset, 
        clip_model: nn.Module,
        cache: EmbeddingCache,
        dataset_name: str,
        split: str
    ):
        self.dataset = dataset
        self.clip_model = clip_model
        self.cache = cache
        self.dataset_name = dataset_name
        self.split = split
        
        # Try to load cached embeddings
        self.embeddings = cache.load(dataset_name, split)
        if self.embeddings is None:
            self.precompute_embeddings()
    
    @torch.no_grad()
    def precompute_embeddings(self):
        """Precompute and cache all embeddings"""
        print(f"Precomputing embeddings for {self.dataset_name} {self.split}...")
        
        self.embeddings = {
            'image': [],
            'text': [],
            'labels': []
        }
        
        dataloader = DataLoader(
            self.dataset, 
            batch_size=256, 
            num_workers=8,
            pin_memory=True
        )
        
        for batch in tqdm(dataloader):
            images, texts, labels = batch
            
            # Encode images
            image_emb = self.clip_model.encode_image(images.cuda())
            self.embeddings['image'].append(image_emb.cpu())
            
            # Encode texts
            text_tokens = clip.tokenize(texts).cuda()
            text_emb = self.clip_model.encode_text(text_tokens)
            self.embeddings['text'].append(text_emb.cpu())
            
            self.embeddings['labels'].append(labels)
        
        # Concatenate all batches
        self.embeddings['image'] = torch.cat(self.embeddings['image'])
        self.embeddings['text'] = torch.cat(self.embeddings['text'])
        self.embeddings['labels'] = torch.cat(self.embeddings['labels'])
        
        # Save cache
        self.cache.save(self.embeddings, self.dataset_name, self.split)
        print(f"Cached {len(self.embeddings['image'])} embeddings")
    
    def __len__(self):
        return len(self.embeddings['image'])
    
    def __getitem__(self, idx):
        return (
            self.embeddings['image'][idx],
            self.embeddings['text'][idx],
            self.embeddings['labels'][idx]
        )


# ============================================================================
# Training Functions
# ============================================================================

class Trainer:
    """Main trainer class with all production features"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.logger = setup_logger("FrEVL", config.log_dir)
        
        # Setup distributed training
        if config.distributed:
            self.setup_distributed()
        
        # Initialize model
        self.model = self.build_model()
        
        # Setup data
        self.train_loader, self.val_loader = self.setup_data()
        
        # Setup optimization
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, config, len(self.train_loader))
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Setup monitoring
        self.setup_monitoring()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # Setup callbacks
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        self.checkpoint_manager = ModelCheckpoint(
            config.checkpoint_dir,
            max_checkpoints=5
        )
        
    def setup_distributed(self):
        """Setup distributed training"""
        dist.init_process_group(backend=self.config.backend)
        self.config.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.config.local_rank)
        self.device = torch.device(f"cuda:{self.config.local_rank}")
        
    def build_model(self) -> nn.Module:
        """Build and prepare model"""
        model = FrEVL(self.config).to(self.device)
        
        if self.config.distributed:
            model = DDP(
                model, 
                device_ids=[self.config.local_rank],
                find_unused_parameters=True
            )
        
        # Load checkpoint if resuming
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)
        
        return model
    
    def setup_data(self) -> Tuple[DataLoader, DataLoader]:
        """Setup data loaders with caching"""
        cache = EmbeddingCache() if self.config.cache_embeddings else None
        
        # Load datasets based on config
        if self.config.dataset == "vqa":
            train_dataset = VQADataset(self.config.data_root, split="train")
            val_dataset = VQADataset(self.config.data_root, split="val")
        elif self.config.dataset == "snli-ve":
            train_dataset = SNLIVEDataset(self.config.data_root, split="train")
            val_dataset = SNLIVEDataset(self.config.data_root, split="val")
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
        
        # Wrap with caching if enabled
        if cache:
            clip_model, _ = clip.load(self.config.clip_model, device="cuda")
            train_dataset = CachedDataset(
                train_dataset, clip_model, cache, 
                self.config.dataset, "train"
            )
            val_dataset = CachedDataset(
                val_dataset, clip_model, cache,
                self.config.dataset, "val"
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=self.config.prefetch_factor
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def setup_monitoring(self):
        """Setup monitoring tools"""
        # Weights & Biases
        if self.config.wandb_project and (not self.config.distributed or self.config.local_rank == 0):
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                name=f"{self.config.model_name}_{datetime.now():%Y%m%d_%H%M%S}"
            )
        
        # TensorBoard
        if self.config.use_tensorboard and (not self.config.distributed or self.config.local_rank == 0):
            self.writer = SummaryWriter(
                log_dir=Path(self.config.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        
        # Metrics tracking
        self.metrics = defaultdict(AverageMeter)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = defaultdict(AverageMeter)
        
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.epoch + 1}/{self.config.epochs}",
            disable=self.config.distributed and self.config.local_rank != 0
        )
        
        for batch_idx, (image_features, text_features, labels) in enumerate(pbar):
            # Move to device
            image_features = image_features.to(self.device)
            text_features = text_features.to(self.device)
            labels = labels.to(self.device)
            
            # Data augmentation (mixup/cutmix)
            if np.random.random() < self.config.mixup_alpha:
                image_features, labels = self.mixup(image_features, labels)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(image_features, text_features, task=self.config.dataset)
                    loss = self.compute_loss(outputs, labels)
            else:
                outputs = self.model(image_features, text_features, task=self.config.dataset)
                loss = self.compute_loss(outputs, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                # Update metrics
                epoch_metrics['loss'].update(loss.item() * self.config.gradient_accumulation_steps)
                epoch_metrics['lr'].update(self.optimizer.param_groups[0]['lr'])
                
                # Log metrics
                if self.global_step % self.config.log_interval == 0:
                    self.log_metrics(epoch_metrics, prefix="train")
                
                # Validation
                if self.global_step % self.config.eval_interval == 0:
                    val_metrics = self.validate()
                    self.log_metrics(val_metrics, prefix="val")
                    self.model.train()
                
                # Save checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{epoch_metrics['loss'].avg:.4f}",
                'lr': f"{epoch_metrics['lr'].avg:.2e}"
            })
        
        return {k: v.avg for k, v in epoch_metrics.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        val_metrics = defaultdict(AverageMeter)
        all_preds = []
        all_labels = []
        
        for image_features, text_features, labels in tqdm(self.val_loader, desc="Validating"):
            image_features = image_features.to(self.device)
            text_features = text_features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(image_features, text_features, task=self.config.dataset)
            loss = self.compute_loss(outputs, labels)
            
            # Get predictions
            preds = outputs.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update metrics
            val_metrics['loss'].update(loss.item())
        
        # Calculate accuracy and other metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        val_metrics['accuracy'].update(accuracy)
        val_metrics['precision'].update(precision)
        val_metrics['recall'].update(recall)
        val_metrics['f1'].update(f1)
        
        # Check for best model
        if accuracy > self.best_metric:
            self.best_metric = accuracy
            self.save_checkpoint(is_best=True)
        
        return {k: v.avg for k, v in val_metrics.items()}
    
    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss with label smoothing"""
        if self.config.label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion(outputs, labels)
    
    def mixup(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixup data augmentation"""
        batch_size = images.size(0)
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_images, mixed_labels
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to monitoring tools"""
        # Console logging
        log_str = f"Step {self.global_step} | "
        log_str += " | ".join([f"{prefix}/{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(log_str)
        
        # Weights & Biases
        if hasattr(self, 'wandb') and wandb.run:
            wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=self.global_step)
        
        # TensorBoard
        if hasattr(self, 'writer'):
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, self.global_step)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': asdict(self.config)
        }
        
        if self.config.mixed_precision and self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with accuracy: {self.best_metric:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        if self.config.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training with config:\n{self.config}")
        
        for self.epoch in range(self.epoch, self.config.epochs):
            # Training epoch
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Log epoch summary
            self.logger.info(
                f"Epoch {self.epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                self.logger.info(f"Early stopping triggered at epoch {self.epoch + 1}")
                break
        
        # Training complete
        self.logger.info(f"Training complete! Best accuracy: {self.best_metric:.4f}")
        
        # Cleanup
        if hasattr(self, 'writer'):
            self.writer.close()
        if wandb.run:
            wandb.finish()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train FrEVL model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--dataset", type=str, default="vqa", choices=["vqa", "snli-ve", "coco"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="frevl")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--resume-from", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()
    
    # Override with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != "config":
            setattr(config, key, value)
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
