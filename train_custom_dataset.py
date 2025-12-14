#!/usr/bin/env python3
"""
Training FrEVL on Custom Dataset
Example showing how to prepare and train on your own vision-language data
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

import torch
from torch.utils.data import DataLoader
from PIL import Image

from frevl import (
    FrEVL,
    FrEVLConfig,
    CustomVLDataset,
    Trainer,
    TrainingConfig,
    create_optimizer,
    create_scheduler,
    setup_logger
)


def prepare_custom_data(
    images_dir: str,
    annotations_file: str,
    output_file: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1
):
    """
    Prepare custom dataset in the required format
    
    Expected input format (CSV or JSON):
    - image: filename of the image
    - text/caption/question: text associated with the image  
    - label: optional label for classification tasks
    - metadata: optional additional information
    
    Args:
        images_dir: Directory containing images
        annotations_file: Path to annotations (CSV or JSON)
        output_file: Path to save processed dataset
        train_split: Percentage for training
        val_split: Percentage for validation
        test_split: Percentage for testing
    """
    
    print("Preparing custom dataset...")
    
    # Load annotations
    if annotations_file.endswith('.csv'):
        df = pd.read_csv(annotations_file)
        data = df.to_dict('records')
    else:
        with open(annotations_file, 'r') as f:
            data = json.load(f)
    
    # Validate images exist
    images_dir = Path(images_dir)
    valid_data = []
    
    for item in data:
        image_path = images_dir / item['image']
        if image_path.exists():
            valid_data.append(item)
        else:
            print(f"Warning: Image not found: {image_path}")
    
    print(f"Found {len(valid_data)} valid samples")
    
    # Split data
    import random
    random.shuffle(valid_data)
    
    n_total = len(valid_data)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_data = valid_data[:n_train]
    val_data = valid_data[n_train:n_train + n_val]
    test_data = valid_data[n_train + n_val:]
    
    # Add split information
    for item in train_data:
        item['split'] = 'train'
    for item in val_data:
        item['split'] = 'val'
    for item in test_data:
        item['split'] = 'test'
    
    # Combine all data
    all_data = train_data + val_data + test_data
    
    # Save processed dataset
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Dataset saved to {output_file}")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    return output_file


def create_custom_dataloaders(
    data_file: str,
    images_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create data loaders for custom dataset
    
    Args:
        data_file: Path to processed dataset file
        images_dir: Directory containing images
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        Dictionary with train, val, and test dataloaders
    """
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = CustomVLDataset(
            data_file=data_file,
            image_dir=images_dir,
            split=split,
            use_augmentation=(split == 'train')
        )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
        
        print(f"{split.capitalize()} dataloader: {len(dataset)} samples, "
              f"{len(dataloaders[split])} batches")
    
    return dataloaders


def train_on_custom_dataset(
    data_file: str,
    images_dir: str,
    model_config: Optional[FrEVLConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    output_dir: str = "./custom_model"
):
    """
    Train FrEVL on custom dataset
    
    Args:
        data_file: Path to processed dataset
        images_dir: Directory containing images
        model_config: Model configuration
        training_config: Training configuration
        output_dir: Directory to save model and logs
    """
    
    # Setup logger
    logger = setup_logger("custom_training", output_dir)
    logger.info("Starting training on custom dataset")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    if model_config is None:
        model_config = FrEVLConfig(
            clip_model="ViT-B/32",
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
            dropout=0.1
        )
    
    model = FrEVL(model_config)
    model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {trainable_params:,} trainable parameters "
                f"({100 * trainable_params / total_params:.1f}% of total)")
    
    # Create dataloaders
    dataloaders = create_custom_dataloaders(
        data_file,
        images_dir,
        batch_size=training_config.batch_size if training_config else 32,
        num_workers=4
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        training_config if training_config else TrainingConfig()
    )
    
    # Create scheduler
    steps_per_epoch = len(dataloaders['train'])
    scheduler = create_scheduler(
        optimizer,
        training_config if training_config else TrainingConfig(),
        steps_per_epoch
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders['val'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        config=training_config if training_config else TrainingConfig()
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(dataloaders['test'])
    logger.info(f"Test results: {test_results}")
    
    # Save final model
    final_model_path = output_dir / "final_model"
    model.save_pretrained(final_model_path)
    logger.info(f"Model saved to {final_model_path}")
    
    return model


def example_medical_vqa():
    """Example: Training on medical VQA dataset"""
    print("\n" + "="*60)
    print("Example: Medical VQA Dataset")
    print("="*60)
    
    # Prepare synthetic medical VQA data
    medical_data = [
        {
            "image": "xray_001.jpg",
            "text": "What type of medical image is this?",
            "label": 0,  # X-ray
            "metadata": {"modality": "xray", "body_part": "chest"}
        },
        {
            "image": "mri_001.jpg",
            "text": "What body part is shown?",
            "label": 1,  # Brain
            "metadata": {"modality": "mri", "body_part": "brain"}
        },
        {
            "image": "ct_001.jpg",
            "text": "Is there any abnormality visible?",
            "label": 2,  # Yes
            "metadata": {"modality": "ct", "body_part": "abdomen"}
        },
        # Add more samples...
    ]
    
    # Save example data
    example_dir = Path("examples/medical_vqa")
    example_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = example_dir / "annotations.json"
    with open(data_file, 'w') as f:
        json.dump(medical_data, f, indent=2)
    
    print(f"Example medical VQA data saved to {data_file}")
    print("To train on this data:")
    print(f"  1. Add medical images to {example_dir}/images/")
    print(f"  2. Run: python train_custom_dataset.py --data {data_file} "
          f"--images {example_dir}/images/")


def example_product_search():
    """Example: Training on e-commerce product search"""
    print("\n" + "="*60)
    print("Example: E-commerce Product Search")
    print("="*60)
    
    # Prepare synthetic product data
    product_data = [
        {
            "image": "product_001.jpg",
            "text": "Red running shoes with white stripes",
            "label": "footwear",
            "metadata": {
                "category": "shoes",
                "color": "red",
                "brand": "SportBrand"
            }
        },
        {
            "image": "product_002.jpg",
            "text": "Wireless noise-canceling headphones",
            "label": "electronics",
            "metadata": {
                "category": "audio",
                "color": "black",
                "features": ["wireless", "noise-canceling"]
            }
        },
        # Add more samples...
    ]
    
    # Save example data
    example_dir = Path("examples/product_search")
    example_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = example_dir / "products.json"
    with open(data_file, 'w') as f:
        json.dump(product_data, f, indent=2)
    
    print(f"Example product data saved to {data_file}")
    print("This can be used for:")
    print("  - Visual product search")
    print("  - Product recommendation")
    print("  - Inventory management")


def main():
    parser = argparse.ArgumentParser(
        description="Train FrEVL on custom dataset"
    )
    
    # Data arguments
    parser.add_argument("--prepare", action="store_true",
                       help="Prepare dataset from raw annotations")
    parser.add_argument("--annotations", type=str,
                       help="Path to raw annotations (CSV or JSON)")
    parser.add_argument("--images", type=str,
                       help="Directory containing images")
    parser.add_argument("--data", type=str,
                       help="Path to processed dataset")
    
    # Training arguments
    parser.add_argument("--output-dir", type=str, default="./custom_model",
                       help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32",
                       help="CLIP backbone model")
    
    # Other arguments
    parser.add_argument("--examples", action="store_true",
                       help="Create example datasets")
    
    args = parser.parse_args()
    
    if args.examples:
        # Create example datasets
        example_medical_vqa()
        example_product_search()
        print("\n✓ Example datasets created!")
        return
    
    # Prepare dataset if requested
    if args.prepare:
        if not args.annotations or not args.images:
            print("Error: --annotations and --images required for data preparation")
            return
        
        data_file = prepare_custom_data(
            images_dir=args.images,
            annotations_file=args.annotations,
            output_file=args.data or "custom_dataset.json"
        )
    else:
        data_file = args.data
    
    if not data_file:
        print("Error: --data required for training (or use --prepare)")
        return
    
    # Setup configurations
    model_config = FrEVLConfig(
        clip_model=args.clip_model,
        hidden_dim=512,
        num_layers=4,
        num_heads=8
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        mixed_precision=torch.cuda.is_available()
    )
    
    # Train model
    model = train_on_custom_dataset(
        data_file=data_file,
        images_dir=args.images,
        model_config=model_config,
        training_config=training_config,
        output_dir=args.output_dir
    )
    
    print("\n Training complete!")
    print(f"Model saved to {args.output_dir}/final_model")


if __name__ == "__main__":
    main()
