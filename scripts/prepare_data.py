#!/usr/bin/env python3
"""
Prepare and preprocess datasets for FrEVL training
Includes data validation, preprocessing, and embedding caching
"""

import os
import sys
import json
import argparse
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import pandas as pd
import h5py
import clip
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils import setup_logger, timer


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DataConfig:
    """Data preparation configuration"""
    data_root: str = "./data"
    cache_dir: str = "./cache"
    output_dir: str = "./processed_data"
    
    # Processing settings
    num_workers: int = mp.cpu_count()
    batch_size: int = 256
    image_size: int = 224
    max_text_length: int = 77
    
    # Caching settings
    cache_embeddings: bool = True
    cache_format: str = "hdf5"  # hdf5, npz, or pt
    chunk_size: int = 1000
    compression: Optional[str] = "gzip"
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Validation
    validate_images: bool = True
    min_image_size: int = 32
    max_aspect_ratio: float = 5.0
    
    # CLIP settings
    clip_model: str = "ViT-B/32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Data Validators
# ============================================================================

class DataValidator:
    """Validate and clean dataset files"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = setup_logger("DataValidator")
        self.stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'invalid_images': [],
            'invalid_texts': [],
            'duplicates': 0
        }
    
    def validate_image(self, image_path: Path) -> bool:
        """Validate image file"""
        try:
            # Check file exists
            if not image_path.exists():
                return False
            
            # Open and validate image
            with Image.open(image_path) as img:
                # Check format
                if img.format not in ['JPEG', 'PNG', 'BMP', 'GIF', 'WEBP']:
                    return False
                
                # Check size
                width, height = img.size
                if width < self.config.min_image_size or height < self.config.min_image_size:
                    return False
                
                # Check aspect ratio
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > self.config.max_aspect_ratio:
                    return False
                
                # Try to convert to RGB
                img.convert('RGB')
                
            return True
            
        except Exception as e:
            self.logger.warning(f"Invalid image {image_path}: {e}")
            return False
    
    def validate_text(self, text: str) -> bool:
        """Validate text input"""
        if not text or not isinstance(text, str):
            return False
        
        # Check length
        if len(text.strip()) < 3:
            return False
        
        # Check for valid characters
        if not any(c.isalnum() for c in text):
            return False
        
        return True
    
    def validate_dataset(self, dataset_path: Path, dataset_type: str) -> Dict:
        """Validate entire dataset"""
        self.logger.info(f"Validating {dataset_type} dataset at {dataset_path}")
        
        if dataset_type == "vqa":
            return self._validate_vqa(dataset_path)
        elif dataset_type == "coco":
            return self._validate_coco(dataset_path)
        elif dataset_type == "snli-ve":
            return self._validate_snli_ve(dataset_path)
        elif dataset_type == "custom":
            return self._validate_custom(dataset_path)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _validate_vqa(self, dataset_path: Path) -> Dict:
        """Validate VQA dataset"""
        valid_samples = []
        
        # Load annotations
        annotations_file = dataset_path / "v2_mscoco_train2014_annotations.json"
        questions_file = dataset_path / "v2_OpenEnded_mscoco_train2014_questions.json"
        
        if not annotations_file.exists() or not questions_file.exists():
            self.logger.error("VQA annotation files not found")
            return {'valid_samples': []}
        
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)['annotations']
        
        with open(questions_file, 'r') as f:
            questions = json.load(f)['questions']
        
        # Create mapping
        qa_pairs = {}
        for q in questions:
            qa_pairs[q['question_id']] = {
                'question': q['question'],
                'image_id': q['image_id']
            }
        
        for ann in annotations:
            if ann['question_id'] in qa_pairs:
                qa_pairs[ann['question_id']]['answers'] = ann['answers']
        
        # Validate each sample
        images_dir = dataset_path.parent / "coco" / "train2014"
        
        for qid, data in tqdm(qa_pairs.items(), desc="Validating VQA"):
            self.stats['total_samples'] += 1
            
            # Validate image
            image_file = images_dir / f"COCO_train2014_{data['image_id']:012d}.jpg"
            if not self.validate_image(image_file):
                self.stats['invalid_images'].append(str(image_file))
                continue
            
            # Validate question
            if not self.validate_text(data['question']):
                self.stats['invalid_texts'].append(data['question'])
                continue
            
            valid_samples.append({
                'question_id': qid,
                'image_path': str(image_file),
                'question': data['question'],
                'answers': data.get('answers', [])
            })
            self.stats['valid_samples'] += 1
        
        self.logger.info(f"Validated {len(valid_samples)}/{self.stats['total_samples']} VQA samples")
        return {'valid_samples': valid_samples}
    
    def _validate_coco(self, dataset_path: Path) -> Dict:
        """Validate COCO dataset"""
        valid_samples = []
        
        # Load captions
        captions_file = dataset_path / "annotations" / "captions_train2014.json"
        if not captions_file.exists():
            self.logger.error("COCO captions file not found")
            return {'valid_samples': []}
        
        with open(captions_file, 'r') as f:
            data = json.load(f)
            images = {img['id']: img for img in data['images']}
            annotations = data['annotations']
        
        # Validate each sample
        images_dir = dataset_path / "train2014"
        
        for ann in tqdm(annotations, desc="Validating COCO"):
            self.stats['total_samples'] += 1
            
            image_id = ann['image_id']
            if image_id not in images:
                continue
            
            # Validate image
            image_info = images[image_id]
            image_file = images_dir / image_info['file_name']
            
            if not self.validate_image(image_file):
                self.stats['invalid_images'].append(str(image_file))
                continue
            
            # Validate caption
            if not self.validate_text(ann['caption']):
                self.stats['invalid_texts'].append(ann['caption'])
                continue
            
            valid_samples.append({
                'image_id': image_id,
                'image_path': str(image_file),
                'caption': ann['caption'],
                'caption_id': ann['id']
            })
            self.stats['valid_samples'] += 1
        
        self.logger.info(f"Validated {len(valid_samples)}/{self.stats['total_samples']} COCO samples")
        return {'valid_samples': valid_samples}
    
    def _validate_snli_ve(self, dataset_path: Path) -> Dict:
        """Validate SNLI-VE dataset"""
        valid_samples = []
        
        # Load annotations
        annotations_file = dataset_path / "snli_ve_train.jsonl"
        if not annotations_file.exists():
            self.logger.error("SNLI-VE annotations not found")
            return {'valid_samples': []}
        
        # Validate each sample
        images_dir = dataset_path / "flickr30k_images"
        
        with open(annotations_file, 'r') as f:
            for line in tqdm(f, desc="Validating SNLI-VE"):
                self.stats['total_samples'] += 1
                
                data = json.loads(line)
                
                # Validate image
                image_file = images_dir / data['Flickr30K_ID']
                if not self.validate_image(image_file):
                    self.stats['invalid_images'].append(str(image_file))
                    continue
                
                # Validate hypothesis
                if not self.validate_text(data['sentence2']):
                    self.stats['invalid_texts'].append(data['sentence2'])
                    continue
                
                valid_samples.append({
                    'image_path': str(image_file),
                    'premise': data['sentence1'],
                    'hypothesis': data['sentence2'],
                    'label': data['gold_label']
                })
                self.stats['valid_samples'] += 1
        
        self.logger.info(f"Validated {len(valid_samples)}/{self.stats['total_samples']} SNLI-VE samples")
        return {'valid_samples': valid_samples}
    
    def _validate_custom(self, dataset_path: Path) -> Dict:
        """Validate custom dataset"""
        valid_samples = []
        
        # Support multiple formats
        if dataset_path.suffix == '.json':
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        elif dataset_path.suffix == '.csv':
            data = pd.read_csv(dataset_path).to_dict('records')
        else:
            self.logger.error(f"Unsupported format: {dataset_path.suffix}")
            return {'valid_samples': []}
        
        images_dir = dataset_path.parent / "images"
        
        for item in tqdm(data, desc="Validating custom dataset"):
            self.stats['total_samples'] += 1
            
            # Validate image
            image_file = images_dir / item['image']
            if not self.validate_image(image_file):
                self.stats['invalid_images'].append(str(image_file))
                continue
            
            # Validate text
            text_field = item.get('text', item.get('caption', item.get('question', '')))
            if not self.validate_text(text_field):
                self.stats['invalid_texts'].append(text_field)
                continue
            
            valid_samples.append({
                'image_path': str(image_file),
                'text': text_field,
                'label': item.get('label'),
                'metadata': item.get('metadata', {})
            })
            self.stats['valid_samples'] += 1
        
        self.logger.info(f"Validated {len(valid_samples)}/{self.stats['total_samples']} custom samples")
        return {'valid_samples': valid_samples}
    
    def print_stats(self):
        """Print validation statistics"""
        print("\n" + "="*50)
        print("Validation Statistics")
        print("="*50)
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Valid samples: {self.stats['valid_samples']}")
        print(f"Invalid images: {len(self.stats['invalid_images'])}")
        print(f"Invalid texts: {len(self.stats['invalid_texts'])}")
        
        if self.stats['invalid_images']:
            print(f"\nFirst 5 invalid images:")
            for img in self.stats['invalid_images'][:5]:
                print(f"  - {img}")
        
        if self.stats['invalid_texts']:
            print(f"\nFirst 5 invalid texts:")
            for txt in self.stats['invalid_texts'][:5]:
                print(f"  - {txt[:50]}...")


# ============================================================================
# Embedding Cache
# ============================================================================

class EmbeddingCache:
    """Cache CLIP embeddings for faster training"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = setup_logger("EmbeddingCache")
        
        # Load CLIP model
        self.device = torch.device(config.device)
        self.clip_model, self.preprocess = clip.load(config.clip_model, device=self.device)
        self.clip_model.eval()
        
        # Cache directory
        self.cache_dir = Path(config.cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode single image to embedding"""
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.clip_model.encode_image(image)
        features = F.normalize(features, p=2, dim=-1)
        return features.cpu().numpy()
    
    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode single text to embedding"""
        tokens = clip.tokenize([text], truncate=True).to(self.device)
        features = self.clip_model.encode_text(tokens)
        features = F.normalize(features, p=2, dim=-1)
        return features.cpu().numpy()
    
    @torch.no_grad()
    def encode_batch_images(self, image_paths: List[str]) -> np.ndarray:
        """Encode batch of images"""
        images = []
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                image = self.preprocess(image)
                images.append(image)
            except Exception as e:
                self.logger.warning(f"Failed to load image {path}: {e}")
                # Use zero embedding for failed images
                images.append(torch.zeros(3, self.config.image_size, self.config.image_size))
        
        images = torch.stack(images).to(self.device)
        features = self.clip_model.encode_image(images)
        features = F.normalize(features, p=2, dim=-1)
        return features.cpu().numpy()
    
    @torch.no_grad()
    def encode_batch_texts(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts"""
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        features = self.clip_model.encode_text(tokens)
        features = F.normalize(features, p=2, dim=-1)
        return features.cpu().numpy()
    
    def cache_dataset(self, dataset: List[Dict], dataset_name: str) -> str:
        """Cache embeddings for entire dataset"""
        self.logger.info(f"Caching embeddings for {dataset_name}")
        
        # Prepare cache file
        cache_file = self.cache_dir / f"{dataset_name}_{self.config.clip_model.replace('/', '_')}.{self.config.cache_format}"
        
        # Check if cache exists
        if cache_file.exists() and not self.config.force_rebuild:
            self.logger.info(f"Cache already exists: {cache_file}")
            return str(cache_file)
        
        # Process in batches
        num_samples = len(dataset)
        num_batches = (num_samples + self.config.batch_size - 1) // self.config.batch_size
        
        all_image_embeddings = []
        all_text_embeddings = []
        all_metadata = []
        
        for batch_idx in tqdm(range(num_batches), desc="Caching embeddings"):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, num_samples)
            batch = dataset[start_idx:end_idx]
            
            # Extract paths and texts
            image_paths = []
            texts = []
            
            for sample in batch:
                image_paths.append(sample.get('image_path'))
                
                # Handle different text fields
                text = sample.get('text', 
                      sample.get('caption',
                      sample.get('question',
                      sample.get('hypothesis', ''))))
                texts.append(text)
            
            # Encode
            image_embeddings = self.encode_batch_images(image_paths)
            text_embeddings = self.encode_batch_texts(texts)
            
            all_image_embeddings.append(image_embeddings)
            all_text_embeddings.append(text_embeddings)
            all_metadata.extend(batch)
        
        # Concatenate all embeddings
        all_image_embeddings = np.vstack(all_image_embeddings)
        all_text_embeddings = np.vstack(all_text_embeddings)
        
        # Save cache
        self.save_cache(
            cache_file,
            all_image_embeddings,
            all_text_embeddings,
            all_metadata
        )
        
        self.logger.info(f"Cached {num_samples} samples to {cache_file}")
        return str(cache_file)
    
    def save_cache(self, 
                   cache_file: Path,
                   image_embeddings: np.ndarray,
                   text_embeddings: np.ndarray,
                   metadata: List[Dict]):
        """Save embeddings to cache file"""
        
        if self.config.cache_format == "hdf5":
            # Save as HDF5
            with h5py.File(cache_file, 'w') as f:
                # Create datasets
                img_dataset = f.create_dataset(
                    'image_embeddings',
                    data=image_embeddings,
                    compression=self.config.compression if self.config.compression else None
                )
                
                txt_dataset = f.create_dataset(
                    'text_embeddings',
                    data=text_embeddings,
                    compression=self.config.compression if self.config.compression else None
                )
                
                # Save metadata as JSON string
                f.attrs['metadata'] = json.dumps(metadata)
                f.attrs['num_samples'] = len(metadata)
                f.attrs['embedding_dim'] = image_embeddings.shape[1]
                f.attrs['clip_model'] = self.config.clip_model
                
        elif self.config.cache_format == "npz":
            # Save as compressed NumPy archive
            np.savez_compressed(
                cache_file,
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                metadata=metadata
            )
            
        elif self.config.cache_format == "pt":
            # Save as PyTorch checkpoint
            torch.save({
                'image_embeddings': torch.from_numpy(image_embeddings),
                'text_embeddings': torch.from_numpy(text_embeddings),
                'metadata': metadata,
                'config': self.config
            }, cache_file)
        else:
            raise ValueError(f"Unknown cache format: {self.config.cache_format}")
    
    def load_cache(self, cache_file: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load embeddings from cache file"""
        cache_file = Path(cache_file)
        
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")
        
        self.logger.info(f"Loading cache from {cache_file}")
        
        if cache_file.suffix == ".h5" or cache_file.suffix == ".hdf5":
            with h5py.File(cache_file, 'r') as f:
                image_embeddings = f['image_embeddings'][:]
                text_embeddings = f['text_embeddings'][:]
                metadata = json.loads(f.attrs['metadata'])
                
        elif cache_file.suffix == ".npz":
            data = np.load(cache_file, allow_pickle=True)
            image_embeddings = data['image_embeddings']
            text_embeddings = data['text_embeddings']
            metadata = data['metadata'].tolist()
            
        elif cache_file.suffix == ".pt":
            data = torch.load(cache_file)
            image_embeddings = data['image_embeddings'].numpy()
            text_embeddings = data['text_embeddings'].numpy()
            metadata = data['metadata']
        else:
            raise ValueError(f"Unknown cache format: {cache_file.suffix}")
        
        self.logger.info(f"Loaded {len(metadata)} cached samples")
        return image_embeddings, text_embeddings, metadata


# ============================================================================
# Data Preprocessor
# ============================================================================

class DataPreprocessor:
    """Preprocess and augment data"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = setup_logger("DataPreprocessor")
        
        # Create augmentation pipeline
        self.augmentation = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create Albumentations augmentation pipeline"""
        
        if not self.config.use_augmentation:
            return A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711]),
                ToTensorV2()
            ])
        
        return A.Compose([
            # Geometric transforms
            A.RandomResizedCrop(
                self.config.image_size,
                self.config.image_size,
                scale=(0.8, 1.0),
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.3
            ),
            
            # Color transforms
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.RandomBrightnessContrast(p=0.3),
            
            # Quality transforms
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.2),
            
            # Final resize and normalize
            A.Resize(self.config.image_size, self.config.image_size),
            A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                       std=[0.26862954, 0.26130258, 0.27577711]),
            ToTensorV2()
        ])
    
    def preprocess_dataset(self, 
                          dataset: List[Dict],
                          dataset_name: str) -> str:
        """Preprocess and save dataset"""
        
        self.logger.info(f"Preprocessing {dataset_name} dataset")
        
        # Create output directory
        output_dir = Path(self.config.output_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into train/val/test
        splits = self.create_splits(dataset)
        
        # Process each split
        for split_name, split_data in splits.items():
            self.logger.info(f"Processing {split_name} split: {len(split_data)} samples")
            
            # Save processed data
            output_file = output_dir / f"{split_name}.json"
            with open(output_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            
            self.logger.info(f"Saved {split_name} to {output_file}")
        
        # Create metadata file
        metadata = {
            'dataset_name': dataset_name,
            'num_samples': len(dataset),
            'splits': {k: len(v) for k, v in splits.items()},
            'config': {
                'image_size': self.config.image_size,
                'max_text_length': self.config.max_text_length,
                'clip_model': self.config.clip_model
            },
            'preprocessing_date': str(pd.Timestamp.now())
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Preprocessing complete. Data saved to {output_dir}")
        return str(output_dir)
    
    def create_splits(self, 
                     dataset: List[Dict],
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> Dict[str, List[Dict]]:
        """Split dataset into train/val/test"""
        
        # Shuffle dataset
        import random
        random.seed(42)
        dataset = dataset.copy()
        random.shuffle(dataset)
        
        # Calculate split sizes
        n_total = len(dataset)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Create splits
        splits = {
            'train': dataset[:n_train],
            'val': dataset[n_train:n_train + n_val],
            'test': dataset[n_train + n_val:]
        }
        
        return splits


# ============================================================================
# Main Data Preparation Pipeline
# ============================================================================

class DataPreparer:
    """Main data preparation pipeline"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = setup_logger("DataPreparer")
        
        # Initialize components
        self.validator = DataValidator(config)
        self.preprocessor = DataPreprocessor(config)
        
        if config.cache_embeddings:
            self.cache = EmbeddingCache(config)
        else:
            self.cache = None
    
    def prepare_dataset(self, dataset_name: str) -> Dict[str, str]:
        """Prepare single dataset"""
        
        self.logger.info(f"Preparing {dataset_name} dataset")
        
        # Get dataset path
        dataset_path = Path(self.config.data_root) / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Validate dataset
        validation_result = self.validator.validate_dataset(dataset_path, dataset_name)
        valid_samples = validation_result['valid_samples']
        
        if not valid_samples:
            self.logger.error(f"No valid samples found in {dataset_name}")
            return {}
        
        # Preprocess dataset
        processed_dir = self.preprocessor.preprocess_dataset(valid_samples, dataset_name)
        
        # Cache embeddings if requested
        cache_files = {}
        if self.config.cache_embeddings and self.cache:
            for split in ['train', 'val', 'test']:
                split_file = Path(processed_dir) / f"{split}.json"
                if split_file.exists():
                    with open(split_file, 'r') as f:
                        split_data = json.load(f)
                    
                    if split_data:
                        cache_file = self.cache.cache_dataset(
                            split_data,
                            f"{dataset_name}_{split}"
                        )
                        cache_files[split] = cache_file
        
        # Print statistics
        self.validator.print_stats()
        
        return {
            'processed_dir': processed_dir,
            'cache_files': cache_files,
            'num_samples': len(valid_samples)
        }
    
    def prepare_all_datasets(self, dataset_names: List[str]) -> Dict:
        """Prepare multiple datasets"""
        
        results = {}
        
        for dataset_name in dataset_names:
            try:
                result = self.prepare_dataset(dataset_name)
                results[dataset_name] = result
                
            except Exception as e:
                self.logger.error(f"Failed to prepare {dataset_name}: {e}")
                results[dataset_name] = {'error': str(e)}
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict):
        """Print preparation summary"""
        
        print("\n" + "="*60)
        print("Data Preparation Summary")
        print("="*60)
        
        for dataset_name, result in results.items():
            print(f"\n{dataset_name}:")
            
            if 'error' in result:
                print(f"   Error: {result['error']}")
            else:
                print(f"   Processed: {result.get('num_samples', 0)} samples")
                print(f"   Location: {result.get('processed_dir', 'N/A')}")
                
                if result.get('cache_files'):
                    print("   Cached embeddings:")
                    for split, cache_file in result['cache_files'].items():
                        size_mb = Path(cache_file).stat().st_size / (1024**2)
                        print(f"    - {split}: {cache_file} ({size_mb:.2f} MB)")
        
        print("\n" + "="*60)
        print("Next Steps:")
        print("  1. Review processed data in ./processed_data/")
        print("  2. Check cached embeddings in ./cache/embeddings/")
        print("  3. Run training: python train.py --dataset <dataset_name>")
        print("="*60)


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare and preprocess datasets for FrEVL"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        choices=["all", "vqa", "coco", "snli-ve", "custom"],
        help="Datasets to prepare"
    )
    
    # Paths
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory containing raw datasets"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory for cached embeddings"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./processed_data",
        help="Directory for processed datasets"
    )
    
    # Processing options
    parser.add_argument(
        "--cache-embeddings",
        action="store_true",
        help="Cache CLIP embeddings for faster training"
    )
    parser.add_argument(
        "--cache-format",
        type=str,
        default="hdf5",
        choices=["hdf5", "npz", "pt"],
        help="Format for cached embeddings"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding extraction"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of parallel workers"
    )
    
    # CLIP options
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50"],
        help="CLIP model for embeddings"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for embedding extraction"
    )
    
    # Validation options
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate images and texts"
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=32,
        help="Minimum image size"
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=5.0,
        help="Maximum aspect ratio"
    )
    
    # Other options
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of existing caches"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create configuration
    config = DataConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        cache_embeddings=args.cache_embeddings,
        cache_format=args.cache_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        clip_model=args.clip_model,
        device=args.device,
        validate_images=args.validate,
        min_image_size=args.min_image_size,
        max_aspect_ratio=args.max_aspect_ratio
    )
    
    if args.force_rebuild:
        config.force_rebuild = True
    
    # Prepare datasets
    if "all" in args.dataset:
        datasets = ["vqa", "coco", "snli-ve"]
    else:
        datasets = args.dataset
    
    # Initialize preparer
    preparer = DataPreparer(config)
    
    # Prepare all datasets
    results = preparer.prepare_all_datasets(datasets)
    
    print("\n Data preparation complete!")


if __name__ == "__main__":
    main()