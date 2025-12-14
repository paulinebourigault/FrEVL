"""
FrEVL Data Loaders
Dataset implementations for VQA, SNLI-VE, MS-COCO and more
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import pickle

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import h5py
import clip
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================================
# Base Dataset Class
# ============================================================================

class VisionLanguageDataset(Dataset):
    """Base class for vision-language datasets"""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Any] = None,
        text_transform: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        use_augmentation: bool = False
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.text_transform = text_transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_samples = max_samples
        self.use_augmentation = use_augmentation
        
        # Setup augmentations
        if use_augmentation and split == "train":
            self.augmentation = self._get_augmentations()
        else:
            self.augmentation = None
        
        # Load data
        self.data = self._load_data()
        
        # Limit samples if specified
        if max_samples:
            self.data = self.data[:max_samples]
        
        # Setup cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_data(self) -> List[Dict]:
        """Load dataset - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _get_augmentations(self) -> A.Compose:
        """Get augmentation pipeline"""
        return A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
            ToTensorV2()
        ])
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and cache image"""
        # Check cache
        if self.cache_dir:
            cache_path = self.cache_dir / f"{Path(image_path).stem}.pkl"
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Save to cache
        if self.cache_dir:
            with open(cache_path, 'wb') as f:
                pickle.dump(image, f)
        
        return image
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get item - to be implemented by subclasses"""
        raise NotImplementedError


# ============================================================================
# VQA v2 Dataset
# ============================================================================

class VQADataset(VisionLanguageDataset):
    """VQA v2 dataset"""
    
    ANSWER_VOCAB_SIZE = 3129
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        version: str = "v2",
        **kwargs
    ):
        self.version = version
        self.answer_to_idx = None
        self.idx_to_answer = None
        
        super().__init__(data_root, split, **kwargs)
    
    def _load_data(self) -> List[Dict]:
        """Load VQA v2 data"""
        data_dir = self.data_root / "vqa" / self.version
        
        # Load questions
        if self.split == "test":
            questions_file = data_dir / f"v2_OpenEnded_mscoco_{self.split}2015_questions.json"
        else:
            questions_file = data_dir / f"v2_OpenEnded_mscoco_{self.split}2014_questions.json"
        
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
        
        # Load annotations (not available for test)
        annotations = {}
        if self.split != "test":
            if self.split == "val":
                ann_file = data_dir / f"v2_mscoco_val2014_annotations.json"
            else:
                ann_file = data_dir / f"v2_mscoco_{self.split}2014_annotations.json"
            
            with open(ann_file, 'r') as f:
                ann_data = json.load(f)
            
            for ann in ann_data['annotations']:
                annotations[ann['question_id']] = ann
        
        # Load answer vocabulary
        vocab_file = data_dir / "answer_vocab.json"
        if vocab_file.exists():
            with open(vocab_file, 'r') as f:
                self.answer_to_idx = json.load(f)
                self.idx_to_answer = {v: k for k, v in self.answer_to_idx.items()}
        else:
            # Build vocabulary from training data
            if self.split == "train":
                self._build_answer_vocab(annotations)
        
        # Prepare data entries
        data = []
        image_dir = self.data_root / "coco" / f"{self.split}2014"
        
        for question in tqdm(questions_data['questions'], desc=f"Loading VQA {self.split}"):
            entry = {
                'question_id': question['question_id'],
                'image_id': question['image_id'],
                'question': question['question'],
                'image_path': image_dir / f"COCO_{self.split}2014_{question['image_id']:012d}.jpg"
            }
            
            # Add answer if available
            if question['question_id'] in annotations:
                ann = annotations[question['question_id']]
                
                # Get most frequent answer
                answer_counts = defaultdict(int)
                for answer in ann['answers']:
                    answer_counts[answer['answer']] += 1
                
                most_frequent = max(answer_counts, key=answer_counts.get)
                
                if self.answer_to_idx and most_frequent in self.answer_to_idx:
                    entry['answer'] = most_frequent
                    entry['answer_idx'] = self.answer_to_idx[most_frequent]
                    entry['answer_scores'] = self._compute_answer_scores(ann['answers'])
            
            data.append(entry)
        
        return data
    
    def _build_answer_vocab(self, annotations: Dict):
        """Build answer vocabulary from training data"""
        answer_counts = defaultdict(int)
        
        for ann in annotations.values():
            for answer in ann['answers']:
                answer_counts[answer['answer']] += 1
        
        # Keep top answers
        top_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
        top_answers = top_answers[:self.ANSWER_VOCAB_SIZE]
        
        self.answer_to_idx = {ans: idx for idx, (ans, _) in enumerate(top_answers)}
        self.idx_to_answer = {idx: ans for ans, idx in self.answer_to_idx.items()}
        
        # Save vocabulary
        vocab_file = self.data_root / "vqa" / self.version / "answer_vocab.json"
        with open(vocab_file, 'w') as f:
            json.dump(self.answer_to_idx, f)
    
    def _compute_answer_scores(self, answers: List[Dict]) -> torch.Tensor:
        """Compute soft answer scores for VQA accuracy"""
        answer_counts = defaultdict(int)
        for answer in answers:
            answer_counts[answer['answer']] += 1
        
        scores = torch.zeros(self.ANSWER_VOCAB_SIZE)
        for answer, count in answer_counts.items():
            if answer in self.answer_to_idx:
                idx = self.answer_to_idx[answer]
                scores[idx] = min(count / 3.0, 1.0)  # VQA accuracy formula
        
        return scores
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get VQA item"""
        entry = self.data[idx]
        
        # Load image
        image = self._load_image(entry['image_path'])
        
        # Apply transforms
        if self.augmentation:
            image = np.array(image)
            augmented = self.augmentation(image=image)
            image_tensor = augmented['image']
        elif self.transform:
            image_tensor = self.transform(image)
        else:
            # Default CLIP preprocessing
            preprocess = clip.load("ViT-B/32", device="cpu")[1]
            image_tensor = preprocess(image)
        
        # Process question
        question = entry['question']
        if self.text_transform:
            question = self.text_transform(question)
        
        # Get label
        if 'answer_idx' in entry:
            label = entry['answer_idx']
            scores = entry.get('answer_scores', None)
        else:
            label = 0  # Dummy label for test set
            scores = None
        
        if scores is not None:
            return image_tensor, question, label, scores
        else:
            return image_tensor, question, label


# ============================================================================
# SNLI-VE Dataset
# ============================================================================

class SNLIVEDataset(VisionLanguageDataset):
    """SNLI-VE: Visual Entailment Dataset"""
    
    LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        **kwargs
    ):
        super().__init__(data_root, split, **kwargs)
    
    def _load_data(self) -> List[Dict]:
        """Load SNLI-VE data"""
        data_dir = self.data_root / "snli-ve"
        
        # Load annotations
        if self.split == "test":
            ann_file = data_dir / "snli_ve_test.jsonl"
        elif self.split == "val":
            ann_file = data_dir / "snli_ve_dev.jsonl"
        else:
            ann_file = data_dir / "snli_ve_train.jsonl"
        
        data = []
        image_dir = data_dir / "flickr30k_images"
        
        with open(ann_file, 'r') as f:
            for line in tqdm(f, desc=f"Loading SNLI-VE {self.split}"):
                entry = json.loads(line)
                
                # Skip if gold label not available
                if entry['gold_label'] not in self.LABELS:
                    continue
                
                data.append({
                    'image_id': entry['Flickr30K_ID'],
                    'image_path': image_dir / f"{entry['Flickr30K_ID']}.jpg",
                    'hypothesis': entry['sentence2'],
                    'label': self.LABELS[entry['gold_label']],
                    'label_name': entry['gold_label']
                })
        
        return data
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get SNLI-VE item"""
        entry = self.data[idx]
        
        # Load image
        image = self._load_image(entry['image_path'])
        
        # Apply transforms
        if self.augmentation:
            image = np.array(image)
            augmented = self.augmentation(image=image)
            image_tensor = augmented['image']
        elif self.transform:
            image_tensor = self.transform(image)
        else:
            preprocess = clip.load("ViT-B/32", device="cpu")[1]
            image_tensor = preprocess(image)
        
        # Process hypothesis
        hypothesis = entry['hypothesis']
        if self.text_transform:
            hypothesis = self.text_transform(hypothesis)
        
        label = entry['label']
        
        return image_tensor, hypothesis, label


# ============================================================================
# MS-COCO Retrieval Dataset
# ============================================================================

class COCODataset(VisionLanguageDataset):
    """MS-COCO dataset for image-text retrieval"""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        captions_per_image: int = 5,
        **kwargs
    ):
        self.captions_per_image = captions_per_image
        super().__init__(data_root, split, **kwargs)
    
    def _load_data(self) -> List[Dict]:
        """Load MS-COCO data"""
        data_dir = self.data_root / "coco"
        
        # Load captions
        if self.split == "test":
            ann_file = data_dir / "annotations" / "image_info_test2014.json"
        elif self.split == "val":
            ann_file = data_dir / "annotations" / "captions_val2014.json"
        else:
            ann_file = data_dir / "annotations" / "captions_train2014.json"
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Group captions by image
        image_captions = defaultdict(list)
        
        if 'annotations' in coco_data:
            for ann in coco_data['annotations']:
                image_captions[ann['image_id']].append(ann['caption'])
        
        # Prepare data entries
        data = []
        image_dir = data_dir / f"{self.split}2014"
        
        for img_info in tqdm(coco_data['images'], desc=f"Loading COCO {self.split}"):
            image_id = img_info['id']
            
            if image_id in image_captions:
                captions = image_captions[image_id][:self.captions_per_image]
            else:
                captions = [""]  # Empty caption for test images
            
            for caption in captions:
                data.append({
                    'image_id': image_id,
                    'image_path': image_dir / img_info['file_name'],
                    'caption': caption,
                    'all_captions': captions
                })
        
        return data
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get COCO item"""
        entry = self.data[idx]
        
        # Load image
        image = self._load_image(entry['image_path'])
        
        # Apply transforms
        if self.augmentation:
            image = np.array(image)
            augmented = self.augmentation(image=image)
            image_tensor = augmented['image']
        elif self.transform:
            image_tensor = self.transform(image)
        else:
            preprocess = clip.load("ViT-B/32", device="cpu")[1]
            image_tensor = preprocess(image)
        
        # Process caption
        caption = entry['caption']
        if self.text_transform:
            caption = self.text_transform(caption)
        
        # For retrieval, we don't have explicit labels
        # Return image_id as label for evaluation purposes
        label = entry['image_id']
        
        return image_tensor, caption, label


# ============================================================================
# Custom Dataset for Fine-tuning
# ============================================================================

class CustomVLDataset(VisionLanguageDataset):
    """Custom vision-language dataset for fine-tuning"""
    
    def __init__(
        self,
        data_file: str,
        image_dir: str,
        split: str = "train",
        **kwargs
    ):
        self.data_file = Path(data_file)
        self.image_dir = Path(image_dir)
        super().__init__(image_dir, split, **kwargs)
    
    def _load_data(self) -> List[Dict]:
        """Load custom dataset from JSON or CSV"""
        
        if self.data_file.suffix == '.json':
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        elif self.data_file.suffix == '.csv':
            df = pd.read_csv(self.data_file)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {self.data_file.suffix}")
        
        # Process data entries
        processed_data = []
        for entry in data:
            if 'split' in entry and entry['split'] != self.split:
                continue
            
            processed_data.append({
                'image_path': self.image_dir / entry['image'],
                'text': entry.get('text', entry.get('caption', entry.get('question', ''))),
                'label': entry.get('label', 0),
                'metadata': entry
            })
        
        return processed_data
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get custom dataset item"""
        entry = self.data[idx]
        
        # Load image
        image = self._load_image(entry['image_path'])
        
        # Apply transforms
        if self.transform:
            image_tensor = self.transform(image)
        else:
            preprocess = clip.load("ViT-B/32", device="cpu")[1]
            image_tensor = preprocess(image)
        
        # Process text
        text = entry['text']
        if self.text_transform:
            text = self.text_transform(text)
        
        label = entry['label']
        
        return image_tensor, text, label, entry['metadata']


# ============================================================================
# DataLoader Factory
# ============================================================================

def create_dataloader(
    dataset_name: str,
    data_root: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
    **dataset_kwargs
) -> DataLoader:
    """
    Create dataloader for specified dataset
    
    Args:
        dataset_name: Name of dataset (vqa, snli-ve, coco, custom)
        data_root: Root directory for data
        split: Dataset split (train, val, test)
        batch_size: Batch size
        num_workers: Number of worker processes
        distributed: Whether to use distributed sampler
        **dataset_kwargs: Additional arguments for dataset
    
    Returns:
        DataLoader instance
    """
    
    # Create dataset
    if dataset_name.lower() == "vqa":
        dataset = VQADataset(data_root, split, **dataset_kwargs)
    elif dataset_name.lower() == "snli-ve":
        dataset = SNLIVEDataset(data_root, split, **dataset_kwargs)
    elif dataset_name.lower() == "coco":
        dataset = COCODataset(data_root, split, **dataset_kwargs)
    elif dataset_name.lower() == "custom":
        dataset = CustomVLDataset(data_root=data_root, split=split, **dataset_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create sampler
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=(split == "train"))
    else:
        sampler = None
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0)
    )
    
    return dataloader


# ============================================================================
# Collate Functions
# ============================================================================

def vqa_collate_fn(batch: List[Tuple]) -> Tuple:
    """Custom collate function for VQA dataset"""
    images = torch.stack([item[0] for item in batch])
    questions = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch])
    
    if len(batch[0]) > 3:
        scores = torch.stack([item[3] for item in batch])
        return images, questions, labels, scores
    
    return images, questions, labels


def retrieval_collate_fn(batch: List[Tuple]) -> Tuple:
    """Custom collate function for retrieval dataset"""
    images = torch.stack([item[0] for item in batch])
    texts = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch])
    
    return images, texts, labels
