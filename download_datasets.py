#!/usr/bin/env python3
"""
Download datasets for FrEVL training and evaluation
Supports VQA v2, SNLI-VE, MS-COCO, and more
"""

import os
import argparse
import json
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional, Dict
import hashlib
import shutil

import requests
from tqdm import tqdm
import gdown
from huggingface_hub import hf_hub_download, snapshot_download


# Dataset URLs and metadata
DATASETS = {
    "vqa": {
        "train_questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        "val_questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        "test_questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
        "train_annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        "val_annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        "size": "2.5GB",
        "description": "Visual Question Answering v2.0 dataset"
    },
    "coco": {
        "train_images": "http://images.cocodataset.org/zips/train2014.zip",
        "val_images": "http://images.cocodataset.org/zips/val2014.zip",
        "test_images": "http://images.cocodataset.org/zips/test2014.zip",
        "train_annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        "size": "20GB",
        "description": "MS-COCO 2014 images and captions"
    },
    "snli-ve": {
        "annotations": "https://github.com/necla-ml/SNLI-VE/archive/refs/heads/master.zip",
        "flickr30k_notice": "Flickr30k images must be downloaded separately from https://shannon.cs.illinois.edu/DenotationGraph/",
        "size": "15GB",
        "description": "SNLI Visual Entailment dataset"
    },
    "conceptual-captions": {
        "train_tsv": "https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv",
        "validation_tsv": "https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv",
        "size": "300GB",
        "description": "Conceptual Captions dataset (URLs only, images need to be downloaded)"
    },
    "visual-genome": {
        "images_part1": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
        "images_part2": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
        "annotations": "http://visualgenome.org/static/data/dataset/objects.json.zip",
        "size": "15GB",
        "description": "Visual Genome dataset with dense annotations"
    }
}

# Checksums for verification
CHECKSUMS = {
    "vqa_train_questions": "443797f5b45c37873c6ad68dd1885e3a",
    "vqa_val_questions": "6761dd1a842b1d5e7e4e49bebad78470",
    "coco_train_images": "0da8c0bd3d6becc4dcb32757491aca88",
    "coco_val_images": "a3d79f5ed8d289b7a7554ce06a5782b3",
}


class DatasetDownloader:
    """Download and prepare datasets for FrEVL"""
    
    def __init__(self, data_dir: str = "./data", cache_dir: str = "./cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(
        self,
        url: str,
        output_path: Path,
        chunk_size: int = 8192,
        resume: bool = True,
        verify_checksum: Optional[str] = None
    ) -> bool:
        """
        Download file with progress bar and resume support
        
        Args:
            url: URL to download from
            output_path: Path to save file
            chunk_size: Download chunk size
            resume: Whether to resume partial downloads
            verify_checksum: Expected MD5 checksum
        
        Returns:
            True if download successful
        """
        
        # Check if file already exists and is complete
        if output_path.exists():
            if verify_checksum:
                if self.verify_checksum(output_path, verify_checksum):
                    print(f" File already exists and verified: {output_path}")
                    return True
                else:
                    print(f" File exists but checksum mismatch, re-downloading: {output_path}")
                    output_path.unlink()
            else:
                print(f" File already exists: {output_path}")
                return True
        
        # Setup headers for resume
        headers = {}
        mode = 'wb'
        initial_pos = 0
        
        if resume and output_path.exists():
            initial_pos = output_path.stat().st_size
            headers = {'Range': f'bytes={initial_pos}-'}
            mode = 'ab'
        
        # Download with progress bar
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=initial_pos,
                    unit='B',
                    unit_scale=True,
                    desc=output_path.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Verify checksum if provided
            if verify_checksum:
                if not self.verify_checksum(output_path, verify_checksum):
                    print(f"✗ Checksum verification failed for {output_path}")
                    return False
            
            print(f" Successfully downloaded: {output_path}")
            return True
            
        except Exception as e:
            print(f" Error downloading {url}: {e}")
            return False
    
    def verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify MD5 checksum of file"""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        
        actual_checksum = md5_hash.hexdigest()
        return actual_checksum == expected_checksum
    
    def extract_archive(self, archive_path: Path, extract_to: Path):
        """Extract zip or tar archive"""
        print(f"Extracting {archive_path.name}...")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unknown archive format: {archive_path.suffix}")
            return
        
        print(f"✓ Extracted to {extract_to}")
    
    def download_vqa(self):
        """Download VQA v2 dataset"""
        print("\n" + "="*50)
        print("Downloading VQA v2 Dataset")
        print("="*50)
        
        vqa_dir = self.data_dir / "vqa" / "v2"
        vqa_dir.mkdir(parents=True, exist_ok=True)
        
        # Download questions and annotations
        for split in ["train", "val", "test"]:
            # Questions
            if f"{split}_questions" in DATASETS["vqa"]:
                url = DATASETS["vqa"][f"{split}_questions"]
                output_file = self.cache_dir / f"vqa_v2_questions_{split}.zip"
                
                if self.download_file(url, output_file):
                    self.extract_archive(output_file, vqa_dir)
            
            # Annotations (not available for test)
            if f"{split}_annotations" in DATASETS["vqa"]:
                url = DATASETS["vqa"][f"{split}_annotations"]
                output_file = self.cache_dir / f"vqa_v2_annotations_{split}.zip"
                
                if self.download_file(url, output_file):
                    self.extract_archive(output_file, vqa_dir)
        
        print("✓ VQA v2 dataset downloaded successfully")
    
    def download_coco(self):
        """Download MS-COCO dataset"""
        print("\n" + "="*50)
        print("Downloading MS-COCO Dataset")
        print("="*50)
        
        coco_dir = self.data_dir / "coco"
        coco_dir.mkdir(parents=True, exist_ok=True)
        
        # Download images
        for split in ["train", "val", "test"]:
            if f"{split}_images" in DATASETS["coco"]:
                url = DATASETS["coco"][f"{split}_images"]
                output_file = self.cache_dir / f"coco_{split}2014.zip"
                
                checksum = CHECKSUMS.get(f"coco_{split}_images")
                if self.download_file(url, output_file, verify_checksum=checksum):
                    self.extract_archive(output_file, coco_dir)
        
        # Download annotations
        if "train_annotations" in DATASETS["coco"]:
            url = DATASETS["coco"]["train_annotations"]
            output_file = self.cache_dir / "coco_annotations_trainval2014.zip"
            
            if self.download_file(url, output_file):
                self.extract_archive(output_file, coco_dir)
        
        print("✓ MS-COCO dataset downloaded successfully")
    
    def download_snli_ve(self):
        """Download SNLI-VE dataset"""
        print("\n" + "="*50)
        print("Downloading SNLI-VE Dataset")
        print("="*50)
        
        snli_dir = self.data_dir / "snli-ve"
        snli_dir.mkdir(parents=True, exist_ok=True)
        
        # Download annotations from GitHub
        url = DATASETS["snli-ve"]["annotations"]
        output_file = self.cache_dir / "snli_ve.zip"
        
        if self.download_file(url, output_file):
            self.extract_archive(output_file, snli_dir)
        
        print("\n" + "!"*50)
        print("IMPORTANT: Flickr30k images must be downloaded separately")
        print("Please visit: https://shannon.cs.illinois.edu/DenotationGraph/")
        print("After downloading, extract images to:", snli_dir / "flickr30k_images")
        print("!"*50)
        
        print("✓ SNLI-VE annotations downloaded successfully")
    
    def download_from_huggingface(self, repo_id: str, output_dir: Path):
        """Download dataset from HuggingFace Hub"""
        print(f"Downloading from HuggingFace: {repo_id}")
        
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=output_dir,
                repo_type="dataset",
                resume_download=True
            )
            print(f"✓ Downloaded {repo_id} to {output_dir}")
        except Exception as e:
            print(f"✗ Error downloading from HuggingFace: {e}")
    
    def prepare_embeddings_cache(self):
        """Pre-compute and cache CLIP embeddings for faster training"""
        print("\n" + "="*50)
        print("Preparing Embeddings Cache")
        print("="*50)
        
        cache_script = Path(__file__).parent / "cache_embeddings.py"
        
        if cache_script.exists():
            os.system(f"python {cache_script} --data-dir {self.data_dir} --cache-dir {self.cache_dir}")
        else:
            print("Cache script not found. Skipping embedding cache preparation.")
    
    def download_all(self, datasets: List[str], cache_embeddings: bool = False):
        """Download all specified datasets"""
        
        for dataset in datasets:
            if dataset == "vqa":
                self.download_vqa()
                self.download_coco()  # VQA requires COCO images
            elif dataset == "coco":
                self.download_coco()
            elif dataset == "snli-ve":
                self.download_snli_ve()
            else:
                print(f"Unknown dataset: {dataset}")
        
        if cache_embeddings:
            self.prepare_embeddings_cache()
        
        # Print summary
        print("\n" + "="*50)
        print("Download Summary")
        print("="*50)
        
        total_size = 0
        for dataset_dir in self.data_dir.iterdir():
            if dataset_dir.is_dir():
                size = sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file())
                size_gb = size / (1024**3)
                total_size += size_gb
                print(f"{dataset_dir.name}: {size_gb:.2f} GB")
        
        print(f"Total: {total_size:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for FrEVL")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "vqa", "coco", "snli-ve", "conceptual-captions", "visual-genome"],
        help="Datasets to download"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to save datasets"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory for download cache"
    )
    parser.add_argument(
        "--cache-embeddings",
        action="store_true",
        help="Pre-compute CLIP embeddings"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume for partial downloads"
    )
    
    args = parser.parse_args()
    
    # Prepare dataset list
    if "all" in args.dataset:
        datasets = ["vqa", "snli-ve"]
    else:
        datasets = args.dataset
    
    # Initialize downloader
    downloader = DatasetDownloader(args.data_dir, args.cache_dir)
    
    # Download datasets
    downloader.download_all(datasets, args.cache_embeddings)
    
    print("\n✓ All downloads complete!")
    print(f"Data saved to: {args.data_dir}")
    
    # Provide next steps
    print("\n" + "="*50)
    print("Next Steps")
    print("="*50)
    print("1. If you downloaded SNLI-VE, download Flickr30k images manually")
    print("2. Run training: python train.py --dataset vqa")
    print("3. Run evaluation: python evaluate.py --dataset all")
    print("4. Launch demo: python demo.py")


if __name__ == "__main__":
    main()
