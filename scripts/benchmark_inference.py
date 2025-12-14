#!/usr/bin/env python3
"""
Benchmark FrEVL inference performance
Measures latency, throughput, and resource usage
"""

import argparse
import json
import time
import gc
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import GPUtil

from model import FrEVL, FrEVLConfig
from data_loader import create_dataloader
from utils import set_random_seed, timer


class InferenceBenchmark:
    """Comprehensive inference benchmarking suite"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        compile_model: bool = False
    ):
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.compile_model = compile_model
        
        # Load model
        self.model = self._load_model()
        
        # Metrics storage
        self.metrics = {
            "latency": [],
            "throughput": [],
            "memory": [],
            "gpu_utilization": []
        }
        
        # System info
        self.system_info = self._get_system_info()
    
    def _load_model(self) -> FrEVL:
        """Load and prepare model for benchmarking"""
        print(f"Loading model from {self.model_path}...")
        
        model = FrEVL.from_pretrained(self.model_path)
        model.to(self.device)
        model.eval()
        
        # Compile model if requested (PyTorch 2.0+)
        if self.compile_model and torch.__version__ >= "2.0":
            print("Compiling model with TorchScript...")
            model = torch.compile(model, mode="max-autotune")
        
        return model
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            "cpu": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "cuda_capability": torch.cuda.get_device_capability(0),
            })
        
        return info
    
    def benchmark_latency(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark inference latency for different batch sizes
        
        Args:
            batch_sizes: List of batch sizes to test
            num_iterations: Number of iterations per batch size
            warmup: Number of warmup iterations
        
        Returns:
            Dictionary with latency statistics
        """
        print("\n" + "="*50)
        print("Latency Benchmark")
        print("="*50)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            # Create dummy input
            images = torch.randn(batch_size, 3, 224, 224).to(self.device)
            texts = ["Sample question"] * batch_size
            
            latencies = []
            
            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = self.model(images=images, text=texts)
            
            # Synchronize
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            # Benchmark
            for _ in tqdm(range(num_iterations), desc="Measuring"):
                # Clear cache
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                
                with torch.no_grad():
                    _ = self.model(images=images, text=texts)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms
            
            # Calculate statistics
            latencies = np.array(latencies)
            results[f"batch_{batch_size}"] = {
                "mean_ms": float(np.mean(latencies)),
                "std_ms": float(np.std(latencies)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "samples_per_second": float(batch_size * 1000 / np.mean(latencies))
            }
            
            print(f"  Mean latency: {results[f'batch_{batch_size}']['mean_ms']:.2f} ms")
            print(f"  P99 latency: {results[f'batch_{batch_size}']['p99_ms']:.2f} ms")
            print(f"  Throughput: {results[f'batch_{batch_size}']['samples_per_second']:.1f} samples/s")
        
        return results
    
    def benchmark_throughput(
        self,
        dataloader: DataLoader,
        max_batches: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark maximum throughput
        
        Args:
            dataloader: DataLoader for testing
            max_batches: Maximum number of batches to process
        
        Returns:
            Throughput statistics
        """
        print("\n" + "="*50)
        print("Throughput Benchmark")
        print("="*50)
        
        total_samples = 0
        total_time = 0
        
        # Warmup
        warmup_batches = min(10, max_batches // 10)
        for i, batch in enumerate(dataloader):
            if i >= warmup_batches:
                break
            
            images = batch[0].to(self.device)
            texts = batch[1]
            
            with torch.no_grad():
                _ = self.model(images=images, text=texts)
        
        # Synchronize
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        
        for i, batch in enumerate(tqdm(dataloader, desc="Processing", total=max_batches)):
            if i >= max_batches:
                break
            
            images = batch[0].to(self.device)
            texts = batch[1]
            batch_size = images.size(0)
            
            with torch.no_grad():
                _ = self.model(images=images, text=texts)
            
            total_samples += batch_size
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        throughput = total_samples / total_time
        
        results = {
            "total_samples": total_samples,
            "total_time_seconds": total_time,
            "samples_per_second": throughput,
            "batches_processed": min(i + 1, max_batches),
            "average_batch_time_ms": (total_time / min(i + 1, max_batches)) * 1000
        }
        
        print(f"\nThroughput: {throughput:.1f} samples/second")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Samples processed: {total_samples}")
        
        return results
    
    def benchmark_memory(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]
    ) -> Dict[str, Any]:
        """
        Benchmark memory usage for different batch sizes
        
        Args:
            batch_sizes: List of batch sizes to test
        
        Returns:
            Memory usage statistics
        """
        print("\n" + "="*50)
        print("Memory Benchmark")
        print("="*50)
        
        if self.device.type != "cuda":
            print("Memory benchmark only available for CUDA devices")
            return {}
        
        results = {}
        
        for batch_size in batch_sizes:
            # Clear cache and reset stats
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create input
            images = torch.randn(batch_size, 3, 224, 224).to(self.device)
            texts = ["Sample question"] * batch_size
            
            # Measure initial memory
            initial_memory = torch.cuda.memory_allocated()
            
            # Run inference
            with torch.no_grad():
                _ = self.model(images=images, text=texts)
            
            # Get memory stats
            current_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            results[f"batch_{batch_size}"] = {
                "initial_mb": initial_memory / (1024**2),
                "current_mb": current_memory / (1024**2),
                "peak_mb": peak_memory / (1024**2),
                "allocated_mb": (current_memory - initial_memory) / (1024**2),
                "peak_allocated_mb": (peak_memory - initial_memory) / (1024**2)
            }
            
            print(f"\nBatch size {batch_size}:")
            print(f"  Peak memory: {results[f'batch_{batch_size}']['peak_mb']:.2f} MB")
            print(f"  Allocated: {results[f'batch_{batch_size}']['peak_allocated_mb']:.2f} MB")
            
            # Clear for next iteration
            del images, texts
            torch.cuda.empty_cache()
        
        return results
    
    def benchmark_optimization_comparison(self) -> Dict[str, Any]:
        """Compare different optimization techniques"""
        print("\n" + "="*50)
        print("Optimization Comparison")
        print("="*50)
        
        batch_size = 8
        num_iterations = 50
        
        results = {}
        
        # Test configurations
        configs = {
            "baseline": {},
            "fp16": {"use_fp16": True},
            "int8": {"use_int8": True},
            "torch_compile": {"compile": True},
            "onnx": {"use_onnx": True}
        }
        
        for name, config in configs.items():
            print(f"\nTesting {name}...")
            
            # Apply optimization
            model = self._apply_optimization(self.model, config)
            
            # Benchmark
            latencies = []
            images = torch.randn(batch_size, 3, 224, 224).to(self.device)
            texts = ["Sample"] * batch_size
            
            for _ in range(num_iterations):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                
                with torch.no_grad():
                    if config.get("use_fp16"):
                        from torch.cuda.amp import autocast
                        with autocast():
                            _ = model(images=images, text=texts)
                    else:
                        _ = model(images=images, text=texts)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            
            results[name] = {
                "mean_ms": float(np.mean(latencies)),
                "std_ms": float(np.std(latencies)),
                "speedup": 1.0  # Will calculate relative to baseline
            }
            
            print(f"  Mean latency: {results[name]['mean_ms']:.2f} ms")
        
        # Calculate speedups
        baseline_latency = results["baseline"]["mean_ms"]
        for name in results:
            results[name]["speedup"] = baseline_latency / results[name]["mean_ms"]
            print(f"\n{name} speedup: {results[name]['speedup']:.2f}x")
        
        return results
    
    def _apply_optimization(self, model: FrEVL, config: Dict) -> nn.Module:
        """Apply optimization to model"""
        
        if config.get("use_int8"):
            # Dynamic quantization
            model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
        elif config.get("compile") and torch.__version__ >= "2.0":
            model = torch.compile(model)
        elif config.get("use_onnx"):
            # Note: ONNX export/inference would need additional implementation
            pass
        
        return model
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Create visualizations of benchmark results"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Latency vs Batch Size
        if "latency" in results:
            ax = axes[0, 0]
            batch_sizes = []
            mean_latencies = []
            p99_latencies = []
            
            for key in sorted(results["latency"].keys()):
                batch_size = int(key.split("_")[1])
                batch_sizes.append(batch_size)
                mean_latencies.append(results["latency"][key]["mean_ms"])
                p99_latencies.append(results["latency"][key]["p99_ms"])
            
            ax.plot(batch_sizes, mean_latencies, 'o-', label='Mean', linewidth=2)
            ax.plot(batch_sizes, p99_latencies, 's-', label='P99', linewidth=2)
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Inference Latency vs Batch Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
        
        # 2. Throughput vs Batch Size
        if "latency" in results:
            ax = axes[0, 1]
            throughputs = []
            
            for key in sorted(results["latency"].keys()):
                throughputs.append(results["latency"][key]["samples_per_second"])
            
            ax.plot(batch_sizes, throughputs, 'o-', color='green', linewidth=2)
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (samples/sec)')
            ax.set_title('Throughput vs Batch Size')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
        
        # 3. Memory Usage
        if "memory" in results:
            ax = axes[1, 0]
            peak_memories = []
            
            for key in sorted(results["memory"].keys()):
                peak_memories.append(results["memory"][key]["peak_mb"])
            
            ax.bar(range(len(batch_sizes)), peak_memories, color='coral')
            ax.set_xticks(range(len(batch_sizes)))
            ax.set_xticklabels(batch_sizes)
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Peak Memory (MB)')
            ax.set_title('Memory Usage vs Batch Size')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Optimization Comparison
        if "optimization" in results:
            ax = axes[1, 1]
            optimizations = list(results["optimization"].keys())
            speedups = [results["optimization"][opt]["speedup"] for opt in optimizations]
            
            bars = ax.bar(range(len(optimizations)), speedups, color='skyblue')
            ax.set_xticks(range(len(optimizations)))
            ax.set_xticklabels(optimizations, rotation=45)
            ax.set_ylabel('Speedup')
            ax.set_title('Optimization Technique Comparison')
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, speedup in zip(bars, speedups):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{speedup:.2f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"\nPlots saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def run_complete_benchmark(
        self,
        dataloader: Optional[DataLoader] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        
        print("\n" + "="*60)
        print("FrEVL Inference Benchmark Suite")
        print("="*60)
        print("\nSystem Information:")
        for key, value in self.system_info.items():
            print(f"  {key}: {value}")
        
        # Run benchmarks
        results = {
            "system_info": self.system_info,
            "timestamp": datetime.now().isoformat(),
            "model_path": str(self.model_path)
        }
        
        # 1. Latency benchmark
        results["latency"] = self.benchmark_latency()
        
        # 2. Memory benchmark
        if self.device.type == "cuda":
            results["memory"] = self.benchmark_memory()
        
        # 3. Throughput benchmark
        if dataloader:
            results["throughput"] = self.benchmark_throughput(dataloader)
        
        # 4. Optimization comparison
        results["optimization"] = self.benchmark_optimization_comparison()
        
        # Save results
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON
            json_path = save_dir / f"benchmark_results_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {json_path}")
            
            # Save plots
            plot_path = save_dir / f"benchmark_plots_{datetime.now():%Y%m%d_%H%M%S}.png"
            self.plot_results(results, plot_path)
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict):
        """Print benchmark summary"""
        
        print("\n" + "="*60)
        print("Benchmark Summary")
        print("="*60)
        
        # Best latency
        if "latency" in results:
            best_single = results["latency"]["batch_1"]["mean_ms"]
            best_batch = min(
                results["latency"].values(),
                key=lambda x: x["mean_ms"] / int(x.get("samples_per_second", 1))
            )
            print(f"\nLatency:")
            print(f"  Single sample: {best_single:.2f} ms")
            print(f"  Best efficiency: {best_batch['mean_ms']:.2f} ms")
        
        # Throughput
        if "throughput" in results:
            print(f"\nThroughput:")
            print(f"  Maximum: {results['throughput']['samples_per_second']:.1f} samples/sec")
        
        # Memory
        if "memory" in results:
            print(f"\nMemory:")
            peak = max(v["peak_mb"] for v in results["memory"].values())
            print(f"  Peak usage: {peak:.1f} MB")
        
        # Best optimization
        if "optimization" in results:
            best_opt = max(
                results["optimization"].items(),
                key=lambda x: x[1]["speedup"]
            )
            print(f"\nBest Optimization:")
            print(f"  {best_opt[0]}: {best_opt[1]['speedup']:.2f}x speedup")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FrEVL inference")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="vqa",
                       help="Dataset for throughput benchmark")
    parser.add_argument("--data-root", type=str, default="./data",
                       help="Data root directory")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for throughput test")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--compile", action="store_true",
                       help="Compile model with TorchScript")
    parser.add_argument("--output", type=str, default="./benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_random_seed(args.seed)
    
    # Create dataloader for throughput test
    dataloader = create_dataloader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=False
    )
    
    # Initialize benchmark
    benchmark = InferenceBenchmark(
        model_path=args.model,
        device=args.device,
        compile_model=args.compile
    )
    
    # Run benchmarks
    results = benchmark.run_complete_benchmark(
        dataloader=dataloader,
        save_dir=args.output
    )
    
    print("\n Benchmarking complete!")


if __name__ == "__main__":
    main()
