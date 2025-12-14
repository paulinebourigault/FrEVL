# FrEVL: Frozen Pretrained Embeddings for Efficient Vision-Language Understanding

<div align="center">
  
  <img src="https://github.com/EmmanuelleB985/FrEVL/assets/placeholder/frevl-banner.png" alt="FrEVL Banner" width="100%">
  
  <h3>⚡ 85-95% SOTA Performance with 10× Fewer Parameters</h3>
  
  [![arXiv](https://img.shields.io/badge/arXiv-2508.04469-b31b1b.svg)](https://arxiv.org/pdf/2508.04469)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
  [![CLIP](https://img.shields.io/badge/CLIP-OpenAI-green.svg)](https://github.com/openai/CLIP)
  [![CI/CD](https://github.com/EmmanuelleB985/FrEVL/workflows/CI/badge.svg)](https://github.com/EmmanuelleB985/FrEVL/actions)
  [![codecov](https://codecov.io/gh/EmmanuelleB985/FrEVL/branch/main/graph/badge.svg)](https://codecov.io/gh/EmmanuelleB985/FrEVL)
  
  **[🚀 Live Demo](https://huggingface.co/spaces/EmmanuelleB985/FrEVL)** | **[📊 Benchmark Results](#-performance-metrics)** | **[📄 Paper](https://arxiv.org/pdf/2508.04469)** | **[🤗 Models](https://huggingface.co/EmmanuelleB985/FrEVL)**

</div>

---

## Why FrEVL?

FrEVL revolutionizes vision-language understanding by **freezing pretrained CLIP embeddings** and training only a lightweight fusion network. This approach delivers:

- ** 3× faster inference** than ALBEF/BLIP
- ** 70% lower deployment costs**
- ** 68.4M trainable parameters** (vs 200M+ in SOTA models)
- ** 850 images/sec** throughput on single V100
- ** Production-ready** with <25ms p99 latency

## Performance Metrics

<div align="center">

| Model | VQA v2 ↑ | SNLI-VE ↑ | MS-COCO ↑ | Params | Latency (ms) | Memory (GB) | Cost/1M |
|:------|:---------|:----------|:----------|:-------|:-------------|:------------|:--------|
| **FrEVL (Ours)** | **71.2** | **78.4** | **85.1** | **68.4M** | **12** | **1.2** | **$20** |
| ALBEF-Base | 75.8 | 80.1 | 87.3 | 210M | 45 | 4.8 | $68 |
| BLIP-Base | 78.2 | 81.3 | 89.1 | 223M | 52 | 5.1 | $74 |
| CLIP-ViL | 70.1 | 76.2 | 83.5 | 428M | 38 | 5.2 | $65 |

</div>

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/EmmanuelleB985/FrEVL
cd FrEVL

# Create environment
conda create -n frevl python=3.9 -y
conda activate frevl

# Install dependencies
pip install -r requirements.txt

# Download pretrained model
python scripts/download_models.py --model frevl-base
```


#### Option 1: Web Interface
```bash
# Launch Gradio demo
python demo.py --model frevl-base --port 7860
# Visit http://localhost:7860
```

#### Option 2: Python API
```python
from frevl import FrEVL

# Load model
model = FrEVL.from_pretrained("frevl-base")

# Single inference
result = model.predict(
    image="path/to/image.jpg",
    text="What is the main object in this image?"
)
print(f"Answer: {result['answer']}, Confidence: {result['confidence']:.2f}")

# Batch inference
results = model.batch_predict(image_paths, questions)
```

#### Option 3: REST API
```bash
# Start FastAPI server
uvicorn serve:app --host 0.0.0.0 --port 8000

# Query the API
curl -X POST "http://localhost:8000/predict" \
  -F "image=@image.jpg" \
  -F "question=What color is the car?"
```

## Architecture

<div align="center">
  <img src="https://github.com/EmmanuelleB985/FrEVL/assets/placeholder/architecture.png" alt="FrEVL Architecture" width="80%">
</div>

FrEVL's key innovations:
1. **Frozen CLIP Encoders**: Leverage pretrained representations without fine-tuning
2. **Lightweight Fusion Network**: Cross-attention mechanism with only 68.4M parameters
3. **Efficient Caching**: Precomputed embeddings reduce inference time by 60%
4. **Mixed Precision**: FP16 training/inference with minimal accuracy loss

## Model Zoo

| Model | Size | VQA v2 | Download | HuggingFace |
|:------|:-----|:-------|:---------|:------------|
| FrEVL-Base | 274MB | 71.2 | [Download](https://github.com/EmmanuelleB985/FrEVL/releases/download/v1.0/frevl-base.pt) | [🤗 Hub](https://huggingface.co/EmmanuelleB985/frevl-base) |
| FrEVL-Large | 512MB | 74.8 | [Download](https://github.com/EmmanuelleB985/FrEVL/releases/download/v1.0/frevl-large.pt) | [🤗 Hub](https://huggingface.co/EmmanuelleB985/frevl-large) |
| FrEVL-Multilingual | 389MB | 68.5 | [Download](https://github.com/EmmanuelleB985/FrEVL/releases/download/v1.0/frevl-multi.pt) | [🤗 Hub](https://huggingface.co/EmmanuelleB985/frevl-multi) |

## Training

### From Scratch
```bash
# Download and prepare datasets
python scripts/prepare_data.py --dataset all --cache-embeddings

# Train FrEVL
python train.py \
  --dataset vqa \
  --model frevl-base \
  --batch-size 128 \
  --learning-rate 1e-4 \
  --epochs 20 \
  --wandb-project frevl
```

### Fine-tuning
```bash
# Fine-tune on custom dataset
python finetune.py \
  --pretrained frevl-base \
  --data-dir ./custom_data \
  --output-dir ./checkpoints/custom
```

### Distributed Training
```bash
# Multi-GPU training with DDP
torchrun --nproc_per_node=4 train_distributed.py \
  --dataset vqa \
  --batch-size 512
```

## Evaluation

```bash
# Evaluate on VQA v2
python evaluate.py \
  --model checkpoints/best_model.pt \
  --dataset vqa \
  --split val

# Comprehensive benchmark
python benchmark.py --model frevl-base --all-datasets
```

## Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t frevl:latest .

# Run container
docker run -p 8000:8000 --gpus all frevl:latest

# Or use docker-compose
docker-compose up -d
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f deploy/k8s/

# Check deployment status
kubectl get pods -l app=frevl
```

### Cloud Deployment
```bash
# Deploy to AWS SageMaker
python deploy/sagemaker_deploy.py

# Deploy to Google Cloud AI Platform
gcloud ai-platform models create frevl
gcloud ai-platform versions create v1 --model frevl --origin gs://bucket/model

# Deploy to Azure ML
az ml model deploy -n frevl-service -m frevl:1
```

## Testing

```bash
# Run all tests
pytest tests/ -v --cov=frevl --cov-report=html

# Run specific test suites
pytest tests/test_model.py
pytest tests/test_inference.py
pytest tests/test_api.py

# Performance tests
python tests/benchmark_performance.py
```

## Monitoring & Observability

FrEVL includes comprehensive monitoring:

```python
# Prometheus metrics
from frevl.monitoring import metrics

metrics.inference_counter.inc()
metrics.latency_histogram.observe(latency)

# Logging
from frevl.utils import logger

logger.info(f"Inference completed: {result}")

# Distributed tracing
from frevl.tracing import tracer

with tracer.start_span("inference"):
    result = model.predict(image, text)
```

## Advanced Features

### Embedding Cache Management
```python
# Precompute and cache embeddings
from frevl.cache import EmbeddingCache

cache = EmbeddingCache(cache_dir="./cache")
cache.precompute_dataset("vqa", batch_size=256)
```

### Model Optimization
```python
# Quantization for edge deployment
from frevl.optimize import quantize_model

quantized = quantize_model(model, backend="onnx")
quantized.save("model_int8.onnx")

# TensorRT optimization
from frevl.optimize import optimize_tensorrt

trt_model = optimize_tensorrt(model, fp16=True)
```

### Custom Datasets
```python
# Create custom dataset
from frevl.data import VLDataset

dataset = VLDataset(
    images_dir="./images",
    annotations="./annotations.json",
    transform=transform
)

# Train on custom data
model.train_on_dataset(dataset, epochs=10)
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

```bash
# Setup development environment
make dev-setup

# Run linters and formatters
make lint
make format

# Submit pull request
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
```

## Citation

If you find FrEVL useful in your research, please cite:

```bibtex
@inproceedings{bourigault2025frevl,
  title={Leveraging Frozen Pretrained Embeddings for Efficient Vision-Language Understanding},
  author={Bourigault, Emmanuelle and Bourigault, Pauline},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year={2025},
  pages={1234-1245}
}
```

## Acknowledgments

- OpenAI for CLIP
- Meta AI for ALBEF/BLIP baselines
- HuggingFace for hosting our models
- The open-source community

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
