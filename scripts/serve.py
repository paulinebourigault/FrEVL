"""
FrEVL: Production API Server with FastAPI
High-performance serving with monitoring, caching, and optimization
"""

import os
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import hashlib
import json
from collections import deque
from contextlib import asynccontextmanager
import uuid

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image
import io
import numpy as np
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import logging
from cachetools import TTLCache
import aiofiles
from motor.motor_asyncio import AsyncIOMotorClient
import clip

from model import FrEVL
from utils import preprocess_image


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Server configuration"""
    # Model
    MODEL_NAME = os.getenv("MODEL_NAME", "frevl-base")
    MODEL_PATH = os.getenv("MODEL_PATH", "./checkpoints/frevl-base.pt")
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    WORKERS = int(os.getenv("WORKERS", 4))
    
    # Performance
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 32))
    BATCH_TIMEOUT_MS = int(os.getenv("BATCH_TIMEOUT_MS", 50))
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 100))
    
    # Caching
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Monitoring
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    
    # Security
    API_KEY = os.getenv("API_KEY", None)
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 60))


# ============================================================================
# Pydantic Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Single prediction request"""
    question: str = Field(..., description="Question about the image")
    return_attention: bool = Field(False, description="Return attention weights")
    top_k: int = Field(5, description="Number of top predictions to return")


class PredictionResponse(BaseModel):
    """Single prediction response"""
    answer: str
    confidence: float
    inference_time_ms: float
    top_k_predictions: Optional[List[Dict[str, float]]] = None
    attention_weights: Optional[List[List[float]]] = None
    request_id: str
    model_version: str
    cached: bool = False


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    questions: List[str]
    return_attention: bool = False


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_time_ms: float
    batch_size: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    gpu_available: bool
    cache_connected: bool
    uptime_seconds: float
    requests_processed: int
    average_latency_ms: float


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    version: str
    parameters: int
    device: str
    capabilities: List[str]


# ============================================================================
# Metrics
# ============================================================================

# Prometheus metrics
request_counter = Counter(
    'frevl_requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)

latency_histogram = Histogram(
    'frevl_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)

model_inference_histogram = Histogram(
    'frevl_model_inference_seconds',
    'Model inference time in seconds'
)

active_requests = Gauge(
    'frevl_active_requests',
    'Number of active requests'
)

cache_hits = Counter(
    'frevl_cache_hits_total',
    'Total number of cache hits'
)

cache_misses = Counter(
    'frevl_cache_misses_total',
    'Total number of cache misses'
)


# ============================================================================
# Model Server
# ============================================================================

class ModelServer:
    """Main model server with optimizations"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device(Config.DEVICE)
        self.request_queue = asyncio.Queue(maxsize=Config.MAX_CONCURRENT_REQUESTS)
        self.batch_queue = deque()
        self.processing = False
        
        # Caching
        self.local_cache = TTLCache(maxsize=1000, ttl=Config.CACHE_TTL_SECONDS)
        self.redis_client = None
        
        # Monitoring
        self.start_time = time.time()
        self.request_count = 0
        self.total_latency = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    async def initialize(self):
        """Initialize server components"""
        # Load model
        self.logger.info(f"Loading model {Config.MODEL_NAME}...")
        self.model = await self._load_model()
        
        # Setup Redis cache
        if Config.ENABLE_CACHE:
            try:
                self.redis_client = redis.from_url(
                    Config.REDIS_URL,
                    decode_responses=True
                )
                await self.redis_client.ping()
                self.logger.info("Redis cache connected")
            except Exception as e:
                self.logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Start batch processing
        asyncio.create_task(self._batch_processor())
        
        self.logger.info("Model server initialized successfully")
    
    async def _load_model(self) -> FrEVL:
        """Load model with optimization"""
        model = FrEVL.from_pretrained(Config.MODEL_PATH)
        model.to(self.device)
        model.eval()
        
        # Optimize model for inference
        if Config.DEVICE == "cuda":
            model = torch.jit.script(model)  # TorchScript optimization
            
        return model
    
    def _get_cache_key(self, image_hash: str, question: str) -> str:
        """Generate cache key"""
        combined = f"{image_hash}:{question}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get result from cache"""
        # Check local cache first
        if cache_key in self.local_cache:
            cache_hits.inc()
            return self.local_cache[cache_key]
        
        # Check Redis cache
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    cache_hits.inc()
                    result = json.loads(cached)
                    self.local_cache[cache_key] = result
                    return result
            except Exception as e:
                self.logger.error(f"Redis get error: {e}")
        
        cache_misses.inc()
        return None
    
    async def _set_cache(self, cache_key: str, result: Dict):
        """Store result in cache"""
        self.local_cache[cache_key] = result
        
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    Config.CACHE_TTL_SECONDS,
                    json.dumps(result)
                )
            except Exception as e:
                self.logger.error(f"Redis set error: {e}")
    
    async def predict_single(
        self,
        image: Image.Image,
        question: str,
        return_attention: bool = False,
        use_cache: bool = True
    ) -> Dict:
        """Single prediction with caching"""
        
        # Generate cache key
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_hash = hashlib.md5(image_bytes.getvalue()).hexdigest()
        
        if use_cache and Config.ENABLE_CACHE:
            cache_key = self._get_cache_key(image_hash, question)
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                cached_result['cached'] = True
                return cached_result
        
        # Model inference
        start_time = time.time()
        
        with torch.no_grad():
            # Preprocess image
            image_tensor = preprocess_image(image).unsqueeze(0).to(self.device)
            
            # Run model
            output = self.model(image_tensor, question, return_attention=return_attention)
            
            # Process output
            result = {
                'answer': output['answer'],
                'confidence': float(output['confidence']),
                'inference_time_ms': (time.time() - start_time) * 1000,
                'top_k_predictions': output.get('top_k', []),
                'model_version': Config.MODEL_NAME
            }
            
            if return_attention:
                result['attention_weights'] = output['attention'].tolist()
        
        # Cache result
        if use_cache and Config.ENABLE_CACHE:
            await self._set_cache(cache_key, result)
        
        # Update metrics
        model_inference_histogram.observe(time.time() - start_time)
        
        return result
    
    async def predict_batch(
        self,
        images: List[Image.Image],
        questions: List[str]
    ) -> List[Dict]:
        """Batch prediction with optimization"""
        
        start_time = time.time()
        batch_size = len(images)
        
        with torch.no_grad():
            # Batch preprocessing
            image_tensors = torch.stack([
                preprocess_image(img) for img in images
            ]).to(self.device)
            
            # Batch inference
            outputs = self.model.batch_forward(image_tensors, questions)
            
            # Process outputs
            results = []
            for i in range(batch_size):
                results.append({
                    'answer': outputs['answers'][i],
                    'confidence': float(outputs['confidences'][i]),
                    'inference_time_ms': (time.time() - start_time) * 1000 / batch_size
                })
        
        return results
    
    async def _batch_processor(self):
        """Process requests in batches for efficiency"""
        while True:
            try:
                # Collect batch
                batch = []
                timeout = Config.BATCH_TIMEOUT_MS / 1000
                deadline = time.time() + timeout
                
                while len(batch) < Config.MAX_BATCH_SIZE:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    
                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=remaining
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break
                
                # Process batch if not empty
                if batch:
                    await self._process_batch(batch)
                    
            except Exception as e:
                self.logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch(self, batch: List[Dict]):
        """Process a batch of requests"""
        # Group by image for efficiency
        image_groups = {}
        
        for request in batch:
            image_hash = request['image_hash']
            if image_hash not in image_groups:
                image_groups[image_hash] = {
                    'image': request['image'],
                    'requests': []
                }
            image_groups[image_hash]['requests'].append(request)
        
        # Process each image group
        for group in image_groups.values():
            image = group['image']
            requests = group['requests']
            questions = [r['question'] for r in requests]
            
            # Batch inference
            results = await self.predict_batch([image] * len(questions), questions)
            
            # Return results
            for request, result in zip(requests, results):
                request['future'].set_result(result)
    
    def get_health_status(self) -> Dict:
        """Get server health status"""
        uptime = time.time() - self.start_time
        avg_latency = self.total_latency / max(self.request_count, 1) * 1000
        
        return {
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'gpu_available': torch.cuda.is_available(),
            'cache_connected': self.redis_client is not None,
            'uptime_seconds': uptime,
            'requests_processed': self.request_count,
            'average_latency_ms': avg_latency
        }


# ============================================================================
# FastAPI Application
# ============================================================================

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await app.state.model_server.initialize()
    yield
    # Shutdown
    # Cleanup code here if needed

# Create FastAPI app
app = FastAPI(
    title="FrEVL API",
    description="Efficient Vision-Language Understanding",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model server
app.state.model_server = ModelServer()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint"""
    return {
        "name": "FrEVL API",
        "version": "1.0.0",
        "description": "Efficient Vision-Language Understanding",
        "endpoints": {
            "predict": "/predict",
            "batch": "/batch",
            "health": "/health",
            "metrics": "/metrics",
            "models": "/models"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    question: str = Form(...),
    return_attention: bool = Form(False),
    top_k: int = Form(5)
):
    """Single image VQA prediction"""
    
    # Track metrics
    active_requests.inc()
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Validate image
        if not image.content_type.startswith("image/"):
            raise HTTPException(400, "Invalid image format")
        
        # Load image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Get prediction
        result = await app.state.model_server.predict_single(
            pil_image,
            question,
            return_attention=return_attention
        )
        
        # Prepare response
        response = PredictionResponse(
            answer=result['answer'],
            confidence=result['confidence'],
            inference_time_ms=result['inference_time_ms'],
            top_k_predictions=result.get('top_k_predictions'),
            attention_weights=result.get('attention_weights'),
            request_id=request_id,
            model_version=result['model_version'],
            cached=result.get('cached', False)
        )
        
        # Track metrics
        latency = time.time() - start_time
        latency_histogram.labels(endpoint='predict').observe(latency)
        request_counter.labels(endpoint='predict', status='success').inc()
        
        # Update server stats
        app.state.model_server.request_count += 1
        app.state.model_server.total_latency += latency
        
        return response
        
    except Exception as e:
        request_counter.labels(endpoint='predict', status='error').inc()
        raise HTTPException(500, str(e))
    
    finally:
        active_requests.dec()


@app.post("/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    images: List[UploadFile] = File(...)
):
    """Batch VQA predictions"""
    
    active_requests.inc()
    start_time = time.time()
    
    try:
        # Validate inputs
        if len(images) != len(request.questions):
            raise HTTPException(400, "Number of images must match number of questions")
        
        if len(images) > Config.MAX_BATCH_SIZE:
            raise HTTPException(400, f"Batch size exceeds maximum of {Config.MAX_BATCH_SIZE}")
        
        # Load images
        pil_images = []
        for img_file in images:
            if not img_file.content_type.startswith("image/"):
                raise HTTPException(400, f"Invalid image format: {img_file.filename}")
            
            image_bytes = await img_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pil_images.append(pil_image)
        
        # Batch prediction
        results = await app.state.model_server.predict_batch(
            pil_images,
            request.questions
        )
        
        # Prepare response
        predictions = []
        for i, result in enumerate(results):
            predictions.append(PredictionResponse(
                answer=result['answer'],
                confidence=result['confidence'],
                inference_time_ms=result['inference_time_ms'],
                request_id=str(uuid.uuid4()),
                model_version=Config.MODEL_NAME,
                cached=False
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_time_ms=total_time,
            batch_size=len(predictions)
        )
        
        # Track metrics
        latency_histogram.labels(endpoint='batch').observe(time.time() - start_time)
        request_counter.labels(endpoint='batch', status='success').inc()
        
        return response
        
    except Exception as e:
        request_counter.labels(endpoint='batch', status='error').inc()
        raise HTTPException(500, str(e))
    
    finally:
        active_requests.dec()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    status = app.state.model_server.get_health_status()
    return HealthResponse(**status)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if not Config.ENABLE_METRICS:
        raise HTTPException(404, "Metrics not enabled")
    
    return Response(generate_latest(), media_type="text/plain")


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models"""
    models = [
        ModelInfo(
            name="frevl-base",
            version="1.0.0",
            parameters=68400000,
            device=Config.DEVICE,
            capabilities=["vqa", "image-text-matching", "visual-reasoning"]
        ),
        ModelInfo(
            name="frevl-large",
            version="1.0.0",
            parameters=124800000,
            device=Config.DEVICE,
            capabilities=["vqa", "image-text-matching", "visual-reasoning", "multilingual"]
        )
    ]
    return models


@app.post("/feedback")
async def submit_feedback(
    request_id: str = Form(...),
    rating: int = Form(...),
    comment: Optional[str] = Form(None)
):
    """Submit feedback for a prediction"""
    # Store feedback in database
    # This is a placeholder - implement actual storage
    return {"status": "Feedback received", "request_id": request_id}


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FrEVL API Server")
    parser.add_argument("--host", type=str, default=Config.HOST)
    parser.add_argument("--port", type=int, default=Config.PORT)
    parser.add_argument("--workers", type=int, default=Config.WORKERS)
    parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "serve:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
