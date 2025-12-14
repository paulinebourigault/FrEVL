"""
Unit tests for FrEVL model
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json

from model import FrEVL, FrEVLConfig, VQAHead, RetrievalHead
from utils import get_model_size


class TestFrEVLConfig:
    """Test model configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = FrEVLConfig()
        assert config.clip_model == "ViT-B/32"
        assert config.hidden_dim == 768
        assert config.num_layers == 6
        assert config.num_heads == 12
        assert config.dropout == 0.1
    
    def test_config_from_json(self, tmp_path):
        """Test loading config from JSON"""
        config_dict = {
            "clip_model": "ViT-B/16",
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.2
        }
        
        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f)
        
        config = FrEVLConfig.from_json(config_file)
        assert config.clip_model == "ViT-B/16"
        assert config.hidden_dim == 512
        assert config.num_layers == 4
    
    def test_config_to_json(self, tmp_path):
        """Test saving config to JSON"""
        config = FrEVLConfig(hidden_dim=512, num_layers=4)
        config_file = tmp_path / "config.json"
        config.to_json(config_file)
        
        assert config_file.exists()
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        assert loaded_config["hidden_dim"] == 512
        assert loaded_config["num_layers"] == 4


class TestFrEVLModel:
    """Test main FrEVL model"""
    
    @pytest.fixture
    def model(self):
        """Create a test model"""
        config = FrEVLConfig(
            clip_model="ViT-B/32",
            hidden_dim=256,
            num_layers=2,
            num_heads=4
        )
        return FrEVL(config)
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data"""
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        texts = ["What is this?", "What color is it?"]
        return images, texts
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert isinstance(model, FrEVL)
        assert hasattr(model, 'clip_model')
        assert hasattr(model, 'vision_proj')
        assert hasattr(model, 'text_proj')
        assert hasattr(model, 'fusion_layers')
        assert hasattr(model, 'vqa_head')
        assert hasattr(model, 'retrieval_head')
        assert len(model.fusion_layers) == 2
    
    def test_forward_pass_vqa(self, model, sample_batch):
        """Test forward pass for VQA task"""
        images, texts = sample_batch
        model.eval()
        
        with torch.no_grad():
            output = model(images=images, text=texts, task="vqa")
        
        assert "logits" in output
        assert "predictions" in output
        assert "confidence" in output
        assert output["logits"].shape == (2, model.config.num_vqa_answers)
        assert output["predictions"].shape == (2,)
        assert output["confidence"].shape == (2,)
    
    def test_forward_pass_retrieval(self, model, sample_batch):
        """Test forward pass for retrieval task"""
        images, texts = sample_batch
        model.eval()
        
        with torch.no_grad():
            output = model(images=images, text=texts, task="retrieval")
        
        assert "image_text_similarity" in output
        assert "retrieval_scores" in output
        assert output["retrieval_scores"].shape == (2,)
    
    def test_forward_with_attention(self, model, sample_batch):
        """Test forward pass with attention return"""
        images, texts = sample_batch
        model.eval()
        
        with torch.no_grad():
            output = model(
                images=images,
                text=texts,
                task="vqa",
                return_attention=True
            )
        
        assert "attention" in output
        assert output["attention"] is not None
    
    def test_batch_forward(self, model, sample_batch):
        """Test batch forward pass"""
        images, texts = sample_batch
        model.eval()
        
        output = model.batch_forward(images, texts, task="vqa")
        assert "logits" in output
        assert output["logits"].shape[0] == 2
    
    def test_frozen_clip_parameters(self, model):
        """Test that CLIP parameters are frozen"""
        for param in model.clip_model.parameters():
            assert not param.requires_grad
    
    def test_trainable_parameters(self, model):
        """Test that fusion layers are trainable"""
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        
        assert len(trainable_params) > 0
        assert any("vision_proj" in name for name in trainable_params)
        assert any("text_proj" in name for name in trainable_params)
        assert any("fusion_layers" in name for name in trainable_params)
    
    def test_model_size(self, model):
        """Test model size calculation"""
        size_info = get_model_size(model)
        assert "total_params" in size_info
        assert "trainable_params" in size_info
        assert size_info["trainable_params"] < size_info["total_params"]
        assert size_info["trainable_params"] < 100_000_000  # Less than 100M
    
    def test_save_and_load_pretrained(self, model, tmp_path):
        """Test saving and loading model"""
        save_path = tmp_path / "test_model"
        model.save_pretrained(save_path)
        
        # Check saved files
        assert (save_path / "config.json").exists()
        assert ((save_path / "model.safetensors").exists() or 
                (save_path / "pytorch_model.pt").exists())
        
        # Load model
        loaded_model = FrEVL.from_pretrained(save_path)
        assert isinstance(loaded_model, FrEVL)
        
        # Check parameters match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(),
            loaded_model.named_parameters()
        ):
            if "clip_model" not in n1:  # Skip CLIP parameters
                assert torch.allclose(p1, p2, atol=1e-6)


class TestVQAHead:
    """Test VQA head module"""
    
    @pytest.fixture
    def vqa_head(self):
        config = FrEVLConfig(hidden_dim=256)
        return VQAHead(config)
    
    def test_vqa_head_forward(self, vqa_head):
        """Test VQA head forward pass"""
        batch_size = 4
        features = torch.randn(batch_size, 256)
        
        output = vqa_head(features)
        
        assert "logits" in output
        assert "probabilities" in output
        assert "predictions" in output
        assert "top_k_predictions" in output
        assert "confidence" in output
        
        assert output["logits"].shape == (batch_size, 3129)
        assert output["probabilities"].shape == (batch_size, 3129)
        assert output["predictions"].shape == (batch_size,)
        assert output["confidence"].shape == (batch_size,)


class TestRetrievalHead:
    """Test retrieval head module"""
    
    @pytest.fixture
    def retrieval_head(self):
        config = FrEVLConfig(hidden_dim=256)
        return RetrievalHead(config)
    
    def test_retrieval_head_forward(self, retrieval_head):
        """Test retrieval head forward pass"""
        batch_size = 4
        pooled_features = torch.randn(batch_size, 256)
        image_features = torch.randn(batch_size, 512)
        text_features = torch.randn(batch_size, 512)
        
        output = retrieval_head(pooled_features, image_features, text_features)
        
        assert "image_text_similarity" in output
        assert "fusion_similarity" in output
        assert "retrieval_scores" in output
        
        assert output["image_text_similarity"].shape == (batch_size, batch_size)
        assert output["fusion_similarity"].shape == (batch_size, batch_size)
        assert output["retrieval_scores"].shape == (batch_size,)


@pytest.mark.gpu
class TestFrEVLGPU:
    """Test model on GPU (if available)"""
    
    @pytest.fixture
    def model_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        config = FrEVLConfig(hidden_dim=256, num_layers=2)
        model = FrEVL(config)
        return model.cuda()
    
    def test_gpu_forward_pass(self, model_gpu):
        """Test forward pass on GPU"""
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224).cuda()
        texts = ["test"] * batch_size
        
        model_gpu.eval()
        with torch.no_grad():
            output = model_gpu(images=images, text=texts, task="vqa")
        
        assert output["logits"].is_cuda
        assert output["predictions"].is_cuda
    
    def test_mixed_precision(self, model_gpu):
        """Test mixed precision training"""
        from torch.cuda.amp import autocast
        
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224).cuda()
        texts = ["test"] * batch_size
        
        model_gpu.eval()
        with autocast():
            output = model_gpu(images=images, text=texts, task="vqa")
        
        assert "logits" in output


@pytest.mark.slow
class TestFrEVLIntegration:
    """Integration tests (marked as slow)"""
    
    def test_full_training_step(self):
        """Test a complete training step"""
        config = FrEVLConfig(hidden_dim=128, num_layers=1, num_heads=2)
        model = FrEVL(config)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Sample data
        images = torch.randn(2, 3, 224, 224)
        texts = ["What is this?", "What color?"]
        labels = torch.tensor([0, 1])
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        output = model(images=images, text=texts, task="vqa")
        loss = criterion(output["logits"], labels)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        
        # Check gradients were computed
        for name, param in model.named_parameters():
            if param.requires_grad and "clip_model" not in name:
                assert param.grad is not None
    
    def test_export_onnx(self, tmp_path):
        """Test ONNX export"""
        pytest.importorskip("onnx")
        
        config = FrEVLConfig(hidden_dim=128, num_layers=1)
        model = FrEVL(config)
        model.eval()
        
        # Dummy inputs
        dummy_image_features = torch.randn(1, 512)
        dummy_text_features = torch.randn(1, 512)
        
        # Export
        onnx_path = tmp_path / "model.onnx"
        torch.onnx.export(
            model,
            (None, None, dummy_image_features, dummy_text_features),
            onnx_path,
            input_names=['image_features', 'text_features'],
            output_names=['output'],
            dynamic_axes={
                'image_features': {0: 'batch'},
                'text_features': {0: 'batch'},
                'output': {0: 'batch'}
            },
            opset_version=14
        )
        
        assert onnx_path.exists()
