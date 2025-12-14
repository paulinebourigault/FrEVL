"""
FrEVL Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import clip
from dataclasses import dataclass
import logging
from pathlib import Path
import json


logger = logging.getLogger(__name__)


@dataclass
class FrEVLConfig:
    """Model configuration"""
    # Architecture
    clip_model: str = "ViT-B/32"
    hidden_dim: int = 768
    num_attention_heads: int = 12
    num_fusion_layers: int = 6
    intermediate_dim: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"
    
    # Task heads
    vqa_vocab_size: int = 3129
    retrieval_embedding_dim: int = 256
    
    # Training
    use_gradient_checkpointing: bool = False
    freeze_clip: bool = True
    
    # Inference
    max_length: int = 77
    temperature: float = 1.0
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'FrEVLConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'FrEVLConfig':
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention module"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: [batch_size, query_len, hidden_dim]
            key: [batch_size, key_len, hidden_dim]
            value: [batch_size, value_len, hidden_dim]
            attention_mask: [batch_size, query_len, key_len]
            return_attention: Whether to return attention weights
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        
        if value is None:
            value = key
        
        # Project and reshape
        Q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.hidden_dim
        )
        output = self.o_proj(context)
        
        if return_attention:
            return output, attention_weights.mean(dim=1)  # Average over heads
        return output, None


class FusionLayer(nn.Module):
    """Single fusion layer with cross-attention and FFN"""
    
    def __init__(self, config: FrEVLConfig):
        super().__init__()
        
        # Cross-attention
        self.cross_attention = MultiHeadCrossAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout
        )
        
        # Layer norms
        self.ln_cross = nn.LayerNorm(config.hidden_dim, eps=1e-12)
        self.ln_ffn = nn.LayerNorm(config.hidden_dim, eps=1e-12)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.intermediate_dim),
            nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through fusion layer"""
        
        # Cross-attention: vision attends to text
        residual = vision_features
        vision_features = self.ln_cross(vision_features)
        attended_features, attention_weights = self.cross_attention(
            query=vision_features,
            key=text_features,
            value=text_features,
            attention_mask=attention_mask,
            return_attention=return_attention
        )
        vision_features = residual + attended_features
        
        # Feed-forward
        residual = vision_features
        vision_features = self.ln_ffn(vision_features)
        vision_features = residual + self.ffn(vision_features)
        
        return vision_features, attention_weights


class FrEVL(nn.Module):
    """FrEVL: Frozen Embeddings Vision-Language Model"""
    
    def __init__(self, config: FrEVLConfig):
        super().__init__()
        self.config = config
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(
            config.clip_model, 
            device="cpu",
            jit=False
        )
        
        # Freeze CLIP if specified
        if config.freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
        
        # Get embedding dimensions from CLIP
        self.vision_dim = self.clip_model.visual.output_dim
        self.text_dim = self.clip_model.token_embedding.embedding_dim
        
        # Projection layers to common dimension
        self.vision_proj = nn.Linear(self.vision_dim, config.hidden_dim)
        self.text_proj = nn.Linear(self.text_dim, config.hidden_dim)
        
        # Position embeddings for vision features
        self.vision_pos_embedding = nn.Parameter(
            torch.randn(1, 197, config.hidden_dim) * 0.02  # 196 patches + 1 CLS
        )
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            FusionLayer(config) for _ in range(config.num_fusion_layers)
        ])
        
        # Task-specific heads
        self.vqa_head = VQAHead(config)
        self.retrieval_head = RetrievalHead(config)
        
        # Pooling strategies
        self.pooler = AttentionPooler(config.hidden_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0.0, std=0.02)
    
    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using CLIP vision encoder"""
        if self.config.freeze_clip:
            self.clip_model.eval()
        return self.clip_model.encode_image(images)
    
    @torch.no_grad()
    def encode_text(self, texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """Encode text using CLIP text encoder"""
        if isinstance(texts, list):
            texts = clip.tokenize(texts, truncate=True).to(
                next(self.clip_model.parameters()).device
            )
        if self.config.freeze_clip:
            self.clip_model.eval()
        return self.clip_model.encode_text(texts)
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        texts: Optional[Union[List[str], torch.Tensor]] = None,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        task: str = "vqa",
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FrEVL
        
        Args:
            images: Raw images [batch_size, 3, H, W]
            texts: Text inputs (list of strings or tokenized tensors)
            image_features: Pre-computed image features [batch_size, vision_dim]
            text_features: Pre-computed text features [batch_size, text_dim]
            task: Task to perform ("vqa", "retrieval", "matching")
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with task-specific outputs
        """
        
        # Encode inputs if not pre-computed
        if image_features is None:
            if images is None:
                raise ValueError("Either images or image_features must be provided")
            image_features = self.encode_image(images)
        
        if text_features is None:
            if texts is None:
                raise ValueError("Either texts or text_features must be provided")
            text_features = self.encode_text(texts)
        
        # Project to common dimension
        vision_emb = self.vision_proj(image_features.unsqueeze(1))  # Add sequence dimension
        text_emb = self.text_proj(text_features.unsqueeze(1))
        
        # Add position embeddings to vision features
        if vision_emb.size(1) == 197:  # Full patch sequence
            vision_emb = vision_emb + self.vision_pos_embedding
        
        # Apply fusion layers
        attention_maps = []
        for layer in self.fusion_layers:
            vision_emb, attn_weights = layer(
                vision_emb, 
                text_emb,
                return_attention=return_attention
            )
            if return_attention and attn_weights is not None:
                attention_maps.append(attn_weights)
        
        # Pool features
        pooled_features = self.pooler(vision_emb)
        
        # Task-specific outputs
        outputs = {}
        
        if task == "vqa":
            outputs = self.vqa_head(pooled_features, text_emb.squeeze(1))
        elif task == "retrieval":
            outputs = self.retrieval_head(pooled_features, text_emb.squeeze(1))
        elif task == "matching":
            # Simple cosine similarity for matching
            vision_norm = F.normalize(pooled_features, dim=-1)
            text_norm = F.normalize(text_emb.squeeze(1), dim=-1)
            outputs["similarity"] = (vision_norm * text_norm).sum(dim=-1)
        else:
            outputs["features"] = pooled_features
        
        # Add attention maps if requested
        if return_attention and attention_maps:
            outputs["attention"] = torch.stack(attention_maps).mean(dim=0)
        
        return outputs
    
    def predict(
        self,
        image: Union[torch.Tensor, Any],
        question: str,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        High-level prediction interface
        
        Args:
            image: PIL Image or tensor
            question: Question string
            return_attention: Whether to return attention visualization
        
        Returns:
            Dictionary with answer, confidence, and optional attention
        """
        self.eval()
        
        # Preprocess image if needed
        if not isinstance(image, torch.Tensor):
            image = self.clip_preprocess(image).unsqueeze(0)
        
        # Move to correct device
        device = next(self.parameters()).device
        image = image.to(device)
        
        with torch.no_grad():
            outputs = self.forward(
                images=image,
                texts=[question],
                task="vqa",
                return_attention=return_attention
            )
            
            # Get top predictions
            probs = F.softmax(outputs["logits"], dim=-1)
            confidence, pred_idx = probs.max(dim=-1)
            
            # Convert to answer (placeholder - need vocab mapping)
            answer = f"answer_{pred_idx.item()}"
            
            result = {
                "answer": answer,
                "confidence": confidence.item(),
                "top_k_predictions": self._get_top_k_predictions(probs[0], k=5)
            }
            
            if return_attention:
                result["attention"] = outputs.get("attention", None)
        
        return result
    
    def _get_top_k_predictions(self, probs: torch.Tensor, k: int = 5) -> List[Dict]:
        """Get top-k predictions with scores"""
        top_probs, top_indices = probs.topk(k)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                "answer": f"answer_{idx.item()}",  # Placeholder
                "confidence": prob.item()
            })
        
        return predictions
    
    def batch_forward(
        self,
        images: torch.Tensor,
        questions: List[str]
    ) -> Dict[str, List]:
        """Optimized batch inference"""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(
                images=images,
                texts=questions,
                task="vqa"
            )
            
            probs = F.softmax(outputs["logits"], dim=-1)
            confidences, predictions = probs.max(dim=-1)
            
            return {
                "answers": [f"answer_{p.item()}" for p in predictions],
                "confidences": confidences.tolist()
            }
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> 'FrEVL':
        """Load pretrained model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load config
        config = FrEVLConfig.from_dict(checkpoint.get("config", {}))
        
        # Create model
        model = cls(config)
        
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
    
    def save_pretrained(self, save_path: str):
        """Save model checkpoint"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "config": self.config.__dict__,
            "model_state_dict": self.state_dict()
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")


class VQAHead(nn.Module):
    """VQA task head"""
    
    def __init__(self, config: FrEVLConfig):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.vqa_vocab_size)
        )
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for VQA"""
        # Concatenate vision and text features
        combined = torch.cat([vision_features, text_features], dim=-1)
        
        # Classification
        logits = self.classifier(combined)
        
        return {"logits": logits}


class RetrievalHead(nn.Module):
    """Retrieval task head"""
    
    def __init__(self, config: FrEVLConfig):
        super().__init__()
        
        self.vision_proj = nn.Linear(
            config.hidden_dim, 
            config.retrieval_embedding_dim
        )
        self.text_proj = nn.Linear(
            config.hidden_dim, 
            config.retrieval_embedding_dim
        )
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for retrieval"""
        # Project to retrieval space
        vision_emb = F.normalize(self.vision_proj(vision_features), dim=-1)
        text_emb = F.normalize(self.text_proj(text_features), dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(vision_emb, text_emb.t()) * self.temperature
        
        return {
            "similarity": similarity,
            "vision_embeddings": vision_emb,
            "text_embeddings": text_emb
        }


class AttentionPooler(nn.Module):
    """Attention-based pooling"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence features using attention
        
        Args:
            features: [batch_size, seq_len, hidden_dim]
        
        Returns:
            pooled: [batch_size, hidden_dim]
        """
        # Compute attention scores
        scores = self.attention_weights(features).squeeze(-1)  # [batch, seq_len]
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Weighted average
        pooled = (features * weights).sum(dim=1)
        
        return pooled


# Utility function for model initialization
def create_model(config_path: Optional[str] = None, **kwargs) -> FrEVL:
    """Create FrEVL model with configuration"""
    if config_path:
        config = FrEVLConfig.from_json(config_path)
    else:
        config = FrEVLConfig(**kwargs)
    
    return FrEVL(config)
