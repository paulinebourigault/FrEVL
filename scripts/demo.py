"""
FrEVL: Interactive Demo with Gradio
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import base64
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr
import clip
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from transformers import AutoTokenizer

from model import FrEVL
from utils import load_checkpoint, preprocess_image


# ============================================================================
# Configuration
# ============================================================================

EXAMPLE_IMAGES = [
    {
        "path": "examples/dog_park.jpg",
        "questions": [
            "What breed is this dog?",
            "What is the dog doing?",
            "Is this indoors or outdoors?"
        ]
    },
    {
        "path": "examples/kitchen.jpg", 
        "questions": [
            "What appliances can you see?",
            "What color are the cabinets?",
            "Is someone cooking?"
        ]
    },
    {
        "path": "examples/street.jpg",
        "questions": [
            "Is it safe to cross the street?",
            "What time of day is it?",
            "How many cars are visible?"
        ]
    }
]

MODEL_OPTIONS = {
    "FrEVL-Base (Fast)": "frevl-base",
    "FrEVL-Large (Accurate)": "frevl-large", 
    "FrEVL-Multilingual": "frevl-multi"
}


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages model loading and caching"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_name: str) -> FrEVL:
        """Load model with caching"""
        if model_name not in self.models:
            print(f"Loading {model_name}...")
            checkpoint_path = f"checkpoints/{model_name}.pt"
            
            if not Path(checkpoint_path).exists():
                # Download from HuggingFace
                from huggingface_hub import hf_hub_download
                checkpoint_path = hf_hub_download(
                    repo_id=f"EmmanuelleB985/{model_name}",
                    filename="model.pt",
                    cache_dir="./cache"
                )
            
            # Load model
            model = FrEVL.from_pretrained(checkpoint_path)
            model.to(self.device)
            model.eval()
            
            self.models[model_name] = model
            
        self.current_model = self.models[model_name]
        return self.current_model
    
    def get_current_model(self) -> FrEVL:
        """Get current active model"""
        if self.current_model is None:
            self.load_model("frevl-base")
        return self.current_model


# ============================================================================
# Visualization Functions
# ============================================================================

def create_attention_heatmap(
    image: Image.Image,
    attention_weights: np.ndarray,
    question: str
) -> Image.Image:
    """Create attention visualization heatmap"""
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Resize attention weights to match image size
    attention_resized = cv2.resize(
        attention_weights,
        (img_array.shape[1], img_array.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_array)
    plt.title("Original Image")
    plt.axis('off')
    
    # Attention overlay
    plt.subplot(1, 2, 2)
    plt.imshow(img_array)
    plt.imshow(attention_resized, cmap='jet', alpha=0.5)
    plt.title(f"Attention: {question[:50]}...")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Save to buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


def create_confidence_chart(
    predictions: Dict[str, float],
    top_k: int = 5
) -> Image.Image:
    """Create confidence score bar chart"""
    # Sort predictions by confidence
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    labels = [item[0] for item in sorted_preds]
    scores = [item[1] for item in sorted_preds]
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(scores))]
    bars = plt.barh(range(len(labels)), scores, color=colors)
    
    # Customize chart
    plt.xlabel('Confidence Score', fontsize=12)
    plt.title('Top Predictions', fontsize=14, fontweight='bold')
    plt.yticks(range(len(labels)), labels, fontsize=11)
    plt.xlim(0, 1)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=10)
    
    # Style
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    sns.despine(left=True, bottom=True)
    
    # Save to buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


# ============================================================================
# Main Demo Interface
# ============================================================================

class FrEVLDemo:
    """Main demo application"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.setup_interface()
        
    def process_single_query(
        self,
        image: Image.Image,
        question: str,
        model_choice: str,
        show_attention: bool,
        show_confidence: bool
    ) -> Tuple[str, Optional[Image.Image], Optional[Image.Image], float, Dict]:
        """Process a single VQA query"""
        
        if image is None:
            return "Please upload an image", None, None, 0.0, {}
        
        if not question.strip():
            return "Please enter a question", None, None, 0.0, {}
        
        # Start timing
        start_time = time.time()
        
        # Load model
        model_name = MODEL_OPTIONS.get(model_choice, "frevl-base")
        model = self.model_manager.load_model(model_name)
        
        # Process image and question
        with torch.no_grad():
            # Preprocess
            image_tensor = preprocess_image(image).unsqueeze(0).to(model.device)
            
            # Get predictions with attention
            output = model(image_tensor, question, return_attention=True)
            
            # Get answer
            answer = output['answer']
            confidence = output['confidence']
            attention_weights = output['attention'] if show_attention else None
            top_predictions = output['top_k_predictions']
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Create visualizations
        attention_viz = None
        confidence_viz = None
        
        if show_attention and attention_weights is not None:
            attention_viz = create_attention_heatmap(
                image, 
                attention_weights.cpu().numpy(),
                question
            )
        
        if show_confidence:
            confidence_viz = create_confidence_chart(top_predictions)
        
        # Format output
        output_text = f"**Answer:** {answer}\n"
        output_text += f"**Confidence:** {confidence:.2%}\n"
        output_text += f"**Inference Time:** {inference_time:.1f} ms"
        
        # Performance metrics
        metrics = {
            "model": model_name,
            "inference_ms": round(inference_time, 2),
            "confidence": round(confidence, 4),
            "gpu_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }
        
        return output_text, attention_viz, confidence_viz, confidence, metrics
    
    def process_batch_queries(
        self,
        image: Image.Image,
        questions: str,
        model_choice: str
    ) -> str:
        """Process multiple questions at once"""
        
        if image is None:
            return "Please upload an image"
        
        if not questions.strip():
            return "Please enter questions (one per line)"
        
        # Parse questions
        question_list = [q.strip() for q in questions.split('\n') if q.strip()]
        
        # Load model
        model_name = MODEL_OPTIONS.get(model_choice, "frevl-base")
        model = self.model_manager.load_model(model_name)
        
        # Process all questions
        results = []
        total_time = 0
        
        for question in question_list:
            start_time = time.time()
            
            with torch.no_grad():
                output = model(image, question)
                answer = output['answer']
                confidence = output['confidence']
            
            inference_time = (time.time() - start_time) * 1000
            total_time += inference_time
            
            results.append(f"**Q:** {question}\n**A:** {answer} (confidence: {confidence:.2%})\n")
        
        # Format output
        output_text = "## Batch Results\n\n"
        output_text += "\n".join(results)
        output_text += f"\n---\n**Total Time:** {total_time:.1f} ms"
        output_text += f"\n**Avg Time per Question:** {total_time/len(question_list):.1f} ms"
        
        return output_text
    
    def compare_models(
        self,
        image: Image.Image,
        question: str
    ) -> Tuple[str, Image.Image]:
        """Compare different model variants"""
        
        if image is None or not question.strip():
            return "Please provide both image and question", None
        
        results = {}
        
        for model_label, model_name in MODEL_OPTIONS.items():
            model = self.model_manager.load_model(model_name)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(image, question)
            inference_time = (time.time() - start_time) * 1000
            
            results[model_label] = {
                'answer': output['answer'],
                'confidence': output['confidence'],
                'time_ms': inference_time
            }
        
        # Create comparison table
        output_text = "## Model Comparison\n\n"
        output_text += "| Model | Answer | Confidence | Time (ms) |\n"
        output_text += "|-------|--------|------------|----------|\n"
        
        for model, data in results.items():
            output_text += f"| {model} | {data['answer']} | "
            output_text += f"{data['confidence']:.2%} | {data['time_ms']:.1f} |\n"
        
        # Create comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confidence comparison
        models = list(results.keys())
        confidences = [results[m]['confidence'] for m in models]
        ax1.bar(models, confidences, color=['#2ecc71', '#3498db', '#9b59b6'])
        ax1.set_ylabel('Confidence Score')
        ax1.set_title('Confidence Comparison')
        ax1.set_ylim(0, 1)
        
        # Speed comparison
        times = [results[m]['time_ms'] for m in models]
        ax2.bar(models, times, color=['#e74c3c', '#f39c12', '#1abc9c'])
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Speed Comparison')
        
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        
        comparison_chart = Image.open(buf)
        
        return output_text, comparison_chart
    
    def setup_interface(self) -> gr.Blocks:
        """Setup Gradio interface"""
        
        with gr.Blocks(
            title="FrEVL: Efficient Vision-Language Understanding",
            theme=gr.themes.Soft(),
            css="""
            .container {max-width: 1200px; margin: auto; padding: 20px;}
            .header {text-align: center; margin-bottom: 30px;}
            .metric-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 10px;
            }
            """
        ) as demo:
            
            # Header
            gr.Markdown("""
            <div class="header">
                <h1>🚀 FrEVL: Frozen Embeddings Vision-Language Understanding</h1>
                <p>Experience state-of-the-art VQA with 10× fewer parameters</p>
                <p>
                    <a href="https://arxiv.org/pdf/2508.04469">📄 Paper</a> |
                    <a href="https://github.com/EmmanuelleB985/FrEVL">💻 Code</a> |
                    <a href="https://huggingface.co/EmmanuelleB985/FrEVL">🤗 Models</a>
                </p>
            </div>
            """)
            
            with gr.Tabs():
                # Tab 1: Single Query
                with gr.TabItem("🎯 Single Query"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            input_image = gr.Image(
                                label="Upload Image",
                                type="pil"
                            )
                            input_question = gr.Textbox(
                                label="Question",
                                placeholder="What do you want to know about this image?",
                                lines=2
                            )
                            model_choice = gr.Dropdown(
                                choices=list(MODEL_OPTIONS.keys()),
                                value="FrEVL-Base (Fast)",
                                label="Model"
                            )
                            
                            with gr.Row():
                                show_attention = gr.Checkbox(
                                    label="Show Attention",
                                    value=True
                                )
                                show_confidence = gr.Checkbox(
                                    label="Show Confidence",
                                    value=True
                                )
                            
                            submit_btn = gr.Button(
                                "🔍 Analyze",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            output_text = gr.Markdown(label="Result")
                            confidence_score = gr.Slider(
                                label="Confidence",
                                minimum=0,
                                maximum=1,
                                interactive=False
                            )
                            
                    with gr.Row():
                        attention_viz = gr.Image(
                            label="Attention Visualization",
                            visible=True
                        )
                        confidence_chart = gr.Image(
                            label="Top Predictions",
                            visible=True
                        )
                    
                    # Metrics display
                    metrics_json = gr.JSON(label="Performance Metrics")
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["examples/dog.jpg", "What breed is this dog?"],
                            ["examples/kitchen.jpg", "What appliances are visible?"],
                            ["examples/street.jpg", "Is it safe to cross?"]
                        ],
                        inputs=[input_image, input_question]
                    )
                    
                    # Connect single query
                    submit_btn.click(
                        fn=self.process_single_query,
                        inputs=[
                            input_image,
                            input_question,
                            model_choice,
                            show_attention,
                            show_confidence
                        ],
                        outputs=[
                            output_text,
                            attention_viz,
                            confidence_chart,
                            confidence_score,
                            metrics_json
                        ]
                    )
                
                # Tab 2: Batch Processing
                with gr.TabItem("📋 Batch Queries"):
                    with gr.Row():
                        with gr.Column():
                            batch_image = gr.Image(
                                label="Upload Image",
                                type="pil"
                            )
                            batch_questions = gr.Textbox(
                                label="Questions (one per line)",
                                placeholder="What color is the sky?\nHow many people are there?\nWhat time of day is it?",
                                lines=5
                            )
                            batch_model = gr.Dropdown(
                                choices=list(MODEL_OPTIONS.keys()),
                                value="FrEVL-Base (Fast)",
                                label="Model"
                            )
                            batch_submit = gr.Button("Process Batch", variant="primary")
                        
                        with gr.Column():
                            batch_output = gr.Markdown(label="Batch Results")
                    
                    batch_submit.click(
                        fn=self.process_batch_queries,
                        inputs=[batch_image, batch_questions, batch_model],
                        outputs=batch_output
                    )
                
                # Tab 3: Model Comparison
                with gr.TabItem("⚖️ Compare Models"):
                    with gr.Row():
                        with gr.Column():
                            compare_image = gr.Image(
                                label="Upload Image",
                                type="pil"
                            )
                            compare_question = gr.Textbox(
                                label="Question",
                                placeholder="Enter your question"
                            )
                            compare_btn = gr.Button("Compare All Models", variant="primary")
                        
                        with gr.Column():
                            compare_output = gr.Markdown(label="Comparison Results")
                            compare_chart = gr.Image(label="Performance Chart")
                    
                    compare_btn.click(
                        fn=self.compare_models,
                        inputs=[compare_image, compare_question],
                        outputs=[compare_output, compare_chart]
                    )
                
                # Tab 4: Live Webcam
                with gr.TabItem("📹 Live Demo"):
                    gr.Markdown("## Real-time Vision-Language Understanding")
                    
                    with gr.Row():
                        webcam = gr.Image(
                            source="webcam",
                            streaming=True,
                            label="Webcam Feed"
                        )
                        live_question = gr.Textbox(
                            label="Question",
                            placeholder="What do you see?"
                        )
                        live_output = gr.Markdown(label="Live Analysis")
                    
                    # Note: Webcam streaming would need additional implementation
                    gr.Markdown("*Note: Webcam feature requires additional setup*")
                
                # Tab 5: API Documentation
                with gr.TabItem("📖 API"):
                    gr.Markdown("""
                    ## API Usage
                    
                    ### Python Client
                    ```python
                    import requests
                    
                    # Single query
                    response = requests.post(
                        "http://localhost:8000/predict",
                        files={"image": open("image.jpg", "rb")},
                        data={"question": "What is this?"}
                    )
                    print(response.json())
                    ```
                    
                    ### cURL
                    ```bash
                    curl -X POST "http://localhost:8000/predict" \\
                      -F "image=@image.jpg" \\
                      -F "question=What color is the car?"
                    ```
                    
                    ### Response Format
                    ```json
                    {
                        "answer": "red",
                        "confidence": 0.95,
                        "inference_time_ms": 12.3,
                        "top_k": [
                            {"answer": "red", "confidence": 0.95},
                            {"answer": "crimson", "confidence": 0.03}
                        ]
                    }
                    ```
                    
                    ### Available Endpoints
                    - `POST /predict` - Single prediction
                    - `POST /batch` - Batch predictions
                    - `GET /models` - List available models
                    - `GET /health` - Health check
                    """)
            
        
        self.demo = demo
        return demo
    
    def launch(self, **kwargs):
        """Launch the demo"""
        return self.demo.launch(**kwargs)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FrEVL Interactive Demo")
    parser.add_argument("--model", type=str, default="frevl-base")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public URL")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    # Create and launch demo
    demo = FrEVLDemo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True
    )


if __name__ == "__main__":
    main()
