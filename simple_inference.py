#!/usr/bin/env python3
"""
Simple FrEVL Inference Example
Demonstrates basic model usage for VQA and image-text retrieval
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Import FrEVL
from frevl import load_model, preprocess_image, create_attention_map


def simple_vqa_example():
    """Simple VQA inference example"""
    print("="*50)
    print("Simple VQA Example")
    print("="*50)
    
    # Load model
    print("Loading model...")
    model = load_model("frevl-base", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess image
    image_path = "examples/images/dog.jpg"  # Replace with your image
    image = Image.open(image_path).convert('RGB')
    
    # Prepare questions
    questions = [
        "What animal is in the image?",
        "What color is the animal?",
        "Is this indoor or outdoor?",
        "What is the animal doing?"
    ]
    
    # Process image
    image_tensor = preprocess_image(image).unsqueeze(0)  # Add batch dimension
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    print(f"\nImage: {image_path}")
    print("Questions and Answers:")
    print("-" * 40)
    
    # Get answers for each question
    model.eval()
    with torch.no_grad():
        for question in questions:
            output = model(
                images=image_tensor,
                text=[question],
                task="vqa"
            )
            
            # Get predicted answer (would need answer vocabulary in real scenario)
            confidence = output['confidence'][0].item()
            answer_idx = output['predictions'][0].item()
            
            print(f"Q: {question}")
            print(f"A: Answer #{answer_idx} (confidence: {confidence:.2%})")
            print()


def batch_inference_example():
    """Batch inference example"""
    print("="*50)
    print("Batch Inference Example")
    print("="*50)
    
    # Load model
    model = load_model("frevl-base", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare batch of images and questions
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)  # Dummy images
    questions = [
        "What is this?",
        "What color is it?",
        "Where is this?",
        "Who is in the image?"
    ]
    
    if torch.cuda.is_available():
        images = images.cuda()
    
    # Batch inference
    model.eval()
    with torch.no_grad():
        output = model(
            images=images,
            text=questions,
            task="vqa"
        )
    
    print(f"Batch size: {batch_size}")
    print(f"Output shape: {output['logits'].shape}")
    print(f"Predictions: {output['predictions'].tolist()}")
    print(f"Confidences: {output['confidence'].tolist()}")


def attention_visualization_example():
    """Visualize attention weights"""
    print("="*50)
    print("Attention Visualization Example")
    print("="*50)
    
    # Load model
    model = load_model("frevl-base", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load image
    image_path = "examples/images/scene.jpg"  # Replace with your image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(image).unsqueeze(0)
    
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    question = "What is the main object in this image?"
    
    # Get predictions with attention
    model.eval()
    with torch.no_grad():
        output = model(
            images=image_tensor,
            text=[question],
            task="vqa",
            return_attention=True
        )
    
    if "attention" in output and output["attention"] is not None:
        # Create attention map
        attention_weights = output["attention"][0]  # Get first sample
        attention_map = create_attention_map(image, attention_weights)
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(attention_map)
        axes[1].set_title(f"Attention Map\nQ: {question}")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig("attention_visualization.png")
        print("Attention visualization saved to attention_visualization.png")
    else:
        print("Attention weights not available")


def retrieval_example():
    """Image-text retrieval example"""
    print("="*50)
    print("Image-Text Retrieval Example")
    print("="*50)
    
    # Load model
    model = load_model("frevl-base", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare images and captions
    num_images = 5
    images = torch.randn(num_images, 3, 224, 224)
    captions = [
        "A dog playing in the park",
        "A cat sleeping on a sofa",
        "People walking on the beach",
        "A car driving on the highway",
        "Food on a restaurant table"
    ]
    
    if torch.cuda.is_available():
        images = images.cuda()
    
    # Compute similarities
    model.eval()
    with torch.no_grad():
        output = model(
            images=images,
            text=captions,
            task="retrieval"
        )
    
    similarity_matrix = output["image_text_similarity"].cpu().numpy()
    
    print("Image-Text Similarity Matrix:")
    print("-" * 40)
    
    # Print similarity scores
    for i in range(num_images):
        print(f"\nImage {i+1}:")
        for j in range(len(captions)):
            score = similarity_matrix[i, j]
            print(f"  Caption {j+1}: {score:.3f} - {captions[j][:30]}...")
    
    # Find best matches
    print("\nBest Matches:")
    print("-" * 40)
    for i in range(num_images):
        best_caption_idx = similarity_matrix[i].argmax()
        best_score = similarity_matrix[i, best_caption_idx]
        print(f"Image {i+1} -> Caption {best_caption_idx+1} (score: {best_score:.3f})")


def custom_model_example():
    """Example with custom model configuration"""
    print("="*50)
    print("Custom Model Configuration Example")
    print("="*50)
    
    from frevl import FrEVL, FrEVLConfig
    
    # Create custom configuration
    config = FrEVLConfig(
        clip_model="ViT-B/16",  # Use different CLIP backbone
        hidden_dim=512,          # Smaller hidden dimension
        num_layers=4,            # Fewer layers
        num_heads=8,             # Fewer attention heads
        dropout=0.2,             # More dropout
        use_flash_attention=True # Enable Flash Attention
    )
    
    # Create model with custom config
    model = FrEVL(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Configuration:")
    print(f"  CLIP Model: {config.clip_model}")
    print(f"  Hidden Dim: {config.hidden_dim}")
    print(f"  Num Layers: {config.num_layers}")
    print(f"  Num Heads: {config.num_heads}")
    print(f"\nModel Size:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Percentage Trainable: {100 * trainable_params / total_params:.1f}%")


def speed_comparison_example():
    """Compare inference speed with different optimizations"""
    print("="*50)
    print("Speed Comparison Example")
    print("="*50)
    
    import time
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("frevl-base", device=device)
    
    # Prepare input
    batch_size = 8
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    questions = ["What is this?"] * batch_size
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(images=images, text=questions)
    
    # Test configurations
    configs = {
        "Normal": lambda: model(images=images, text=questions),
    }
    
    if device == "cuda":
        # Add FP16 test
        from torch.cuda.amp import autocast
        configs["FP16"] = lambda: None  # Would need autocast context
        
        # Add compiled version if PyTorch 2.0+
        if torch.__version__ >= "2.0":
            compiled_model = torch.compile(model)
            configs["Compiled"] = lambda: compiled_model(images=images, text=questions)
    
    # Run benchmarks
    num_iterations = 100
    results = {}
    
    for name, inference_fn in configs.items():
        if name == "FP16" and device == "cuda":
            # Special handling for FP16
            times = []
            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                with torch.no_grad(), autocast():
                    _ = model(images=images, text=questions)
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
        elif name != "FP16":
            times = []
            for _ in range(num_iterations):
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                with torch.no_grad():
                    _ = inference_fn()
                
                if device == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
        else:
            continue
        
        avg_time = sum(times) / len(times)
        results[name] = avg_time
        print(f"{name:10s}: {avg_time:.2f} ms (throughput: {batch_size * 1000 / avg_time:.1f} samples/s)")
    
    # Show speedup
    if "Normal" in results:
        baseline = results["Normal"]
        print("\nSpeedup relative to baseline:")
        for name, time_ms in results.items():
            speedup = baseline / time_ms
            print(f"  {name:10s}: {speedup:.2f}x")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("FrEVL Usage Examples")
    print("="*60 + "\n")
    
    # Check if example images exist
    example_dir = Path("examples/images")
    if not example_dir.exists():
        print("Creating example images directory...")
        example_dir.mkdir(parents=True, exist_ok=True)
        print(f"Please add images to {example_dir}")
        print("Using dummy tensors for demonstration\n")
    
    # Run examples
    try:
        # 1. Simple VQA
        if (example_dir / "dog.jpg").exists():
            simple_vqa_example()
        else:
            print("Skipping VQA example - add dog.jpg to examples/images/")
        
        print("\n")
        
        # 2. Batch inference
        batch_inference_example()
        print("\n")
        
        # 3. Attention visualization
        if (example_dir / "scene.jpg").exists():
            attention_visualization_example()
        else:
            print("Skipping attention example - add scene.jpg to examples/images/")
        
        print("\n")
        
        # 4. Retrieval
        retrieval_example()
        print("\n")
        
        # 5. Custom model
        custom_model_example()
        print("\n")
        
        # 6. Speed comparison
        speed_comparison_example()
        
    except Exception as e:
        print(f"Error in example: {e}")
        print("Make sure you have downloaded the model first:")
        print("  python scripts/download_models.py --model frevl-base")
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
