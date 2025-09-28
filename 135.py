# Project 135. CLIP for image-text matching
# Description:
# CLIP (Contrastive Language–Image Pretraining) by OpenAI learns to understand images and natural language by jointly training on image–text pairs. In this project, we use a pre-trained CLIP model to match an image to its most relevant textual descriptions (or vice versa), enabling applications like zero-shot classification, caption ranking, and visual search.

# Modern Python Implementation Using OpenAI CLIP + Advanced Features
# Install dependencies: pip install -r requirements.txt

import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
from io import BytesIO

def download_sample_image():
    """Download a sample image if it doesn't exist"""
    if not os.path.exists("cat.jpg"):
        try:
            print("Downloading sample image...")
            url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
            response = requests.get(url)
            response.raise_for_status()
            
            with open("cat.jpg", 'wb') as f:
                f.write(response.content)
            print("✅ Sample image downloaded!")
        except Exception as e:
            print(f"❌ Failed to download sample image: {e}")
            print("Please provide your own image named 'cat.jpg'")

def main():
    """Enhanced CLIP demonstration with modern features"""
    print("🚀 CLIP Image-Text Matching Demo")
    print("=" * 50)
    
    # Load CLIP model and preprocessing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("✅ CLIP model loaded successfully!")
    
    # Download sample image
    download_sample_image()
    
    # Load and preprocess image
    image_path = "cat.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Please provide an image named 'cat.jpg' or run the demo again to download a sample.")
        return
    
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        print(f"✅ Image loaded: {image_path}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Enhanced candidate text descriptions
    text_descriptions = [
        "A photo of a cat",
        "A photo of a dog",
        "A bowl of food",
        "A sunny beach",
        "A person riding a bike",
        "A fluffy orange cat sitting on a windowsill",
        "A cute pet animal",
        "An outdoor scene",
        "A domestic animal",
        "Something furry and cute"
    ]
    
    print(f"\n📝 Analyzing image against {len(text_descriptions)} text descriptions...")
    
    # Tokenize text inputs
    text_tokens = clip.tokenize(text_descriptions, truncate=True).to(device)
    
    # Encode image and text with CLIP
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarities = torch.cosine_similarity(image_features, text_features, dim=-1)
        probabilities = torch.softmax(similarities * 100, dim=-1)  # Temperature scaling
        
        # Compute logits for comparison
        logits_per_image, _ = model(image, text_tokens)
        probs_original = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    # Display results with enhanced formatting
    print(f"\n🖼️ Input Image: {image_path}")
    print("=" * 50)
    
    # Create results list with both methods
    results = []
    for i, (desc, sim, prob, prob_orig) in enumerate(zip(text_descriptions, similarities[0], probabilities[0], probs_original[0])):
        results.append({
            'description': desc,
            'similarity': float(sim),
            'probability_normalized': float(prob),
            'probability_original': float(prob_orig)
        })
    
    # Sort by similarity
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    print("🧠 Top Text Matches for Image:")
    print("-" * 40)
    for i, result in enumerate(results[:5]):
        print(f"{i+1:2d}. {result['description']:<40} | Similarity: {result['similarity']:.3f} | Prob: {result['probability_normalized']:.2%}")
    
    # Additional analysis
    print(f"\n📊 Analysis Summary:")
    print(f"   • Best match: {results[0]['description']}")
    print(f"   • Confidence: {results[0]['probability_normalized']:.2%}")
    print(f"   • Similarity score: {results[0]['similarity']:.3f}")
    print(f"   • Total descriptions analyzed: {len(text_descriptions)}")
    
    # Display image
    try:
        plt.figure(figsize=(8, 6))
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"📷 Input Image: {os.path.basename(image_path)}", fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Note: Could not display image: {e}")
    
    print(f"\n💡 What This Project Demonstrates:")
    print("   • Uses CLIP to match images and text in a shared embedding space")
    print("   • Computes cosine similarity to rank text prompts against the image")
    print("   • Demonstrates zero-shot learning (no fine-tuning needed!)")
    print("   • Shows both similarity scores and probability distributions")
    print("   • Includes modern error handling and user feedback")
    
    print(f"\n🚀 Next Steps:")
    print("   • Run 'streamlit run app.py' for interactive web interface")
    print("   • Try different images and text descriptions")
    print("   • Explore batch processing with multiple images")
    print("   • Experiment with different CLIP model variants")

if __name__ == "__main__":
    main()