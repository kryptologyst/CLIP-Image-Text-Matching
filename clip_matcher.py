import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from typing import List, Tuple, Dict, Optional
import requests
from io import BytesIO
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CLIPImageTextMatcher:
    """
    Modern CLIP implementation for image-text matching with advanced features
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize CLIP model
        
        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14, etc.)
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"Loading CLIP model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        print("✅ CLIP model loaded successfully!")
    
    def encode_image(self, image_input) -> torch.Tensor:
        """
        Encode image(s) using CLIP
        
        Args:
            image_input: PIL Image, image path, or batch of images
            
        Returns:
            Image features tensor
        """
        if isinstance(image_input, str):
            # Load from path
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            image = image_input
            
        # Preprocess and encode
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        return image_features
    
    def encode_text(self, text_input: List[str]) -> torch.Tensor:
        """
        Encode text using CLIP
        
        Args:
            text_input: List of text descriptions
            
        Returns:
            Text features tensor
        """
        text_tokens = clip.tokenize(text_input, truncate=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features
    
    def match_image_text(self, image_input, text_descriptions: List[str]) -> Dict:
        """
        Match image with text descriptions
        
        Args:
            image_input: Image (PIL, path, or tensor)
            text_descriptions: List of text descriptions to match against
            
        Returns:
            Dictionary with similarity scores and rankings
        """
        # Encode image and text
        image_features = self.encode_image(image_input)
        text_features = self.encode_text(text_descriptions)
        
        # Compute similarities
        with torch.no_grad():
            similarities = torch.cosine_similarity(image_features, text_features, dim=-1)
            probabilities = torch.softmax(similarities * 100, dim=-1)  # Temperature scaling
            
        # Create results
        results = []
        for i, (desc, sim, prob) in enumerate(zip(text_descriptions, similarities[0], probabilities[0])):
            results.append({
                'rank': i + 1,
                'description': desc,
                'similarity': float(sim),
                'probability': float(prob)
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result['rank'] = i + 1
            
        return {
            'image_features': image_features.cpu(),
            'text_features': text_features.cpu(),
            'results': results,
            'best_match': results[0] if results else None
        }
    
    def batch_match_images(self, image_paths: List[str], text_descriptions: List[str]) -> List[Dict]:
        """
        Match multiple images with text descriptions
        
        Args:
            image_paths: List of image file paths
            text_descriptions: List of text descriptions
            
        Returns:
            List of matching results for each image
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        for image_path in tqdm(image_paths, desc="Matching images"):
            try:
                result = self.match_image_text(image_path, text_descriptions)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'results': []
                })
        
        return results
    
    def text_to_image_search(self, query_text: str, image_paths: List[str]) -> List[Dict]:
        """
        Search for images using text query
        
        Args:
            query_text: Text description to search for
            image_paths: List of image paths to search through
            
        Returns:
            List of images ranked by relevance to query
        """
        # Encode query text
        text_features = self.encode_text([query_text])
        
        results = []
        
        print(f"Searching {len(image_paths)} images for: '{query_text}'")
        for image_path in tqdm(image_paths, desc="Searching images"):
            try:
                image_features = self.encode_image(image_path)
                
                with torch.no_grad():
                    similarity = torch.cosine_similarity(text_features, image_features, dim=-1)
                    
                results.append({
                    'image_path': image_path,
                    'similarity': float(similarity[0]),
                    'query': query_text
                })
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'similarity': 0.0,
                    'error': str(e),
                    'query': query_text
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results

def create_mock_database() -> Dict:
    """
    Create a mock database with sample image-text pairs for testing
    """
    mock_data = {
        "images": [
            {
                "id": 1,
                "path": "sample_images/cat.jpg",
                "description": "A fluffy orange cat sitting on a windowsill",
                "category": "animals",
                "tags": ["cat", "orange", "fluffy", "windowsill", "indoor"]
            },
            {
                "id": 2,
                "path": "sample_images/dog.jpg", 
                "description": "A golden retriever playing in the park",
                "category": "animals",
                "tags": ["dog", "golden retriever", "park", "playing", "outdoor"]
            },
            {
                "id": 3,
                "path": "sample_images/food.jpg",
                "description": "A delicious pizza with pepperoni and cheese",
                "category": "food",
                "tags": ["pizza", "pepperoni", "cheese", "delicious", "italian"]
            },
            {
                "id": 4,
                "path": "sample_images/nature.jpg",
                "description": "A beautiful mountain landscape at sunset",
                "category": "nature",
                "tags": ["mountain", "sunset", "landscape", "beautiful", "outdoor"]
            },
            {
                "id": 5,
                "path": "sample_images/city.jpg",
                "description": "A busy city street with tall buildings",
                "category": "urban",
                "tags": ["city", "buildings", "street", "urban", "busy"]
            }
        ],
        "text_queries": [
            "Find images of animals",
            "Show me food pictures",
            "Beautiful nature scenes",
            "Urban cityscapes",
            "Cute pets",
            "Delicious meals",
            "Outdoor activities",
            "Indoor scenes"
        ],
        "categories": ["animals", "food", "nature", "urban", "people", "objects"],
        "metadata": {
            "total_images": 5,
            "created_date": "2024-01-01",
            "version": "1.0"
        }
    }
    
    return mock_data

def download_sample_images():
    """
    Download sample images for testing (if they don't exist)
    """
    sample_images = {
        "cat.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
        "dog.jpg": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400", 
        "food.jpg": "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400",
        "nature.jpg": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
        "city.jpg": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400"
    }
    
    os.makedirs("sample_images", exist_ok=True)
    
    for filename, url in sample_images.items():
        filepath = f"sample_images/{filename}"
        if not os.path.exists(filepath):
            try:
                print(f"Downloading {filename}...")
                response = requests.get(url)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")

def main():
    """
    Main demonstration function
    """
    print("🚀 CLIP Image-Text Matching Demo")
    print("=" * 50)
    
    # Initialize CLIP matcher
    matcher = CLIPImageTextMatcher()
    
    # Create mock database
    mock_db = create_mock_database()
    
    # Download sample images
    download_sample_images()
    
    # Example 1: Single image-text matching
    print("\n📸 Example 1: Single Image-Text Matching")
    print("-" * 40)
    
    sample_image = "sample_images/cat.jpg"
    if os.path.exists(sample_image):
        text_descriptions = [
            "A photo of a cat",
            "A photo of a dog", 
            "A bowl of food",
            "A sunny beach",
            "A person riding a bike"
        ]
        
        result = matcher.match_image_text(sample_image, text_descriptions)
        
        print(f"Image: {sample_image}")
        print("Top matches:")
        for match in result['results'][:3]:
            print(f"  {match['rank']}. {match['description']} - {match['probability']:.2%}")
    
    # Example 2: Text-to-image search
    print("\n🔍 Example 2: Text-to-Image Search")
    print("-" * 40)
    
    query = "cute animals"
    image_paths = [f"sample_images/{img['path'].split('/')[-1]}" for img in mock_db['images']]
    image_paths = [path for path in image_paths if os.path.exists(path)]
    
    if image_paths:
        search_results = matcher.text_to_image_search(query, image_paths)
        
        print(f"Query: '{query}'")
        print("Top results:")
        for i, result in enumerate(search_results[:3]):
            if 'error' not in result:
                print(f"  {i+1}. {result['image_path']} - {result['similarity']:.3f}")
    
    # Example 3: Batch processing
    print("\n📊 Example 3: Batch Processing")
    print("-" * 40)
    
    if image_paths:
        batch_results = matcher.batch_match_images(image_paths[:3], text_descriptions)
        
        print("Batch processing results:")
        for result in batch_results:
            if 'error' not in result:
                best_match = result['best_match']
                print(f"  {result['image_path']}: {best_match['description']} ({best_match['probability']:.2%})")
    
    print("\n✅ Demo completed!")
    print("\n💡 Next steps:")
    print("  - Run 'streamlit run app.py' for interactive web interface")
    print("  - Check out the mock database in mock_database.json")
    print("  - Explore different CLIP models and parameters")

if __name__ == "__main__":
    main()
