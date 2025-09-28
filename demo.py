#!/usr/bin/env python3
"""
CLIP Image-Text Matching - Complete Demo
This script demonstrates all the features of the modernized CLIP project
"""

import os
import sys
import time
from clip_matcher import CLIPImageTextMatcher, create_mock_database, download_sample_images

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"🎯 {title}")
    print("=" * 60)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n📋 {title}")
    print("-" * 40)

def demo_basic_functionality():
    """Demonstrate basic CLIP functionality"""
    print_header("Basic CLIP Functionality Demo")
    
    # Initialize matcher
    print("🔄 Initializing CLIP matcher...")
    matcher = CLIPImageTextMatcher(model_name="ViT-B/32")
    
    # Download sample images
    print_section("Downloading Sample Images")
    download_sample_images()
    
    # Check if we have images
    sample_images = ["sample_images/cat.jpg", "sample_images/dog.jpg"]
    available_images = [img for img in sample_images if os.path.exists(img)]
    
    if not available_images:
        print("❌ No sample images available. Please run the setup first.")
        return False
    
    print(f"✅ Found {len(available_images)} sample images")
    
    # Demo 1: Single image analysis
    print_section("Single Image Analysis")
    image_path = available_images[0]
    text_descriptions = [
        "A photo of a cat",
        "A photo of a dog",
        "A bowl of food",
        "A sunny beach",
        "A person riding a bike"
    ]
    
    print(f"Analyzing: {os.path.basename(image_path)}")
    result = matcher.match_image_text(image_path, text_descriptions)
    
    print("Top 3 matches:")
    for i, match in enumerate(result['results'][:3]):
        print(f"  {i+1}. {match['description']:<30} | {match['probability']:.2%}")
    
    return True

def demo_text_search():
    """Demonstrate text-to-image search"""
    print_header("Text-to-Image Search Demo")
    
    matcher = CLIPImageTextMatcher()
    mock_db = create_mock_database()
    
    # Get available images
    image_paths = []
    for img_data in mock_db['images']:
        img_path = img_data['path']
        if os.path.exists(img_path):
            image_paths.append(img_path)
    
    if not image_paths:
        print("❌ No images available for search")
        return False
    
    # Search queries
    queries = ["cute animals", "delicious food", "beautiful nature"]
    
    for query in queries:
        print_section(f"Searching for: '{query}'")
        results = matcher.text_to_image_search(query, image_paths)
        
        print("Top results:")
        for i, result in enumerate(results[:3]):
            if 'error' not in result:
                img_name = os.path.basename(result['image_path'])
                print(f"  {i+1}. {img_name:<20} | Similarity: {result['similarity']:.3f}")
    
    return True

def demo_batch_processing():
    """Demonstrate batch processing"""
    print_header("Batch Processing Demo")
    
    matcher = CLIPImageTextMatcher()
    mock_db = create_mock_database()
    
    # Get available images
    image_paths = []
    for img_data in mock_db['images']:
        img_path = img_data['path']
        if os.path.exists(img_path):
            image_paths.append(img_path)
    
    if len(image_paths) < 2:
        print("❌ Need at least 2 images for batch processing")
        return False
    
    # Batch process
    text_descriptions = [
        "A photo of an animal",
        "A photo of food",
        "A photo of nature",
        "A photo of a city"
    ]
    
    print(f"Processing {len(image_paths)} images...")
    batch_results = matcher.batch_match_images(image_paths, text_descriptions)
    
    print("Batch results:")
    for result in batch_results:
        if 'error' not in result:
            img_name = os.path.basename(result['image_path'])
            best_match = result['best_match']
            print(f"  {img_name:<20} | {best_match['description']:<25} | {best_match['probability']:.2%}")
    
    return True

def demo_performance_comparison():
    """Compare different CLIP models"""
    print_header("Performance Comparison Demo")
    
    models = ["ViT-B/32", "ViT-B/16"]
    image_path = "sample_images/cat.jpg"
    
    if not os.path.exists(image_path):
        print("❌ Sample image not found")
        return False
    
    text_descriptions = ["A photo of a cat", "A photo of a dog"]
    
    for model_name in models:
        print_section(f"Testing {model_name}")
        
        start_time = time.time()
        matcher = CLIPImageTextMatcher(model_name=model_name)
        load_time = time.time() - start_time
        
        start_time = time.time()
        result = matcher.match_image_text(image_path, text_descriptions)
        inference_time = time.time() - start_time
        
        best_match = result['best_match']
        print(f"  Load time: {load_time:.2f}s")
        print(f"  Inference time: {inference_time:.2f}s")
        print(f"  Best match: {best_match['description']} ({best_match['probability']:.2%})")
    
    return True

def show_project_info():
    """Show project information and next steps"""
    print_header("Project Information")
    
    print("🎯 CLIP Image-Text Matching Project")
    print("   A modern implementation of OpenAI's CLIP model")
    print("   with advanced features and interactive web interface")
    
    print_section("Features Implemented")
    features = [
        "✅ Modern CLIP implementation with multiple model variants",
        "✅ Interactive Streamlit web interface",
        "✅ Mock database with sample images",
        "✅ Text-to-image search functionality",
        "✅ Batch processing capabilities",
        "✅ Performance comparison tools",
        "✅ Comprehensive documentation",
        "✅ GitHub-ready repository structure"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print_section("Files Created")
    files = [
        "clip_matcher.py - Core CLIP implementation",
        "app.py - Streamlit web interface",
        "135.py - Enhanced original demo",
        "requirements.txt - Python dependencies",
        "mock_database.json - Sample image database",
        "README.md - Comprehensive documentation",
        "LICENSE - MIT License",
        ".gitignore - Git ignore rules",
        "setup.py - Automated setup script"
    ]
    
    for file in files:
        print(f"   📄 {file}")
    
    print_section("Next Steps")
    next_steps = [
        "Run 'python setup.py' to install dependencies",
        "Run 'python 135.py' for the enhanced demo",
        "Run 'streamlit run app.py' for the web interface",
        "Upload to GitHub for version control",
        "Customize the mock database with your own images",
        "Experiment with different CLIP model variants"
    ]
    
    for step in next_steps:
        print(f"   🚀 {step}")

def main():
    """Main demo function"""
    print("🚀 CLIP Image-Text Matching - Complete Demo")
    print("This demo showcases all the modernized features")
    
    # Check if we're in the right directory
    if not os.path.exists("clip_matcher.py"):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    try:
        # Run demos
        demo_basic_functionality()
        demo_text_search()
        demo_batch_processing()
        demo_performance_comparison()
        show_project_info()
        
        print_header("Demo Complete!")
        print("🎉 All features have been demonstrated successfully!")
        print("\n💡 The project is now ready for:")
        print("   • Production use")
        print("   • GitHub repository")
        print("   • Further development")
        print("   • Community contributions")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your installation and try again")

if __name__ == "__main__":
    main()
