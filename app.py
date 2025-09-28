import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
import json
import os
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from clip_matcher import CLIPImageTextMatcher, create_mock_database, download_sample_images

# Page configuration
st.set_page_config(
    page_title="CLIP Image-Text Matching",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_clip_model(model_name="ViT-B/32"):
    """Load CLIP model with caching"""
    return CLIPImageTextMatcher(model_name)

@st.cache_data
def load_mock_database():
    """Load mock database with caching"""
    return create_mock_database()

def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ CLIP Image-Text Matching</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced AI-powered image and text understanding using OpenAI's CLIP model")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    model_options = {
        "ViT-B/32": "Fast, good quality (recommended)",
        "ViT-B/16": "Better quality, slower",
        "ViT-L/14": "Best quality, slowest"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select CLIP Model:",
        options=list(model_options.keys()),
        help="Choose model based on speed vs quality trade-off"
    )
    
    st.sidebar.markdown(f"**Model Info:** {model_options[selected_model]}")
    
    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.markdown(f"**Device:** {device}")
    
    # Load model
    with st.spinner("Loading CLIP model..."):
        matcher = load_clip_model(selected_model)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🖼️ Image Analysis", 
        "🔍 Text Search", 
        "📊 Batch Processing", 
        "📚 Mock Database", 
        "ℹ️ About"
    ])
    
    with tab1:
        st.header("🖼️ Image Analysis")
        st.markdown("Upload an image and get AI-powered text descriptions and matches.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                help="Upload an image to analyze with CLIP"
            )
            
            # Or use sample images
            st.markdown("**Or use sample images:**")
            sample_images = {
                "Cat": "sample_images/cat.jpg",
                "Dog": "sample_images/dog.jpg", 
                "Food": "sample_images/food.jpg",
                "Nature": "sample_images/nature.jpg",
                "City": "sample_images/city.jpg"
            }
            
            selected_sample = st.selectbox("Select sample image:", ["None"] + list(sample_images.keys()))
            
            # Text descriptions input
            st.subheader("Text Descriptions")
            text_input = st.text_area(
                "Enter text descriptions (one per line):",
                value="A photo of a cat\nA photo of a dog\nA bowl of food\nA sunny beach\nA person riding a bike",
                height=150,
                help="Enter multiple text descriptions to match against the image"
            )
            
            text_descriptions = [line.strip() for line in text_input.split('\n') if line.strip()]
            
            # Analyze button
            analyze_btn = st.button("🔍 Analyze Image", type="primary")
        
        with col2:
            st.subheader("Results")
            
            if analyze_btn:
                if uploaded_file or selected_sample != "None":
                    # Determine image source
                    if uploaded_file:
                        image = Image.open(uploaded_file)
                        image_name = uploaded_file.name
                    else:
                        image_path = sample_images[selected_sample]
                        if os.path.exists(image_path):
                            image = Image.open(image_path)
                            image_name = selected_sample
                        else:
                            st.error(f"Sample image not found: {image_path}")
                            st.stop()
                    
                    # Display image
                    st.image(image, caption=f"Analyzing: {image_name}", use_column_width=True)
                    
                    # Analyze with CLIP
                    with st.spinner("Analyzing image with CLIP..."):
                        try:
                            result = matcher.match_image_text(image, text_descriptions)
                            
                            # Display results
                            st.success("✅ Analysis complete!")
                            
                            # Metrics
                            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                            
                            with col_metrics1:
                                st.metric("Best Match", f"{result['best_match']['probability']:.1%}")
                            
                            with col_metrics2:
                                st.metric("Similarity Score", f"{result['best_match']['similarity']:.3f}")
                            
                            with col_metrics3:
                                st.metric("Total Descriptions", len(text_descriptions))
                            
                            # Results table
                            st.subheader("📊 Detailed Results")
                            
                            results_df = pd.DataFrame(result['results'])
                            results_df = results_df[['rank', 'description', 'similarity', 'probability']]
                            results_df['probability'] = results_df['probability'].apply(lambda x: f"{x:.2%}")
                            results_df['similarity'] = results_df['similarity'].apply(lambda x: f"{x:.3f}")
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Visualization
                            st.subheader("📈 Similarity Visualization")
                            
                            fig = px.bar(
                                results_df, 
                                x='description', 
                                y='similarity',
                                title="Similarity Scores",
                                color='similarity',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(
                                xaxis_tickangle=-45,
                                height=400,
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error analyzing image: {str(e)}")
                else:
                    st.warning("Please upload an image or select a sample image.")
    
    with tab2:
        st.header("🔍 Text-to-Image Search")
        st.markdown("Search through images using natural language queries.")
        
        # Load mock database
        mock_db = load_mock_database()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Search Query")
            
            # Query input
            query = st.text_input(
                "Enter your search query:",
                value="cute animals",
                help="Describe what you're looking for in the images"
            )
            
            # Predefined queries
            st.markdown("**Quick queries:**")
            quick_queries = mock_db['text_queries']
            selected_query = st.selectbox("Or select a predefined query:", ["Custom"] + quick_queries)
            
            if selected_query != "Custom":
                query = selected_query
                st.text_input("Current query:", value=query, disabled=True)
            
            # Search button
            search_btn = st.button("🔍 Search Images", type="primary")
        
        with col2:
            st.subheader("Search Results")
            
            if search_btn and query:
                # Get available images
                image_paths = []
                for img_data in mock_db['images']:
                    img_path = img_data['path']
                    if os.path.exists(img_path):
                        image_paths.append(img_path)
                
                if image_paths:
                    with st.spinner(f"Searching {len(image_paths)} images..."):
                        try:
                            search_results = matcher.text_to_image_search(query, image_paths)
                            
                            st.success(f"✅ Found {len(search_results)} results!")
                            
                            # Display top results
                            st.subheader("🏆 Top Results")
                            
                            for i, result in enumerate(search_results[:5]):
                                if 'error' not in result:
                                    with st.container():
                                        col_img, col_info = st.columns([1, 2])
                                        
                                        with col_img:
                                            try:
                                                img = Image.open(result['image_path'])
                                                st.image(img, width=150)
                                            except:
                                                st.write("Image not found")
                                        
                                        with col_info:
                                            similarity = result['similarity']
                                            img_name = os.path.basename(result['image_path'])
                                            
                                            st.write(f"**{i+1}. {img_name}**")
                                            st.write(f"Similarity: {similarity:.3f}")
                                            
                                            # Find description from mock DB
                                            for img_data in mock_db['images']:
                                                if img_data['path'] == result['image_path']:
                                                    st.write(f"Description: {img_data['description']}")
                                                    st.write(f"Category: {img_data['category']}")
                                                    break
                                            
                                            # Progress bar for similarity
                                            st.progress(similarity)
                                        
                                        st.divider()
                            
                            # Results summary
                            st.subheader("📊 Search Summary")
                            
                            similarities = [r['similarity'] for r in search_results if 'error' not in r]
                            if similarities:
                                avg_similarity = np.mean(similarities)
                                max_similarity = max(similarities)
                                min_similarity = min(similarities)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Average Similarity", f"{avg_similarity:.3f}")
                                with col2:
                                    st.metric("Highest Similarity", f"{max_similarity:.3f}")
                                with col3:
                                    st.metric("Lowest Similarity", f"{min_similarity:.3f}")
                        
                        except Exception as e:
                            st.error(f"Error during search: {str(e)}")
                else:
                    st.warning("No sample images found. Please run the demo first to download sample images.")
            elif search_btn:
                st.warning("Please enter a search query.")
    
    with tab3:
        st.header("📊 Batch Processing")
        st.markdown("Process multiple images at once for efficient analysis.")
        
        # Load mock database
        mock_db = load_mock_database()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Batch Configuration")
            
            # Text descriptions for batch
            batch_text_input = st.text_area(
                "Text descriptions for batch processing:",
                value="A photo of a cat\nA photo of a dog\nA bowl of food\nA sunny beach\nA person riding a bike",
                height=150
            )
            
            batch_text_descriptions = [line.strip() for line in batch_text_input.split('\n') if line.strip()]
            
            # Image selection
            st.markdown("**Select images to process:**")
            available_images = []
            for img_data in mock_db['images']:
                if os.path.exists(img_data['path']):
                    available_images.append(img_data)
            
            selected_images = st.multiselect(
                "Choose images:",
                options=[img['path'] for img in available_images],
                default=[img['path'] for img in available_images[:3]]
            )
            
            # Process button
            process_btn = st.button("🚀 Process Batch", type="primary")
        
        with col2:
            st.subheader("Batch Results")
            
            if process_btn and selected_images and batch_text_descriptions:
                with st.spinner(f"Processing {len(selected_images)} images..."):
                    try:
                        batch_results = matcher.batch_match_images(selected_images, batch_text_descriptions)
                        
                        st.success(f"✅ Processed {len(batch_results)} images!")
                        
                        # Create results dataframe
                        results_data = []
                        for result in batch_results:
                            if 'error' not in result:
                                img_name = os.path.basename(result['image_path'])
                                best_match = result['best_match']
                                results_data.append({
                                    'Image': img_name,
                                    'Best Match': best_match['description'],
                                    'Probability': best_match['probability'],
                                    'Similarity': best_match['similarity']
                                })
                        
                        if results_data:
                            results_df = pd.DataFrame(results_data)
                            results_df['Probability'] = results_df['Probability'].apply(lambda x: f"{x:.2%}")
                            results_df['Similarity'] = results_df['Similarity'].apply(lambda x: f"{x:.3f}")
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Summary statistics
                            st.subheader("📈 Batch Summary")
                            
                            probabilities = [float(r['Probability'].replace('%', '')) for r in results_data]
                            similarities = [float(r['Similarity']) for r in results_data]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Images Processed", len(results_data))
                            with col2:
                                st.metric("Avg Probability", f"{np.mean(probabilities):.1f}%")
                            with col3:
                                st.metric("Avg Similarity", f"{np.mean(similarities):.3f}")
                            with col4:
                                st.metric("Best Match", f"{max(probabilities):.1f}%")
                        
                    except Exception as e:
                        st.error(f"Error during batch processing: {str(e)}")
            elif process_btn:
                st.warning("Please select images and enter text descriptions.")
    
    with tab4:
        st.header("📚 Mock Database")
        st.markdown("Explore the sample image database used for demonstrations.")
        
        # Load mock database
        mock_db = load_mock_database()
        
        # Database info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Images", mock_db['metadata']['total_images'])
        with col2:
            st.metric("Categories", len(mock_db['categories']))
        with col3:
            st.metric("Version", mock_db['metadata']['version'])
        
        # Images table
        st.subheader("🖼️ Image Database")
        
        images_df = pd.DataFrame(mock_db['images'])
        st.dataframe(images_df, use_container_width=True)
        
        # Categories
        st.subheader("📂 Categories")
        for category in mock_db['categories']:
            category_images = [img for img in mock_db['images'] if img['category'] == category]
            st.write(f"**{category.title()}:** {len(category_images)} images")
        
        # Download sample images button
        st.subheader("📥 Download Sample Images")
        if st.button("⬇️ Download Sample Images"):
            with st.spinner("Downloading sample images..."):
                download_sample_images()
            st.success("✅ Sample images downloaded!")
            st.rerun()
    
    with tab5:
        st.header("ℹ️ About CLIP Image-Text Matching")
        
        st.markdown("""
        ### What is CLIP?
        
        **CLIP (Contrastive Language–Image Pretraining)** is a neural network developed by OpenAI that learns to understand images and natural language by jointly training on image–text pairs. It creates a shared embedding space where similar images and text descriptions are close together.
        
        ### Key Features
        
        - **Zero-shot Learning**: No fine-tuning required for new tasks
        - **Multimodal Understanding**: Works with both images and text
        - **High Accuracy**: State-of-the-art performance on various benchmarks
        - **Flexible**: Can be used for classification, search, and more
        
        ### How It Works
        
        1. **Image Encoding**: Images are processed through a vision transformer
        2. **Text Encoding**: Text descriptions are processed through a transformer
        3. **Similarity Computation**: Cosine similarity between embeddings
        4. **Ranking**: Results are ranked by similarity scores
        
        ### Applications
        
        - Image classification and tagging
        - Visual search and retrieval
        - Content moderation
        - Accessibility tools
        - Creative applications
        
        ### Technical Details
        
        - **Model**: OpenAI CLIP (ViT-B/32, ViT-B/16, ViT-L/14)
        - **Framework**: PyTorch
        - **Preprocessing**: Standard CLIP preprocessing pipeline
        - **Similarity**: Cosine similarity in embedding space
        
        ### Performance Tips
        
        - Use ViT-B/32 for speed, ViT-L/14 for accuracy
        - GPU acceleration recommended for large batches
        - Higher resolution images generally perform better
        - More descriptive text queries yield better results
        """)
        
        # Model comparison
        st.subheader("🔬 Model Comparison")
        
        model_comparison = pd.DataFrame({
            'Model': ['ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
            'Parameters': ['151M', '151M', '427M'],
            'Speed': ['Fast', 'Medium', 'Slow'],
            'Quality': ['Good', 'Better', 'Best'],
            'Recommended For': ['General use', 'Better accuracy', 'Maximum quality']
        })
        
        st.dataframe(model_comparison, use_container_width=True)

if __name__ == "__main__":
    main()
