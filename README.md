# CLIP Image-Text Matching

A modern implementation of OpenAI's CLIP model for advanced image-text matching with an interactive web interface.

## Features

- **Modern CLIP Implementation**: Uses the latest OpenAI CLIP models (ViT-B/32, ViT-B/16, ViT-L/14)
- **Interactive Web UI**: Beautiful Streamlit interface for easy interaction
- **Multiple Analysis Modes**:
  - Single image analysis with custom text descriptions
  - Text-to-image search across multiple images
  - Batch processing for multiple images
- **Mock Database**: Pre-configured sample images and descriptions for testing
- **Advanced Visualizations**: Interactive charts and similarity metrics
- **Zero-shot Learning**: No fine-tuning required
- **GPU Acceleration**: Automatic CUDA support when available

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/clip-image-text-matching.git
   cd clip-image-text-matching
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo**:
   ```bash
   python clip_matcher.py
   ```

4. **Launch the web interface**:
   ```bash
   streamlit run app.py
   ```

### First Run

The first time you run the application, it will:
- Download the CLIP model (may take a few minutes)
- Download sample images for testing
- Create the mock database

## Usage

### Command Line Interface

```python
from clip_matcher import CLIPImageTextMatcher

# Initialize the matcher
matcher = CLIPImageTextMatcher(model_name="ViT-B/32")

# Analyze a single image
result = matcher.match_image_text("path/to/image.jpg", [
    "A photo of a cat",
    "A photo of a dog",
    "A bowl of food"
])

print(f"Best match: {result['best_match']['description']}")
print(f"Confidence: {result['best_match']['probability']:.2%}")
```

### Web Interface

1. **Image Analysis Tab**: Upload an image and get AI-powered text matches
2. **Text Search Tab**: Search through images using natural language queries
3. **Batch Processing Tab**: Process multiple images efficiently
4. **Mock Database Tab**: Explore the sample image database

## 🔧 Configuration

### Model Selection

Choose the CLIP model based on your needs:

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| ViT-B/32 | Fast | Good | General use, real-time applications |
| ViT-B/16 | Medium | Better | Balanced performance |
| ViT-L/14 | Slow | Best | Maximum accuracy, offline processing |

### Device Configuration

The application automatically detects and uses GPU acceleration when available:
- **CUDA**: Automatic detection and usage
- **CPU**: Fallback for systems without GPU
- **Manual**: Override device selection in code

## API Reference

### CLIPImageTextMatcher Class

#### `__init__(model_name="ViT-B/32", device=None)`
Initialize the CLIP model.

**Parameters:**
- `model_name` (str): CLIP model variant
- `device` (str, optional): Device to run on ('cuda', 'cpu', or None for auto-detection)

#### `match_image_text(image_input, text_descriptions)`
Match an image with text descriptions.

**Parameters:**
- `image_input`: PIL Image, image path, or image tensor
- `text_descriptions` (List[str]): List of text descriptions to match against

**Returns:**
- Dictionary with similarity scores, probabilities, and rankings

#### `text_to_image_search(query_text, image_paths)`
Search for images using a text query.

**Parameters:**
- `query_text` (str): Text description to search for
- `image_paths` (List[str]): List of image paths to search through

**Returns:**
- List of images ranked by relevance to query

#### `batch_match_images(image_paths, text_descriptions)`
Process multiple images with text descriptions.

**Parameters:**
- `image_paths` (List[str]): List of image file paths
- `text_descriptions` (List[str]): List of text descriptions

**Returns:**
- List of matching results for each image

## Project Structure

```
clip-image-text-matching/
├── app.py                 # Streamlit web interface
├── clip_matcher.py        # Core CLIP implementation
├── 135.py                 # Original implementation
├── requirements.txt       # Python dependencies
├── mock_database.json     # Sample image database
├── sample_images/         # Downloaded sample images
├── README.md             # This file
├── .gitignore           # Git ignore rules
└── LICENSE              # MIT License
```

## Use Cases

### Content Moderation
- Automatically detect inappropriate content in images
- Match images with policy descriptions

### Visual Search
- Find similar images using text queries
- E-commerce product search

### Accessibility
- Generate alt text for images
- Image description for visually impaired users

### Creative Applications
- Art style matching
- Mood-based image filtering
- Content recommendation

## Technical Details

### Architecture
- **Vision Encoder**: Vision Transformer (ViT)
- **Text Encoder**: Transformer-based text encoder
- **Similarity**: Cosine similarity in shared embedding space
- **Framework**: PyTorch with OpenAI CLIP

### Performance
- **Inference Speed**: ~100ms per image (ViT-B/32, GPU)
- **Memory Usage**: ~1GB VRAM (ViT-B/32)
- **Accuracy**: State-of-the-art on various benchmarks

### Preprocessing
- **Images**: Resize to 224x224, normalize with ImageNet stats
- **Text**: Tokenize with CLIP tokenizer, max length 77 tokens

## 🛠️ Development

### Adding New Features

1. **Custom Models**: Extend `CLIPImageTextMatcher` class
2. **New Analysis Types**: Add methods to the matcher class
3. **UI Components**: Extend Streamlit interface in `app.py`

### Testing

```bash
# Run the demo
python clip_matcher.py

# Test specific functionality
python -c "from clip_matcher import CLIPImageTextMatcher; matcher = CLIPImageTextMatcher(); print('✅ Model loaded successfully')"
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **OpenAI**: For developing the CLIP model
- **Hugging Face**: For the transformers library
- **Streamlit**: For the web interface framework
- **PyTorch**: For the deep learning framework

## Support

- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join community discussions on GitHub Discussions
- **Documentation**: Check the inline code documentation

## Future Enhancements

- [ ] Support for video analysis
- [ ] Real-time camera input
- [ ] Custom model fine-tuning
- [ ] API endpoint for external applications
- [ ] Mobile app integration
- [ ] Multi-language support


# CLIP-Image-Text-Matching
