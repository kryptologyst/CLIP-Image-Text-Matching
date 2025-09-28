#!/usr/bin/env python3
"""
Setup script for CLIP Image-Text Matching project
This script helps users get started quickly with the project
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available!")
        return False

def install_requirements():
    """Install required packages"""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return False
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    return run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements")

def create_directories():
    """Create necessary directories"""
    directories = ["sample_images", ".streamlit"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")

def test_installation():
    """Test if the installation works"""
    print("🧪 Testing installation...")
    
    try:
        # Test basic imports
        import torch
        import clip
        from PIL import Image
        import streamlit
        print("✅ All required packages imported successfully!")
        
        # Test CLIP model loading
        print("🔄 Testing CLIP model loading...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print(f"✅ CLIP model loaded successfully on {device}!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing installation: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 CLIP Image-Text Matching Setup")
    print("=" * 50)
    
    # Check system requirements
    print("\n📋 Checking system requirements...")
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        print("❌ Failed to install requirements!")
        sys.exit(1)
    
    # Test installation
    print("\n🧪 Testing installation...")
    if not test_installation():
        print("❌ Installation test failed!")
        sys.exit(1)
    
    # Success message
    print("\n🎉 Setup completed successfully!")
    print("\n📖 Next steps:")
    print("   1. Run the demo: python 135.py")
    print("   2. Launch web interface: streamlit run app.py")
    print("   3. Explore the code in clip_matcher.py")
    print("\n💡 Tips:")
    print("   • The first run will download sample images")
    print("   • GPU acceleration is automatically detected")
    print("   • Check README.md for detailed usage instructions")
    
    # Platform-specific notes
    system = platform.system()
    if system == "Darwin":  # macOS
        print("\n🍎 macOS Notes:")
        print("   • If you encounter issues with PyTorch, try: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    elif system == "Windows":
        print("\n🪟 Windows Notes:")
        print("   • Make sure you have Visual Studio Build Tools installed")
        print("   • Consider using WSL2 for better performance")
    elif system == "Linux":
        print("\n🐧 Linux Notes:")
        print("   • For CUDA support, install NVIDIA drivers and CUDA toolkit")
        print("   • Consider using a virtual environment: python -m venv venv")

if __name__ == "__main__":
    main()
