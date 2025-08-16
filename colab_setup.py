#!/usr/bin/env python3
"""
Colab Setup Script for CadQuery Code Generator
Run this in Google Colab to set up the complete environment and demo.
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install all required dependencies."""
    print("Installing dependencies...")
    
    packages = [
        "torch torchvision torchaudio",
        "transformers datasets timm peft accelerate",
        "cadquery trimesh opencv-python Pillow",
        "lark pyyaml rich tyro tqdm matplotlib seaborn",
        "numpy scipy scikit-learn"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run(f"pip install {package}", shell=True, check=True)
    
    print("✓ Dependencies installed successfully!")

def check_gpu():
    """Check GPU availability and info."""
    print("\nChecking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"✓ CUDA version: {torch.version.cuda}")
        else:
            print("⚠ No GPU available - will use CPU (slower)")
    except Exception as e:
        print(f"✗ GPU check failed: {e}")

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    directories = ["src", "configs", "grammars", "results", "checkpoints"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created {dir_name}/")

def setup_environment():
    """Set up the complete environment."""
    print("🚀 Setting up CadQuery Code Generator for Colab...")
    print("=" * 60)
    
    # Install dependencies
    install_dependencies()
    
    # Check GPU
    check_gpu()
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 60)
    print("🎉 Environment setup complete!")
    print("\nNext steps:")
    print("1. Upload the src/ folder and other files")
    print("2. Run the demo cells below")
    print("3. Train models with the provided commands")

if __name__ == "__main__":
    setup_environment()
