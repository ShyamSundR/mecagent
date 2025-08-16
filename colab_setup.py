#!/usr/bin/env python3
"""
Colab setup script for CadQuery code generator.
This script handles module imports and provides easy training commands.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Setup the environment for training."""
    print("🚀 Setting up CadQuery Code Generator for Colab...")
    
    # Add src to Python path
    if 'src' not in sys.path:
        sys.path.append('src')
    
    # Install dependencies if not already installed
    try:
        import torch
        print("✓ PyTorch already installed")
    except ImportError:
        print("Installing PyTorch...")
        subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio"], check=True)
    
    try:
        import transformers
        print("✓ Transformers already installed")
    except ImportError:
        print("Installing transformers...")
        subprocess.run(["pip", "install", "transformers", "datasets", "timm", "peft", "accelerate"], check=True)
    
    try:
        import cadquery
        print("✓ CadQuery already installed")
    except ImportError:
        print("Installing CadQuery...")
        subprocess.run(["pip", "install", "cadquery", "trimesh", "opencv-python", "Pillow"], check=True)
    
    try:
        import lark
        print("✓ Lark already installed")
    except ImportError:
        print("Installing utilities...")
        subprocess.run(["pip", "install", "lark", "pyyaml", "rich", "tyro", "tqdm", "matplotlib", "seaborn"], check=True)
    
    print("✅ Environment setup completed!")

def test_imports():
    """Test that all imports work correctly."""
    print("\n🧪 Testing imports...")
    
    try:
        # Test basic imports
        import torch
        import transformers
        print("✓ Core ML libraries imported")
        
        # Test our modules
        sys.path.append('src')
        from data import CadQueryDataset
        from models.baseline import create_baseline_model
        from utils import setup_logging, set_seed
        print("✓ Our modules imported successfully")
        
        # Test metrics
        from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
        from metrics.best_iou import get_iou_best
        print("✓ Metrics imported successfully")
        
        print("✅ All imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def get_training_commands():
    """Get the training commands for Colab."""
    print("\n📋 Training Commands for Colab:")
    print("=" * 50)
    
    print("\n1. Quick Training (Baseline Model):")
    print("python train_baseline.py --epochs 2 --max_samples 1000 --mixed_precision")
    
    print("\n2. Full Training (Baseline Model):")
    print("python train_baseline.py --epochs 5 --max_samples 5000 --mixed_precision")
    
    print("\n3. Enhanced Model Training:")
    print("python -c \"import sys; sys.path.append('src'); from train import main; main()\"")
    
    print("\n4. Test Dataset Loading:")
    print("python -c \"import sys; sys.path.append('src'); from data import CadQueryDataset; ds = CadQueryDataset(max_samples=10); print(f'Dataset loaded: {len(ds)} samples')\"")
    
    print("\n5. Test Model Creation:")
    print("python -c \"import sys; sys.path.append('src'); from models.baseline import create_baseline_model; model = create_baseline_model(); print('Model created successfully')\"")

if __name__ == "__main__":
    setup_environment()
    if test_imports():
        get_training_commands()
        print("\n🎉 Ready for training! Use the commands above to start training.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
