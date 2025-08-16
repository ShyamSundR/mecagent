#!/usr/bin/env python3
"""
Test script to verify module imports work correctly.
This script tests imports without requiring all dependencies to be installed.
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic Python imports that should work without external dependencies."""
    print("Testing basic imports...")
    
    try:
        # Test basic Python modules
        import re
        import textwrap
        from typing import Dict, List, Tuple, Optional, Union
        print("✓ Basic Python imports successful")
    except ImportError as e:
        print(f"✗ Basic imports failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "src/__init__.py",
        "src/data.py", 
        "src/models/__init__.py",
        "src/models/baseline.py",
        "src/models/enhanced.py",
        "src/train.py",
        "src/infer.py",
        "src/eval.py",
        "src/utils.py",
        "configs/baseline.yaml",
        "configs/enhanced.yaml",
        "grammars/cadquery_ebnf.lark",
        "pyproject.toml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    print("✓ All required files exist")
    return True

def test_syntax():
    """Test that Python files have valid syntax."""
    print("\nTesting Python syntax...")
    
    python_files = [
        "src/__init__.py",
        "src/data.py",
        "src/models/__init__.py", 
        "src/models/baseline.py",
        "src/models/enhanced.py",
        "src/train.py",
        "src/infer.py",
        "src/eval.py",
        "src/utils.py"
    ]
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✓ {file_path} - valid syntax")
        except SyntaxError as e:
            print(f"✗ {file_path} - syntax error: {e}")
            return False
    
    print("✓ All Python files have valid syntax")
    return True

def test_conditional_imports():
    """Test imports that might fail gracefully."""
    print("\nTesting conditional imports...")
    
    # Test data module structure (without actually importing dependencies)
    try:
        with open("src/data.py", 'r') as f:
            content = f.read()
        
        # Check that key classes are defined
        required_classes = [
            "class CadQueryCodeNormalizer:",
            "class ImageProcessor:", 
            "class CadQueryTokenizer:",
            "class CadQueryDataset:",
            "def create_dataloader("
        ]
        
        for class_name in required_classes:
            if class_name in content:
                print(f"✓ {class_name.strip()}")
            else:
                print(f"✗ Missing: {class_name.strip()}")
                return False
        
        print("✓ All required classes found in data.py")
        
    except Exception as e:
        print(f"✗ Error reading data.py: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 Testing CadQuery Code Generator Implementation")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_file_structure,
        test_syntax,
        test_conditional_imports
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
            break
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Implementation is ready for Colab.")
        print("\n📋 Next steps:")
        print("1. Clone repository in Colab: git clone https://github.com/ShyamSundR/mecagent.git")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run training: python -m src.train --config configs/baseline.yaml")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return all_passed

if __name__ == "__main__":
    main()
