#!/usr/bin/env python3
"""
Simple test script to verify the CadQuery code generator setup.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    # Test standard library imports
    try:
        import json
        import torch
        import numpy as np
        print("✓ Standard libraries imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import standard libraries: {e}")
        return False
    
    # Test if src directory exists
    src_path = Path("src")
    if not src_path.exists():
        print("✗ src directory not found")
        return False
    print("✓ src directory found")
    
    # Test if key files exist
    required_files = [
        "src/data.py",
        "src/models/baseline.py", 
        "src/models/enhanced.py",
        "src/train.py",
        "src/infer.py",
        "src/eval.py",
        "src/utils.py",
        "configs/baseline.yaml",
        "configs/enhanced.yaml",
        "grammars/cadquery_ebnf.lark",
        "metrics/valid_syntax_rate.py",
        "metrics/best_iou.py"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"✗ Required file not found: {file_path}")
            return False
    
    print("✓ All required files found")
    return True

def test_metrics():
    """Test the provided metrics."""
    print("\nTesting metrics...")
    
    try:
        from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
        from metrics.best_iou import get_iou_best
        
        # Test VSR
        test_code = """
        import cadquery as cq
        result = cq.Workplane("XY").box(10, 10, 10)
        """
        vsr = evaluate_syntax_rate_simple({"test": test_code})
        print(f"✓ VSR test: {vsr}")
        
        # Test IOU
        code1 = """
        import cadquery as cq
        result = cq.Workplane("XY").box(10, 10, 10)
        """
        code2 = """
        import cadquery as cq
        result = cq.Workplane("XY").cylinder(5, 10)
        """
        iou = get_iou_best(code1, code2)
        print(f"✓ IOU test: {iou:.3f}")
        
        return True
        
    except ImportError as e:
        if "cadquery" in str(e):
            print("⚠ CadQuery not installed (expected for setup test)")
            print("  This is normal - install dependencies with 'uv sync' to run full tests")
            return True  # Don't fail the test for missing cadquery
        else:
            print(f"✗ Metrics test failed: {e}")
            return False
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False

def test_project_structure():
    """Test project structure."""
    print("\nTesting project structure...")
    
    # Check directories
    directories = ["src", "configs", "grammars", "results", "checkpoints"]
    for dir_name in directories:
        if Path(dir_name).exists():
            print(f"✓ Directory exists: {dir_name}")
        else:
            print(f"⚠ Directory missing: {dir_name} (will be created during setup)")
    
    # Check Makefile
    if Path("Makefile").exists():
        print("✓ Makefile found")
    else:
        print("✗ Makefile not found")
        return False
    
    return True

def main():
    """Run all tests."""
    print("CadQuery Code Generator - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_metrics,
        test_project_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: uv sync")
        print("2. Activate environment: source .venv/bin/activate")
        print("3. Run demo: make demo")
        print("4. Train models: make train-baseline")
    else:
        print("⚠ Some tests failed. Please check the setup.")
        print("Note: Missing cadquery is expected - install dependencies first")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
