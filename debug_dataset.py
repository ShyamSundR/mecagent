#!/usr/bin/env python3
"""
Debug script to check the dataset structure.
"""

from datasets import load_dataset

# Load a small sample of the dataset
print("Loading dataset...")
ds = load_dataset("CADCODER/GenCAD-Code", split="train")

print(f"Dataset size: {len(ds)}")
print(f"Dataset features: {ds.features}")

# Check the first sample
print("\nFirst sample keys:")
sample = ds[0]
for key, value in sample.items():
    print(f"  {key}: {type(value)} - {str(value)[:100]}...")

# Check if 'code' key exists
if 'code' in sample:
    print("\n✓ 'code' key found!")
else:
    print("\n❌ 'code' key not found!")
    print("Available keys:", list(sample.keys()))
    
    # Look for code-related keys
    code_keys = [k for k in sample.keys() if 'code' in k.lower() or 'text' in k.lower() or 'cad' in k.lower()]
    print("Potential code keys:", code_keys)
