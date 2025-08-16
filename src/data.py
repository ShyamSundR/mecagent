"""
Data processing module for CadQuery code generation.
Handles dataset loading, image transforms, code normalization, and tokenization.
"""

import os
import re
import textwrap
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
# timm imports removed - using torchvision transforms instead


class CadQueryCodeNormalizer:
    """Normalizes and canonicalizes CadQuery code for consistent training."""
    
    def __init__(self):
        self.common_vars = {
            'height': 'height', 'width': 'width', 'thickness': 'thickness',
            'diameter': 'diameter', 'radius': 'radius', 'length': 'length',
            'depth': 'depth', 'padding': 'padding', 'margin': 'margin'
        }
    
    def normalize_code(self, code: str) -> str:
        """Normalize CadQuery code for consistent training."""
        # Remove comments and extra whitespace
        code = self._remove_comments(code)
        code = self._normalize_whitespace(code)
        
        # Ensure result variable is present
        code = self._ensure_result_variable(code)
        
        # Normalize common variable names
        code = self._normalize_variable_names(code)
        
        return code.strip()
    
    def _remove_comments(self, code: str) -> str:
        """Remove Python comments from code."""
        lines = []
        for line in code.split('\n'):
            # Remove inline comments
            if '#' in line:
                line = line[:line.index('#')]
            if line.strip():
                lines.append(line)
        return '\n'.join(lines)
    
    def _normalize_whitespace(self, code: str) -> str:
        """Normalize whitespace and indentation."""
        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code)
        code = re.sub(r'\s*([=+\-*/()\[\]{}.,:])\s*', r'\1', code)
        
        # Fix indentation
        code = textwrap.dedent(code)
        
        return code
    
    def _ensure_result_variable(self, code: str) -> str:
        """Ensure the code has a result variable assignment."""
        # Check if result is already assigned
        if 'result =' in code or 'result=' in code:
            return code
        
        # Look for CadQuery operations that should be assigned to result
        cq_patterns = [
            r'cq\.Workplane\("XY"\)',
            r'\.box\(',
            r'\.cylinder\(',
            r'\.sphere\(',
        ]
        
        for pattern in cq_patterns:
            if re.search(pattern, code):
                # Find the last CadQuery operation and assign it to result
                lines = code.split('\n')
                for i, line in enumerate(lines):
                    if any(re.search(p, line) for p in cq_patterns):
                        # Check if this line is already assigned
                        if '=' not in line or line.strip().startswith('='):
                            lines[i] = f"result = {line.strip()}"
                            break
                return '\n'.join(lines)
        
        return code
    
    def _normalize_variable_names(self, code: str) -> str:
        """Normalize common variable names for consistency."""
        # This is a simple approach - in practice, you might want more sophisticated
        # variable name normalization based on context
        return code


class ImageProcessor:
    """Handles image preprocessing for the vision encoder."""
    
    def __init__(self, model_name: str = "vit_base_patch16_224", image_size: int = 224):
        self.model_name = model_name
        self.image_size = image_size
        
        # Use simple transforms instead of timm's complex config
        from torchvision import transforms
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def process_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """Process image to tensor format for the model."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms
        tensor = self.transform(image)
        return tensor


class CadQueryTokenizer:
    """Handles tokenization for CadQuery code."""
    
    def __init__(self, model_name: str = "Salesforce/codet5-small", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Add special tokens if needed
        special_tokens = {
            'additional_special_tokens': [
                '<cadquery>', '</cadquery>',
                '<workplane>', '</workplane>',
                '<operation>', '</operation>'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
    
    def tokenize_code(self, code: str, padding: bool = True, truncation: bool = True) -> Dict[str, torch.Tensor]:
        """Tokenize CadQuery code."""
        return self.tokenizer(
            code,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def decode_tokens(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to code."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class CadQueryDataset:
    """Dataset wrapper for CadQuery image-code pairs."""
    
    def __init__(
        self,
        dataset_name: str = "CADCODER/GenCAD-Code",
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: int = 224,
        max_code_length: int = 512,
        tokenizer_name: str = "Salesforce/codet5-small"
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        
        # Initialize processors
        self.image_processor = ImageProcessor(image_size=image_size)
        self.code_normalizer = CadQueryCodeNormalizer()
        self.tokenizer = CadQueryTokenizer(tokenizer_name, max_code_length)
        
        # Load dataset
        self.dataset = self._load_dataset()
        
    def _load_dataset(self) -> Dataset:
        """Load the HuggingFace dataset."""
        dataset = load_dataset(self.dataset_name, split=self.split)
        
        if self.max_samples:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))
        
        return dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.dataset[idx]
        
        # Process image - try different possible keys
        image = None
        for key in ['image', 'img', 'picture']:
            if key in sample:
                image = sample[key]
                break
        
        if image is None:
            raise KeyError(f"No image found in sample. Available keys: {list(sample.keys())}")
        
        image_tensor = self.image_processor.process_image(image)
        
        # Process code - try different possible keys
        code = None
        for key in ['code', 'text', 'cadquery_code', 'script']:
            if key in sample:
                code = sample[key]
                break
        
        if code is None:
            raise KeyError(f"No code found in sample. Available keys: {list(sample.keys())}")
        
        normalized_code = self.code_normalizer.normalize_code(code)
        code_tokens = self.tokenizer.tokenize_code(normalized_code)
        
        return {
            'image': image_tensor,
            'input_ids': code_tokens['input_ids'].squeeze(0),
            'attention_mask': code_tokens['attention_mask'].squeeze(0),
            'code': normalized_code,
            'original_code': code
        }
    
    def get_sample(self, idx: int) -> Dict:
        """Get a sample without tensor conversion for inspection."""
        sample = self.dataset[idx]
        
        # Find image and code keys
        image = None
        for key in ['image', 'img', 'picture']:
            if key in sample:
                image = sample[key]
                break
        
        code = None
        for key in ['code', 'text', 'cadquery_code', 'script']:
            if key in sample:
                code = sample[key]
                break
        
        return {
            'image': image,
            'code': code,
            'normalized_code': self.code_normalizer.normalize_code(code) if code else None
        }


def create_dataloader(
    dataset: CadQueryDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the dataset."""
    
    def collate_fn(batch):
        """Custom collate function to handle variable length sequences."""
        images = torch.stack([item['image'] for item in batch])
        
        # Pad sequences to max length in batch
        max_length = max(len(item['input_ids']) for item in batch)
        
        input_ids = []
        attention_masks = []
        
        for item in batch:
            seq_len = len(item['input_ids'])
            padding_len = max_length - seq_len
            
            input_ids.append(torch.cat([
                item['input_ids'],
                torch.zeros(padding_len, dtype=torch.long)
            ]))
            
            attention_masks.append(torch.cat([
                item['attention_mask'],
                torch.zeros(padding_len, dtype=torch.long)
            ]))
        
        return {
            'images': images,
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'codes': [item['code'] for item in batch],
            'original_codes': [item['original_code'] for item in batch]
        }
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test the data processing pipeline
    print("Testing data processing pipeline...")
    
    # Create a small test dataset
    dataset = CadQueryDataset(max_samples=10)
    print(f"Dataset size: {len(dataset)}")
    
    # Test a sample
    sample = dataset.get_sample(0)
    print(f"Original code length: {len(sample['code'])}")
    print(f"Normalized code length: {len(sample['normalized_code'])}")
    
    # Test tensor conversion
    tensor_sample = dataset[0]
    print(f"Image tensor shape: {tensor_sample['image'].shape}")
    print(f"Input IDs shape: {tensor_sample['input_ids'].shape}")
    
    print("Data processing pipeline test completed!")
