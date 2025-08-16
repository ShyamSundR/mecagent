#!/usr/bin/env python3
"""
Standalone training script for CadQuery code generation baseline model.
This script can be run directly without module import issues.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time
from tqdm import tqdm
import json

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from data import CadQueryDataset, create_dataloader
from models.baseline import create_baseline_model
from utils import setup_logging, set_seed, save_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CadQuery code generation baseline model")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum samples to use")
    parser.add_argument("--output_dir", type=str, default="checkpoints/baseline", help="Output directory")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    return parser.parse_args()


def train_baseline_model(args):
    """Train the baseline model."""
    # Setup
    set_seed(args.seed)
    setup_logging()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    print("Creating dataset...")
    dataset = CadQueryDataset(
        dataset_name="CADCODER/GenCAD-Code",
        split="train",
        max_samples=args.max_samples,
        image_size=224,
        max_code_length=512
    )
    
    # Debug: Check dataset structure
    print("Checking dataset structure...")
    try:
        sample = dataset.get_sample(0)
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Image type: {type(sample['image'])}")
        print(f"Code type: {type(sample['code'])}")
        print(f"Code length: {len(sample['code']) if sample['code'] else 'None'}")
        print(f"First 100 chars of code: {sample['code'][:100] if sample['code'] else 'None'}")
    except Exception as e:
        print(f"Error checking dataset: {e}")
        # Try to get raw dataset info
        raw_sample = dataset.dataset[0]
        print(f"Raw dataset keys: {list(raw_sample.keys())}")
        return None
    
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create model
    print("Creating baseline model...")
    model = create_baseline_model(
        vision_model_name="vit_base_patch16_224",
        code_model_name="Salesforce/codet5-small",
        image_size=224,
        max_code_length=512,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        freeze_vision=True,
        unfreeze_last_n_blocks=2
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=args.learning_rate, 
        weight_decay=0.01
    )
    
    # Calculate total steps
    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(0.1 * total_steps)
    
    # Use a simple linear scheduler
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    
    # Mixed precision - handle different PyTorch versions
    scaler = None
    if args.mixed_precision and device.type == 'cuda':
        try:
            # Try new API first
            from torch.amp import GradScaler
            scaler = GradScaler('cuda')
        except (ImportError, TypeError):
            # Fall back to old API
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Total steps: {total_steps}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch['images'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Create labels (for language modeling, labels = input_ids)
                labels = input_ids.clone()
                
                # Forward pass
                if args.mixed_precision and scaler is not None:
                    # Handle different PyTorch versions for autocast
                    try:
                        from torch.amp import autocast
                        with autocast(device_type='cuda'):
                            outputs = model(
                                images=images,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                            loss = outputs.loss / args.gradient_accumulation_steps
                    except (ImportError, TypeError):
                        from torch.cuda.amp import autocast
                        with autocast():
                            outputs = model(
                                images=images,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                            loss = outputs.loss / args.gradient_accumulation_steps
                    
                    # Backward pass
                    scaler.scale(loss).backward()
                else:
                    outputs = model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / args.gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if args.max_grad_norm > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Optimizer step
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    if global_step < warmup_steps:
                        scheduler.step()
                    
                    optimizer.zero_grad()
                    global_step += 1
                
                epoch_loss += loss.item() * args.gradient_accumulation_steps
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    'avg_loss': f"{epoch_loss/(batch_idx+1):.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                    'step': global_step
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"Batch keys: {list(batch.keys())}")
                print(f"Images shape: {batch['images'].shape if 'images' in batch else 'N/A'}")
                print(f"Input IDs shape: {batch['input_ids'].shape if 'input_ids' in batch else 'N/A'}")
                continue
        
        # Save checkpoint after each epoch
        checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch+1}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_eval_loss=epoch_loss / len(dataloader),
                config=vars(args),
                path=checkpoint_path
            )
            
            print(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
        
        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss/len(dataloader):.4f}")
    
    print("Training completed!")
    return model


def test_model_generation(model, dataset, device):
    """Test model generation on a few samples."""
    print("\nTesting model generation...")
    model.eval()
    
    with torch.no_grad():
        for i in range(min(3, len(dataset))):
            try:
                sample = dataset[i]
                image = sample['image'].unsqueeze(0).to(device)
                
                # Generate code
                generated_ids = model.generate(
                    images=image,
                    max_length=128,
                    num_beams=2,
                    early_stopping=True
                )
                
                generated_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                original_code = sample['code']
                
                print(f"\nSample {i+1}:")
                print(f"Original code: {original_code[:100]}...")
                print(f"Generated code: {generated_text[:100]}...")
                
            except Exception as e:
                print(f"Error generating for sample {i}: {e}")


if __name__ == "__main__":
    args = parse_args()
    model = train_baseline_model(args)
    
    if model is not None:
        # Test generation
        print("\nLoading dataset for testing...")
        test_dataset = CadQueryDataset(
            dataset_name="CADCODER/GenCAD-Code",
            split="train",
            max_samples=10,
            image_size=224,
            max_code_length=512
        )
        test_model_generation(model, test_dataset, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("✅ Baseline model training completed successfully!")
    else:
        print("❌ Training failed!")