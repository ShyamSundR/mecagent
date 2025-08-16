#!/usr/bin/env python3
"""
Standalone training script for CadQuery code generation enhanced model.
This script can be run directly without module import issues.
"""

import os
import sys
import argparse
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
from models.enhanced import create_enhanced_model
from utils import setup_logging, set_seed, save_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CadQuery code generation enhanced model")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum samples to use")
    parser.add_argument("--output_dir", type=str, default="checkpoints/enhanced", help="Output directory")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def train_enhanced_model(args):
    """Train the enhanced model."""
    # Setup
    set_seed(args.seed)
    setup_logging()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    print("Creating enhanced dataset...")
    dataset = CadQueryDataset(
        dataset_name="CADCODER/GenCAD-Code",
        split="train",
        max_samples=args.max_samples,
        image_size=224,
        max_code_length=512
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    print(f"Enhanced dataset: {len(dataset)} samples")
    print(f"Dataloader batches: {len(dataloader)}")
    
    # Create enhanced model
    print("Creating enhanced model...")
    enhanced_model = create_enhanced_model(
        vision_model_name="vit_base_patch16_224",
        code_model_name="Salesforce/codet5-small",
        image_size=224,
        max_code_length=512,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        freeze_vision=True,
        unfreeze_last_n_blocks=2,
        use_grammar=True,
        use_execution_guidance=True,
        use_reranking=True,
        best_of_n=6
    )
    
    enhanced_model = enhanced_model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in enhanced_model.parameters())
    trainable_params = sum(p.numel() for p in enhanced_model.parameters() if p.requires_grad)
    print(f"Enhanced model parameters:")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")
    print(f"- Enhanced features: Grammar-aware + Execution-guided + Reranking")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(enhanced_model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision
    scaler = GradScaler() if args.mixed_precision else None
    
    # Training loop
    print(f"Starting enhanced training for {args.epochs} epochs...")
    enhanced_model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Enhanced Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            if args.mixed_precision:
                with autocast():
                    outputs = enhanced_model(images, input_ids, attention_mask, labels=input_ids)
                    loss = outputs.loss
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = enhanced_model(images, input_ids, attention_mask, labels=input_ids)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss/(batch_idx+1):.4f}"
            })
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 1 == 0:  # Save every epoch
            checkpoint_path = Path(args.output_dir) / f"enhanced_checkpoint_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_checkpoint(
                model=enhanced_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=(epoch + 1) * len(dataloader),
                best_eval_loss=epoch_loss / len(dataloader),
                config=vars(args),
                path=checkpoint_path
            )
            
            print(f"Enhanced checkpoint saved: {checkpoint_path}")
    
    print("Enhanced training completed!")
    return enhanced_model


if __name__ == "__main__":
    args = parse_args()
    model = train_enhanced_model(args)
    print("Enhanced model training completed successfully!")
