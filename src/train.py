"""
Training script for CadQuery code generation models.
Supports baseline and enhanced models with LoRA, mixed precision, and gradient accumulation.
"""

import os
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

from .data import CadQueryDataset, create_dataloader
from .models.baseline import create_baseline_model
from .models.enhanced import create_enhanced_model
from .utils import setup_logging, set_seed, save_checkpoint, load_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CadQuery code generation model")
    
    # Model configuration
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_type", type=str, choices=["baseline", "enhanced"], default="baseline",
                       help="Model type to train")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # Data configuration
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to use")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--max_code_length", type=int, default=512, help="Maximum code length")
    
    # Model configuration
    parser.add_argument("--vision_model", type=str, default="vit_base_patch16_224", help="Vision model name")
    parser.add_argument("--code_model", type=str, default="Salesforce/codet5-small", help="Code model name")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    # Enhanced model features
    parser.add_argument("--use_grammar", action="store_true", help="Use grammar-aware decoding")
    parser.add_argument("--use_execution_guidance", action="store_true", help="Use execution guidance")
    parser.add_argument("--use_reranking", action="store_true", help="Use heuristic reranking")
    parser.add_argument("--best_of_n", type=int, default=6, help="Number of candidates for best-of-N")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    
    # Hardware configuration
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
    """Create model based on type and configuration."""
    if model_type == "baseline":
        return create_baseline_model(
            vision_model_name=config.get("vision_model", "vit_base_patch16_224"),
            code_model_name=config.get("code_model", "Salesforce/codet5-small"),
            image_size=config.get("image_size", 224),
            max_code_length=config.get("max_code_length", 512),
            use_lora=config.get("use_lora", True),
            lora_rank=config.get("lora_rank", 16),
            lora_alpha=config.get("lora_alpha", 32),
            freeze_vision=config.get("freeze_vision", True),
            unfreeze_last_n_blocks=config.get("unfreeze_last_n_blocks", 2)
        )
    elif model_type == "enhanced":
        return create_enhanced_model(
            vision_model_name=config.get("vision_model", "vit_base_patch16_224"),
            code_model_name=config.get("code_model", "Salesforce/codet5-small"),
            image_size=config.get("image_size", 224),
            max_code_length=config.get("max_code_length", 512),
            use_lora=config.get("use_lora", True),
            lora_rank=config.get("lora_rank", 16),
            lora_alpha=config.get("lora_alpha", 32),
            freeze_vision=config.get("freeze_vision", True),
            unfreeze_last_n_blocks=config.get("unfreeze_last_n_blocks", 2),
            use_grammar=config.get("use_grammar", True),
            use_execution_guidance=config.get("use_execution_guidance", True),
            use_reranking=config.get("use_reranking", True),
            best_of_n=config.get("best_of_n", 6),
            execution_timeout=config.get("execution_timeout", 5.0)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer for the model."""
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Create optimizer
    optimizer = optim.AdamW(
        trainable_params,
        lr=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], total_steps: int) -> Any:
    """Create learning rate scheduler."""
    from transformers import get_linear_schedule_with_warmup
    
    warmup_steps = config.get("warmup_steps", int(0.1 * total_steps))
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    return scheduler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    device: torch.device,
    config: Dict[str, Any],
    epoch: int,
    global_step: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()
        
        # Forward pass with mixed precision
        with autocast(enabled=config.get("mixed_precision", False)):
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
        
        # Scale loss for gradient accumulation
        loss = loss / config.get("gradient_accumulation_steps", 1)
        
        # Backward pass
        if config.get("mixed_precision", False):
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.get("gradient_accumulation_steps", 1) == 0:
            # Gradient clipping
            if config.get("max_grad_norm", 1.0) > 0:
                if config.get("mixed_precision", False):
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.get("max_grad_norm", 1.0)
                )
            
            # Optimizer step
            if config.get("mixed_precision", False):
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        # Update progress bar
        total_loss += loss.item() * config.get("gradient_accumulation_steps", 1)
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            'step': global_step
        })
    
    return {
        'loss': total_loss / num_batches,
        'global_step': global_step
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            
            # Forward pass
            with autocast(enabled=config.get("mixed_precision", False)):
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            total_loss += loss.item()
    
    return {
        'eval_loss': total_loss / num_batches
    }


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    for key, value in vars(args).items():
        if value is not None and key not in ['config', 'model_type']:
            config[key] = value
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = CadQueryDataset(
        split="train",
        max_samples=config.get("max_samples"),
        image_size=config.get("image_size", 224),
        max_code_length=config.get("max_code_length", 512),
        tokenizer_name=config.get("code_model", "Salesforce/codet5-small")
    )
    
    eval_dataset = CadQueryDataset(
        split="test",
        max_samples=min(1000, len(train_dataset) // 10),  # Use subset for evaluation
        image_size=config.get("image_size", 224),
        max_code_length=config.get("max_code_length", 512),
        tokenizer_name=config.get("code_model", "Salesforce/codet5-small")
    )
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=True,
        num_workers=config.get("num_workers", 4)
    )
    
    eval_dataloader = create_dataloader(
        eval_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False,
        num_workers=config.get("num_workers", 4)
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = create_model(args.model_type, config)
    model.to(device)
    
    # Print model info
    param_counts = model.count_parameters()
    logger.info(f"Total parameters: {param_counts['total']:,}")
    logger.info(f"Trainable parameters: {param_counts['trainable']:,}")
    logger.info(f"Frozen parameters: {param_counts['frozen']:,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    total_steps = len(train_dataloader) * config.get("epochs", 10) // config.get("gradient_accumulation_steps", 1)
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler() if config.get("mixed_precision", False) else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_eval_loss = float('inf')
    
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = load_checkpoint(args.resume_from, model, optimizer, scheduler)
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        best_eval_loss = checkpoint.get("best_eval_loss", float('inf'))
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config.get("epochs", 10)):
        # Train epoch
        train_metrics = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            config=config,
            epoch=epoch,
            global_step=global_step
        )
        
        global_step = train_metrics['global_step']
        
        # Evaluate
        if global_step % config.get("eval_steps", 500) == 0:
            eval_metrics = evaluate_model(
                model=model,
                dataloader=eval_dataloader,
                device=device,
                config=config
            )
            
            logger.info(f"Epoch {epoch}, Step {global_step}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
            
            # Save best model
            if eval_metrics['eval_loss'] < best_eval_loss:
                best_eval_loss = eval_metrics['eval_loss']
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_eval_loss=best_eval_loss,
                    config=config,
                    path=output_dir / "best_model.pt"
                )
                logger.info(f"New best model saved with eval loss: {best_eval_loss:.4f}")
        
        # Save checkpoint
        if global_step % config.get("save_steps", 1000) == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_eval_loss=best_eval_loss,
                config=config,
                path=output_dir / f"checkpoint_step_{global_step}.pt"
            )
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=config.get("epochs", 10) - 1,
        global_step=global_step,
        best_eval_loss=best_eval_loss,
        config=config,
        path=output_dir / "final_model.pt"
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
