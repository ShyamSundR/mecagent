"""
Inference script for CadQuery code generation.
Supports batch inference from images to code strings with various decoding strategies.
"""

import os
import argparse
import json
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import logging

from PIL import Image
import numpy as np

from .data import CadQueryDataset, ImageProcessor
from .models.baseline import create_baseline_model
from .models.enhanced import create_enhanced_model
from .utils import setup_logging, set_seed, load_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inference for CadQuery code generation")
    
    # Model configuration
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_type", type=str, choices=["baseline", "enhanced"], default="baseline",
                       help="Model type")
    
    # Input/Output
    parser.add_argument("--images", type=str, required=True, help="Path to images directory or single image")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    
    # Sampling parameters (for enhanced model)
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--best_of_n", type=int, default=6, help="Number of candidates for best-of-N")
    
    # Enhanced model features
    parser.add_argument("--use_grammar", action="store_true", help="Use grammar-aware decoding")
    parser.add_argument("--use_execution_guidance", action="store_true", help="Use execution guidance")
    parser.add_argument("--use_reranking", action="store_true", help="Use heuristic reranking")
    
    # Hardware
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_type: str, config: Dict[str, Any]) -> torch.nn.Module:
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


def load_images(images_path: str) -> List[Dict[str, Any]]:
    """Load images from directory or single file."""
    images_path = Path(images_path)
    
    if images_path.is_file():
        # Single image
        return [{"path": str(images_path), "name": images_path.stem}]
    elif images_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        images = []
        
        for img_path in images_path.iterdir():
            if img_path.suffix.lower() in image_extensions:
                images.append({
                    "path": str(img_path),
                    "name": img_path.stem
                })
        
        return sorted(images, key=lambda x: x["name"])
    else:
        raise ValueError(f"Images path does not exist: {images_path}")


def process_images(images: List[Dict[str, Any]], image_processor: ImageProcessor) -> torch.Tensor:
    """Process images to tensors."""
    processed_images = []
    
    for img_info in tqdm(images, desc="Processing images"):
        try:
            image = Image.open(img_info["path"]).convert('RGB')
            tensor = image_processor.process_image(image)
            processed_images.append(tensor)
        except Exception as e:
            logging.warning(f"Failed to process image {img_info['path']}: {e}")
            # Use a placeholder tensor
            placeholder = torch.zeros(3, 224, 224)
            processed_images.append(placeholder)
    
    return torch.stack(processed_images)


def generate_code_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    config: Dict[str, Any],
    args: argparse.Namespace
) -> List[str]:
    """Generate code for a batch of images."""
    model.eval()
    
    with torch.no_grad():
        if args.model_type == "enhanced" and (args.use_grammar or args.use_execution_guidance or args.use_reranking):
            # Use enhanced generation
            generated_codes = model.generate_enhanced(
                images=images,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                best_of_n=args.best_of_n
            )
        else:
            # Use standard generation
            generated_codes = model.generate_text(
                images=images,
                max_length=args.max_length,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                early_stopping=args.early_stopping
            )
    
    return generated_codes


def save_predictions(predictions: List[Dict[str, Any]], output_path: str):
    """Save predictions to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    logging.info(f"Predictions saved to {output_path}")


def main():
    """Main inference function."""
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
        if value is not None and key not in ['checkpoint', 'config', 'model_type', 'images', 'output']:
            config[key] = value
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = create_model(args.model_type, config)
    model.to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, model)
    
    # Print model info
    param_counts = model.count_parameters()
    logger.info(f"Total parameters: {param_counts['total']:,}")
    logger.info(f"Trainable parameters: {param_counts['trainable']:,}")
    
    # Load images
    logger.info(f"Loading images from {args.images}")
    images = load_images(args.images)
    logger.info(f"Found {len(images)} images")
    
    # Create image processor
    image_processor = ImageProcessor(
        model_name=config.get("vision_model", "vit_base_patch16_224"),
        image_size=config.get("image_size", 224)
    )
    
    # Process images
    logger.info("Processing images...")
    image_tensors = process_images(images, image_processor)
    
    # Generate code in batches
    logger.info("Generating code...")
    predictions = []
    
    batch_size = args.batch_size
    num_batches = (len(image_tensors) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(image_tensors), batch_size), desc="Generating code"):
        batch_images = image_tensors[i:i + batch_size]
        batch_image_info = images[i:i + batch_size]
        
        # Generate code for batch
        generated_codes = generate_code_batch(
            model=model,
            images=batch_images,
            config=config,
            args=args
        )
        
        # Create predictions
        for j, (img_info, code) in enumerate(zip(batch_image_info, generated_codes)):
            prediction = {
                "id": img_info["name"],
                "image_path": img_info["path"],
                "generated_code": code,
                "model_type": args.model_type,
                "generation_params": {
                    "max_length": args.max_length,
                    "num_beams": args.num_beams,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "use_grammar": args.use_grammar,
                    "use_execution_guidance": args.use_execution_guidance,
                    "use_reranking": args.use_reranking
                }
            }
            predictions.append(prediction)
    
    # Save predictions
    logger.info(f"Saving {len(predictions)} predictions...")
    save_predictions(predictions, args.output)
    
    # Print summary
    logger.info("Inference completed!")
    logger.info(f"Generated code for {len(predictions)} images")
    logger.info(f"Results saved to {args.output}")
    
    # Print sample predictions
    if predictions:
        logger.info("Sample predictions:")
        for i, pred in enumerate(predictions[:3]):
            logger.info(f"  {i+1}. {pred['id']}: {pred['generated_code'][:100]}...")


if __name__ == "__main__":
    main()
