"""
Evaluation script for CadQuery code generation.
Computes VSR and Best IOU using provided metrics on predictions vs references.
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import logging

from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best
from utils import setup_logging, set_seed, save_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate CadQuery code generation")
    
    # Input files
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSONL file")
    parser.add_argument("--references", type=str, required=True, help="Path to references JSONL file")
    
    # Output
    parser.add_argument("--output", type=str, default="results/evaluation_results.json", help="Output results file")
    parser.add_argument("--csv_output", type=str, default="results/evaluation_results.csv", help="Output CSV file")
    
    # Evaluation options
    parser.add_argument("--compute_vsr", action="store_true", default=True, help="Compute Valid Syntax Rate")
    parser.add_argument("--compute_iou", action="store_true", default=True, help="Compute Best IOU")
    parser.add_argument("--iou_timeout", type=float, default=10.0, help="Timeout for IOU computation")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to evaluate")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def match_predictions_references(predictions: List[Dict], references: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Match predictions with references by ID."""
    # Create lookup dictionaries
    pred_lookup = {pred["id"]: pred for pred in predictions}
    ref_lookup = {ref["id"]: ref for ref in references}
    
    # Find common IDs
    common_ids = set(pred_lookup.keys()) & set(ref_lookup.keys())
    
    # Create matched pairs
    matched_pairs = []
    for id_val in sorted(common_ids):
        matched_pairs.append((pred_lookup[id_val], ref_lookup[id_val]))
    
    return matched_pairs


def compute_vsr(predictions: List[Dict]) -> Dict[str, Any]:
    """Compute Valid Syntax Rate for predictions."""
    logger = logging.getLogger(__name__)
    logger.info("Computing Valid Syntax Rate...")
    
    # Extract codes
    codes = {pred["id"]: pred["generated_code"] for pred in predictions}
    
    # Compute VSR
    vsr = evaluate_syntax_rate_simple(codes)
    
    # Count valid/invalid
    valid_count = int(vsr * len(codes))
    invalid_count = len(codes) - valid_count
    
    results = {
        "vsr": vsr,
        "total_samples": len(codes),
        "valid_samples": valid_count,
        "invalid_samples": invalid_count,
        "valid_codes": [id_val for id_val, code in codes.items() if evaluate_syntax_rate_simple({id_val: code}) > 0],
        "invalid_codes": [id_val for id_val, code in codes.items() if evaluate_syntax_rate_simple({id_val: code}) == 0]
    }
    
    logger.info(f"VSR: {vsr:.3f} ({valid_count}/{len(codes)} valid)")
    return results


def compute_iou_batch(matched_pairs: List[Tuple[Dict, Dict]], timeout: float = 10.0) -> Dict[str, Any]:
    """Compute Best IOU for matched prediction-reference pairs."""
    logger = logging.getLogger(__name__)
    logger.info("Computing Best IOU...")
    
    iou_scores = []
    successful_pairs = []
    failed_pairs = []
    
    for pred, ref in tqdm(matched_pairs, desc="Computing IOU"):
        try:
            pred_code = pred["generated_code"]
            ref_code = ref["code"]  # Assuming reference has "code" field
            
            # Compute IOU
            iou = get_iou_best(pred_code, ref_code)
            iou_scores.append(iou)
            successful_pairs.append({
                "id": pred["id"],
                "pred_code": pred_code,
                "ref_code": ref_code,
                "iou": iou
            })
            
        except Exception as e:
            logger.warning(f"Failed to compute IOU for {pred['id']}: {e}")
            failed_pairs.append({
                "id": pred["id"],
                "error": str(e)
            })
    
    if not iou_scores:
        logger.warning("No successful IOU computations!")
        return {
            "mean_iou": 0.0,
            "std_iou": 0.0,
            "min_iou": 0.0,
            "max_iou": 0.0,
            "total_samples": len(matched_pairs),
            "successful_samples": 0,
            "failed_samples": len(matched_pairs),
            "iou_scores": [],
            "successful_pairs": [],
            "failed_pairs": failed_pairs
        }
    
    results = {
        "mean_iou": np.mean(iou_scores),
        "std_iou": np.std(iou_scores),
        "min_iou": np.min(iou_scores),
        "max_iou": np.max(iou_scores),
        "total_samples": len(matched_pairs),
        "successful_samples": len(iou_scores),
        "failed_samples": len(failed_pairs),
        "iou_scores": iou_scores,
        "successful_pairs": successful_pairs,
        "failed_pairs": failed_pairs
    }
    
    logger.info(f"Mean IOU: {results['mean_iou']:.3f} ± {results['std_iou']:.3f}")
    logger.info(f"IOU range: [{results['min_iou']:.3f}, {results['max_iou']:.3f}]")
    logger.info(f"Success rate: {results['successful_samples']}/{results['total_samples']}")
    
    return results


def analyze_errors(predictions: List[Dict], references: List[Dict]) -> Dict[str, Any]:
    """Analyze common error patterns."""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing error patterns...")
    
    # Extract codes
    pred_codes = {pred["id"]: pred["generated_code"] for pred in predictions}
    ref_codes = {ref["id"]: ref["code"] for ref in references}
    
    # Find common IDs
    common_ids = set(pred_codes.keys()) & set(ref_codes.keys())
    
    error_analysis = {
        "total_pairs": len(common_ids),
        "syntax_errors": 0,
        "semantic_errors": 0,
        "missing_operations": 0,
        "extra_operations": 0,
        "parameter_errors": 0,
        "common_error_patterns": {}
    }
    
    for id_val in common_ids:
        pred_code = pred_codes[id_val]
        ref_code = ref_codes[id_val]
        
        # Check syntax
        try:
            exec(pred_code, {"cadquery": None, "cq": None})
            syntax_valid = True
        except:
            syntax_valid = False
            error_analysis["syntax_errors"] += 1
        
        # Check for missing/extra operations
        pred_ops = set()
        ref_ops = set()
        
        # Extract operations (simplified)
        for line in pred_code.split('\n'):
            if '.box(' in line:
                pred_ops.add('box')
            elif '.cylinder(' in line:
                pred_ops.add('cylinder')
            elif '.hole(' in line:
                pred_ops.add('hole')
            # Add more operations as needed
        
        for line in ref_code.split('\n'):
            if '.box(' in line:
                ref_ops.add('box')
            elif '.cylinder(' in line:
                ref_ops.add('cylinder')
            elif '.hole(' in line:
                ref_ops.add('hole')
            # Add more operations as needed
        
        missing = ref_ops - pred_ops
        extra = pred_ops - ref_ops
        
        if missing:
            error_analysis["missing_operations"] += 1
        if extra:
            error_analysis["extra_operations"] += 1
    
    logger.info(f"Error analysis: {error_analysis}")
    return error_analysis


def create_results_table(vsr_results: Dict, iou_results: Dict, error_analysis: Dict) -> pd.DataFrame:
    """Create a results table for easy viewing."""
    data = {
        "Metric": [
            "Valid Syntax Rate (VSR)",
            "Mean Best IOU",
            "IOU Std Dev",
            "IOU Min",
            "IOU Max",
            "Total Samples",
            "Successful IOU Computations",
            "Failed IOU Computations",
            "Syntax Errors",
            "Missing Operations",
            "Extra Operations"
        ],
        "Value": [
            f"{vsr_results['vsr']:.3f}",
            f"{iou_results['mean_iou']:.3f}",
            f"{iou_results['std_iou']:.3f}",
            f"{iou_results['min_iou']:.3f}",
            f"{iou_results['max_iou']:.3f}",
            iou_results['total_samples'],
            iou_results['successful_samples'],
            iou_results['failed_samples'],
            error_analysis['syntax_errors'],
            error_analysis['missing_operations'],
            error_analysis['extra_operations']
        ]
    }
    
    return pd.DataFrame(data)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info(f"Loading predictions from {args.predictions}")
    predictions = load_jsonl(args.predictions)
    
    logger.info(f"Loading references from {args.references}")
    references = load_jsonl(args.references)
    
    logger.info(f"Loaded {len(predictions)} predictions and {len(references)} references")
    
    # Match predictions with references
    matched_pairs = match_predictions_references(predictions, references)
    logger.info(f"Matched {len(matched_pairs)} prediction-reference pairs")
    
    # Limit samples if specified
    if args.max_samples and len(matched_pairs) > args.max_samples:
        logger.info(f"Limiting evaluation to {args.max_samples} samples")
        matched_pairs = matched_pairs[:args.max_samples]
    
    # Extract predictions and references from matched pairs
    eval_predictions = [pred for pred, ref in matched_pairs]
    eval_references = [ref for pred, ref in matched_pairs]
    
    # Initialize results
    results = {
        "evaluation_config": vars(args),
        "dataset_info": {
            "total_predictions": len(predictions),
            "total_references": len(references),
            "matched_pairs": len(matched_pairs),
            "evaluated_samples": len(eval_predictions)
        }
    }
    
    # Compute VSR
    if args.compute_vsr:
        vsr_results = compute_vsr(eval_predictions)
        results["vsr_results"] = vsr_results
    
    # Compute IOU
    if args.compute_iou:
        iou_results = compute_iou_batch(matched_pairs, timeout=args.iou_timeout)
        results["iou_results"] = iou_results
    
    # Analyze errors
    error_analysis = analyze_errors(eval_predictions, eval_references)
    results["error_analysis"] = error_analysis
    
    # Create summary
    summary = {
        "vsr": vsr_results["vsr"] if args.compute_vsr else None,
        "mean_iou": iou_results["mean_iou"] if args.compute_iou else None,
        "iou_std": iou_results["std_iou"] if args.compute_iou else None,
        "total_evaluated": len(eval_predictions)
    }
    results["summary"] = summary
    
    # Save results
    logger.info(f"Saving results to {args.output}")
    save_results(results, args.output)
    
    # Create and save CSV table
    if args.compute_vsr and args.compute_iou:
        df = create_results_table(vsr_results, iou_results, error_analysis)
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Results table saved to {args.csv_output}")
        
        # Print table
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)
    
    # Print summary
    logger.info("Evaluation completed!")
    logger.info(f"VSR: {summary['vsr']:.3f}" if summary['vsr'] is not None else "VSR: Not computed")
    logger.info(f"Mean IOU: {summary['mean_iou']:.3f}" if summary['mean_iou'] is not None else "Mean IOU: Not computed")
    logger.info(f"Total evaluated: {summary['total_evaluated']}")


if __name__ == "__main__":
    main()
