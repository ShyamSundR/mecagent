"""
Utility functions for the CadQuery code generation project.
Includes logging, seeding, checkpoint I/O, and safe execution utilities.
"""

import os
import sys
import logging
import random
import numpy as np
import torch
import json
import time
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from contextlib import contextmanager
import traceback


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    best_eval_loss: float,
    config: Dict[str, Any],
    path: Union[str, Path]
) -> None:
    """Save model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'best_eval_loss': best_eval_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': config,
    }
    
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Dict[str, Any]:
    """Load model checkpoint."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logging.info(f"Checkpoint loaded from {path}")
    logging.info(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    logging.info(f"Global step: {checkpoint.get('global_step', 'N/A')}")
    logging.info(f"Best eval loss: {checkpoint.get('best_eval_loss', 'N/A')}")
    
    return checkpoint


class TimeoutError(Exception):
    """Custom timeout error."""
    pass


@contextmanager
def timeout(seconds: float):
    """Context manager for timeout functionality."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def safe_execute(
    code: str,
    timeout_seconds: float = 5.0,
    globals_dict: Optional[Dict[str, Any]] = None,
    locals_dict: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Any, Optional[str]]:
    """Safely execute Python code with timeout and error handling."""
    if globals_dict is None:
        globals_dict = {}
    
    if locals_dict is None:
        locals_dict = {}
    
    # Create a safe execution environment
    safe_globals = {
        '__builtins__': {
            'print': print,
            'range': range,
            'len': len,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'bool': bool,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'round': round,
            'pow': pow,
            'divmod': divmod,
        }
    }
    
    # Add CadQuery imports
    try:
        import cadquery as cq
        import numpy as np
        safe_globals.update({
            'cq': cq,
            'cadquery': cq,
            'np': np,
            'numpy': np
        })
    except ImportError:
        pass
    
    # Update with provided globals
    safe_globals.update(globals_dict)
    
    try:
        # Execute with timeout
        with timeout(timeout_seconds):
            exec(code, safe_globals, locals_dict)
        
        # Try to get the result object
        result = None
        if 'result' in safe_globals:
            result = safe_globals['result']
        
        return True, result, None
        
    except TimeoutError:
        return False, None, f"Execution timed out after {timeout_seconds} seconds"
    except Exception as e:
        error_msg = f"Execution error: {str(e)}"
        return False, None, error_msg


def validate_cadquery_code(code: str, timeout_seconds: float = 5.0) -> Tuple[bool, Optional[str]]:
    """Validate CadQuery code by attempting to execute it."""
    success, result, error = safe_execute(code, timeout_seconds)
    return success, error


def extract_code_features(code: str) -> Dict[str, Any]:
    """Extract features from CadQuery code for analysis."""
    features = {
        'length': len(code),
        'lines': len(code.split('\n')),
        'has_result': 'result =' in code or 'result=' in code,
        'has_workplane': 'cq.Workplane' in code,
        'has_imports': 'import cadquery' in code or 'import cq' in code,
        'operation_count': 0,
        'variable_count': 0,
        'comment_count': code.count('#'),
        'parentheses_balanced': code.count('(') == code.count(')'),
        'operations': {},
    }
    
    # Count operations
    operations = [
        'box', 'cylinder', 'sphere', 'cone', 'wedge', 'torus',
        'hole', 'cboreHole', 'countersinkHole',
        'faces', 'edges', 'vertices',
        'workplane', 'rect', 'circle', 'polygon',
        'cut', 'union', 'intersect',
        'translate', 'rotate', 'scale', 'mirror',
        'fillet', 'chamfer'
    ]
    
    for op in operations:
        count = code.count(f'.{op}(')
        if count > 0:
            features['operations'][op] = count
            features['operation_count'] += count
    
    # Count variable assignments
    import re
    var_pattern = r'(\w+)\s*=\s*[\d.]+'
    features['variable_count'] = len(re.findall(var_pattern, code))
    
    return features


def calculate_code_similarity(code1: str, code2: str) -> float:
    """Calculate similarity between two code snippets."""
    features1 = extract_code_features(code1)
    features2 = extract_code_features(code2)
    
    # Simple similarity based on feature overlap
    similarity = 0.0
    total_features = 0
    
    # Compare basic features
    basic_features = ['has_result', 'has_workplane', 'has_imports', 'parentheses_balanced']
    for feature in basic_features:
        if features1[feature] == features2[feature]:
            similarity += 1.0
        total_features += 1
    
    # Compare operation sets
    ops1 = set(features1['operations'].keys())
    ops2 = set(features2['operations'].keys())
    if ops1 or ops2:
        intersection = len(ops1.intersection(ops2))
        union = len(ops1.union(ops2))
        similarity += intersection / union if union > 0 else 0.0
        total_features += 1
    
    # Compare variable counts (normalized)
    var_diff = abs(features1['variable_count'] - features2['variable_count'])
    max_vars = max(features1['variable_count'], features2['variable_count'])
    if max_vars > 0:
        similarity += 1.0 - (var_diff / max_vars)
    else:
        similarity += 1.0
    total_features += 1
    
    return similarity / total_features if total_features > 0 else 0.0


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'rss_mb': 0.0, 'vms_mb': 0.0, 'percent': 0.0}


def log_memory_usage(logger: Optional[logging.Logger] = None) -> None:
    """Log current memory usage."""
    memory = get_memory_usage()
    message = f"Memory usage: RSS={memory['rss_mb']:.1f}MB, VMS={memory['vms_mb']:.1f}MB, {memory['percent']:.1f}%"
    
    if logger:
        logger.info(message)
    else:
        print(message)


def create_progress_bar(iterable, desc: str = "", **kwargs):
    """Create a progress bar with consistent styling."""
    from tqdm import tqdm
    
    return tqdm(
        iterable,
        desc=desc,
        leave=True,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        **kwargs
    )


def save_results(results: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save results to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results = convert_numpy(results)
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {path}")


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load results from JSON file."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    
    with open(path, 'r') as f:
        results = json.load(f)
    
    logging.info(f"Results loaded from {path}")
    return results


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    logger.info("Logging test successful")
    
    # Test seeding
    set_seed(42)
    print(f"Random number after seeding: {random.randint(1, 100)}")
    
    # Test code validation
    test_code = """
    import cadquery as cq
    result = cq.Workplane("XY").box(10, 10, 10)
    """
    success, error = validate_cadquery_code(test_code)
    print(f"Code validation: {success}, Error: {error}")
    
    # Test code features
    features = extract_code_features(test_code)
    print(f"Code features: {features}")
    
    print("Utility tests completed!")
