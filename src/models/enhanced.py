"""
Enhanced model for CadQuery code generation.
Baseline + grammar-aware decoding + execution-guided sampling + heuristic reranking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
import time
import subprocess
import tempfile
import os
from pathlib import Path
import re
import ast
from collections import Counter

from lark import Lark, Transformer, v_args
from lark.exceptions import LarkError

from .baseline import CadQueryBaselineModel, create_baseline_model


class CadQueryGrammarParser:
    """Grammar-aware parser for CadQuery code validation and repair."""
    
    def __init__(self, grammar_file: Optional[str] = None):
        if grammar_file is None:
            grammar_file = "grammars/cadquery_ebnf.lark"
        
        # Load grammar
        try:
            with open(grammar_file, 'r') as f:
                grammar_text = f.read()
            self.parser = Lark(grammar_text, start='start', parser='earley')
        except FileNotFoundError:
            # Fallback to basic grammar
            self.parser = self._create_basic_grammar()
    
    def _create_basic_grammar(self) -> Lark:
        """Create a basic CadQuery grammar for validation."""
        basic_grammar = """
        start: statement*
        
        statement: assignment | expression | comment
        
        assignment: NAME "=" expression
        expression: workplane_chain | method_call | literal
        
        workplane_chain: workplane method_call*
        workplane: "cq.Workplane" "(" STRING ")"
        method_call: "." NAME "(" arguments? ")"
        arguments: argument ("," argument)*
        argument: literal | NAME | expression
        
        literal: NUMBER | STRING | "True" | "False" | "None"
        
        comment: /#[^\\n]*/
        
        STRING: /"[^"]*"/
        NUMBER: /\\d+(\\.\\d+)?/
        NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
        
        %import common.WS
        %ignore WS
        """
        return Lark(basic_grammar, start='start', parser='earley')
    
    def is_grammatically_valid(self, code: str) -> bool:
        """Check if code is grammatically valid according to CadQuery grammar."""
        try:
            self.parser.parse(code)
            return True
        except LarkError:
            return False
    
    def repair_code(self, code: str) -> str:
        """Attempt to repair common syntax errors in CadQuery code."""
        repaired = code
        
        # Fix common issues
        repaired = self._fix_parentheses(repaired)
        repaired = self._fix_workplane_chain(repaired)
        repaired = self._fix_result_assignment(repaired)
        repaired = self._fix_method_chaining(repaired)
        
        return repaired
    
    def _fix_parentheses(self, code: str) -> str:
        """Fix unbalanced parentheses."""
        # Count parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        
        if open_parens > close_parens:
            # Add missing closing parentheses
            code += ')' * (open_parens - close_parens)
        elif close_parens > open_parens:
            # Remove extra closing parentheses from the end
            while close_parens > open_parens and code.endswith(')'):
                code = code[:-1]
                close_parens -= 1
        
        return code
    
    def _fix_workplane_chain(self, code: str) -> str:
        """Ensure proper workplane chain structure."""
        # Ensure workplane is properly initialized
        if 'cq.Workplane' in code and not code.strip().startswith('result'):
            # Add result assignment if missing
            if 'result =' not in code and 'result=' not in code:
                code = f"result = {code}"
        
        return code
    
    def _fix_result_assignment(self, code: str) -> str:
        """Ensure result variable is properly assigned."""
        # Look for CadQuery operations without assignment
        cq_patterns = [
            r'cq\.Workplane\("XY"\)',
            r'\.box\(',
            r'\.cylinder\(',
            r'\.sphere\(',
        ]
        
        for pattern in cq_patterns:
            if re.search(pattern, code):
                lines = code.split('\n')
                for i, line in enumerate(lines):
                    if any(re.search(p, line) for p in cq_patterns):
                        if '=' not in line or line.strip().startswith('='):
                            lines[i] = f"result = {line.strip()}"
                            break
                code = '\n'.join(lines)
                break
        
        return code
    
    def _fix_method_chaining(self, code: str) -> str:
        """Fix method chaining issues."""
        # Ensure proper method chaining
        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Fix lines that start with a dot but aren't properly chained
            if line.strip().startswith('.') and i > 0:
                prev_line = lines[i-1].strip()
                if not prev_line.endswith(')'):
                    # Add missing closing parenthesis
                    lines[i-1] = prev_line + ')'
        
        return '\n'.join(lines)


class ExecutionValidator:
    """Validates code execution in a safe environment."""
    
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
    
    def is_executable(self, code: str) -> bool:
        """Check if code can be executed without errors."""
        try:
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
                }
            }
            
            # Add CadQuery imports
            import cadquery as cq
            import numpy as np
            safe_globals.update({
                'cq': cq,
                'cadquery': cq,
                'np': np,
                'numpy': np
            })
            
            # Execute with timeout
            exec(code, safe_globals)
            return True
            
        except Exception as e:
            return False
    
    def execute_with_timeout(self, code: str) -> Tuple[bool, Any]:
        """Execute code with timeout and return success status and result."""
        try:
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
                }
            }
            
            # Add CadQuery imports
            import cadquery as cq
            import numpy as np
            safe_globals.update({
                'cq': cq,
                'cadquery': cq,
                'np': np,
                'numpy': np
            })
            
            # Execute with timeout
            result = exec(code, safe_globals)
            
            # Try to get the result object
            if 'result' in safe_globals:
                return True, safe_globals['result']
            else:
                return True, None
                
        except Exception as e:
            return False, str(e)


class HeuristicReranker:
    """Lightweight heuristic reranker for code quality."""
    
    def __init__(self):
        self.operation_weights = {
            'box': 1.0,
            'cylinder': 1.0,
            'sphere': 1.0,
            'hole': 0.8,
            'cboreHole': 0.9,
            'rect': 0.7,
            'circle': 0.7,
            'faces': 0.6,
            'vertices': 0.6,
            'edges': 0.6,
            'workplane': 0.5,
        }
    
    def score_code(self, code: str) -> float:
        """Score code based on heuristics."""
        score = 0.0
        
        # Operation diversity score
        operation_score = self._score_operations(code)
        score += operation_score * 0.4
        
        # Structure score
        structure_score = self._score_structure(code)
        score += structure_score * 0.3
        
        # Completeness score
        completeness_score = self._score_completeness(code)
        score += completeness_score * 0.3
        
        return score
    
    def _score_operations(self, code: str) -> float:
        """Score based on operation diversity and quality."""
        score = 0.0
        
        # Count operations
        operation_counts = Counter()
        for op, weight in self.operation_weights.items():
            count = code.count(f'.{op}(')
            operation_counts[op] = count
            score += count * weight
        
        # Bonus for operation diversity
        unique_ops = len([op for op, count in operation_counts.items() if count > 0])
        score += unique_ops * 0.1
        
        return min(score, 10.0)  # Cap at 10
    
    def _score_structure(self, code: str) -> float:
        """Score based on code structure."""
        score = 0.0
        
        # Proper result assignment
        if 'result =' in code or 'result=' in code:
            score += 2.0
        
        # Proper workplane initialization
        if 'cq.Workplane' in code:
            score += 1.0
        
        # Method chaining
        chain_count = code.count(').')
        score += min(chain_count * 0.5, 5.0)
        
        # Balanced parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens == close_parens:
            score += 1.0
        
        return score
    
    def _score_completeness(self, code: str) -> float:
        """Score based on code completeness."""
        score = 0.0
        
        # Variable definitions
        var_pattern = r'(\w+)\s*=\s*[\d.]+'
        vars_found = len(re.findall(var_pattern, code))
        score += min(vars_found * 0.5, 3.0)
        
        # Comments
        comment_count = code.count('#')
        score += min(comment_count * 0.2, 1.0)
        
        # Code length (not too short, not too long)
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        if 5 <= len(lines) <= 50:
            score += 1.0
        
        return score


class CadQueryEnhancedModel(CadQueryBaselineModel):
    """Enhanced model with grammar-aware decoding and execution-guided sampling."""
    
    def __init__(
        self,
        vision_model_name: str = "vit_base_patch16_224",
        code_model_name: str = "Salesforce/codet5-small",
        image_size: int = 224,
        max_code_length: int = 512,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        freeze_vision: bool = True,
        unfreeze_last_n_blocks: int = 2,
        use_grammar: bool = True,
        use_execution_guidance: bool = True,
        use_reranking: bool = True,
        best_of_n: int = 6,
        execution_timeout: float = 5.0
    ):
        super().__init__(
            vision_model_name=vision_model_name,
            code_model_name=code_model_name,
            image_size=image_size,
            max_code_length=max_code_length,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_vision=freeze_vision,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks
        )
        
        # Enhanced components
        self.use_grammar = use_grammar
        self.use_execution_guidance = use_execution_guidance
        self.use_reranking = use_reranking
        self.best_of_n = best_of_n
        self.execution_timeout = execution_timeout
        
        if use_grammar:
            self.grammar_parser = CadQueryGrammarParser()
        
        if use_execution_guidance:
            self.execution_validator = ExecutionValidator(timeout=execution_timeout)
        
        if use_reranking:
            self.reranker = HeuristicReranker()
    
    def generate_enhanced(
        self,
        images: torch.Tensor,
        max_length: Optional[int] = None,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """Generate code with enhanced features."""
        if not (self.use_grammar or self.use_execution_guidance or self.use_reranking):
            # Fall back to baseline generation
            return self.generate_text(
                images=images,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                **kwargs
            )
        
        # Generate multiple candidates
        candidates = self._generate_candidates(
            images=images,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # Apply enhancement pipeline
        enhanced_candidates = self._apply_enhancement_pipeline(candidates)
        
        # Return the best candidate for each image
        return [candidates[0] for candidates in enhanced_candidates]
    
    def _generate_candidates(
        self,
        images: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        **kwargs
    ) -> List[List[str]]:
        """Generate multiple candidates for each image."""
        batch_size = images.size(0)
        all_candidates = []
        
        for i in range(batch_size):
            image = images[i:i+1]  # Single image
            candidates = []
            
            for _ in range(self.best_of_n):
                # Generate with sampling
                generated_ids = self.code_decoder.model.generate(
                    inputs=torch.zeros(1, 1, dtype=torch.long, device=images.device),
                    encoder_hidden_states=self.projection(self.vision_encoder(image)),
                    max_length=max_length or self.max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
                
                # Decode to text
                candidate = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                candidates.append(candidate)
            
            all_candidates.append(candidates)
        
        return all_candidates
    
    def _apply_enhancement_pipeline(self, candidates: List[List[str]]) -> List[List[str]]:
        """Apply the enhancement pipeline to candidates."""
        enhanced_candidates = []
        
        for image_candidates in candidates:
            enhanced_image_candidates = []
            
            for candidate in image_candidates:
                enhanced_candidate = candidate
                
                # 1. Grammar validation and repair
                if self.use_grammar:
                    if not self.grammar_parser.is_grammatically_valid(enhanced_candidate):
                        enhanced_candidate = self.grammar_parser.repair_code(enhanced_candidate)
                
                # 2. Execution validation
                if self.use_execution_guidance:
                    is_executable, _ = self.execution_validator.execute_with_timeout(enhanced_candidate)
                    if not is_executable:
                        continue  # Skip this candidate
                
                # 3. Reranking
                if self.use_reranking:
                    score = self.reranker.score_code(enhanced_candidate)
                    enhanced_image_candidates.append((enhanced_candidate, score))
                else:
                    enhanced_image_candidates.append((enhanced_candidate, 0.0))
            
            # Sort by score and keep top candidates
            enhanced_image_candidates.sort(key=lambda x: x[1], reverse=True)
            enhanced_candidates.append([candidate for candidate, _ in enhanced_image_candidates])
        
        return enhanced_candidates


def create_enhanced_model(
    vision_model_name: str = "vit_base_patch16_224",
    code_model_name: str = "Salesforce/codet5-small",
    image_size: int = 224,
    max_code_length: int = 512,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    freeze_vision: bool = True,
    unfreeze_last_n_blocks: int = 2,
    use_grammar: bool = True,
    use_execution_guidance: bool = True,
    use_reranking: bool = True,
    best_of_n: int = 6,
    execution_timeout: float = 5.0
) -> CadQueryEnhancedModel:
    """Factory function to create enhanced model."""
    return CadQueryEnhancedModel(
        vision_model_name=vision_model_name,
        code_model_name=code_model_name,
        image_size=image_size,
        max_code_length=max_code_length,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        freeze_vision=freeze_vision,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        use_grammar=use_grammar,
        use_execution_guidance=use_execution_guidance,
        use_reranking=use_reranking,
        best_of_n=best_of_n,
        execution_timeout=execution_timeout
    )


if __name__ == "__main__":
    # Test the enhanced model
    print("Testing enhanced model...")
    
    # Create model
    model = create_enhanced_model()
    
    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Frozen parameters: {param_counts['frozen']:,}")
    
    # Test enhancement pipeline
    test_code = """
    height = 60.0
    width = 80.0
    thickness = 10.0
    result = cq.Workplane("XY").box(height, width, thickness)
    """
    
    print(f"Grammar valid: {model.grammar_parser.is_grammatically_valid(test_code)}")
    print(f"Executable: {model.execution_validator.is_executable(test_code)}")
    print(f"Reranker score: {model.reranker.score_code(test_code):.2f}")
    
    print("Enhanced model test completed!")
