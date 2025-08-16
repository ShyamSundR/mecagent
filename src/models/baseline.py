"""
Baseline model for CadQuery code generation.
ViT encoder + CodeT5 decoder with LoRA for efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import timm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class VisionEncoder(nn.Module):
    """Vision Transformer encoder for image processing."""
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 2
    ):
        super().__init__()
        
        # Load pretrained ViT
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Return all tokens
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Freeze/unfreeze layers
        if freeze_backbone:
            self._freeze_backbone(unfreeze_last_n_blocks)
    
    def _freeze_backbone(self, unfreeze_last_n_blocks: int):
        """Freeze backbone layers except last N blocks."""
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last N transformer blocks
        if hasattr(self.backbone, 'blocks'):
            total_blocks = len(self.backbone.blocks)
            start_unfreeze = max(0, total_blocks - unfreeze_last_n_blocks)
            
            for i in range(start_unfreeze, total_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through vision encoder."""
        # Get features from backbone
        features = self.backbone.forward_features(images)
        
        # Add CLS token if not present
        if features.dim() == 3 and features.size(1) > 1:
            # Use mean pooling if no CLS token
            features = features.mean(dim=1, keepdim=True)
        
        return features


class CodeDecoder(nn.Module):
    """CodeT5 decoder for code generation."""
    
    def __init__(
        self,
        model_name: str = "Salesforce/codet5-small",
        max_length: int = 512,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32
    ):
        super().__init__()
        
        # Load pretrained CodeT5
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
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
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha)
    
    def _apply_lora(self, rank: int, alpha: int):
        """Apply LoRA to the model for efficient fine-tuning."""
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.1,
            target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through code decoder."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            labels=labels
        )
        
        return outputs
    
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        max_length: Optional[int] = None,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate code from encoder hidden states."""
        if max_length is None:
            max_length = self.max_length
        
        # Create dummy input for generation
        batch_size = encoder_hidden_states.size(0)
        dummy_input = torch.zeros(batch_size, 1, dtype=torch.long, device=encoder_hidden_states.device)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        generated_ids = self.model.generate(
            inputs=dummy_input,
            encoder_hidden_states=encoder_hidden_states,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        return generated_ids


class CadQueryBaselineModel(nn.Module):
    """Baseline model combining ViT encoder and CodeT5 decoder."""
    
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
        unfreeze_last_n_blocks: int = 2
    ):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(
            model_name=vision_model_name,
            freeze_backbone=freeze_vision,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks
        )
        
        # Code decoder
        self.code_decoder = CodeDecoder(
            model_name=code_model_name,
            max_length=max_code_length,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha
        )
        
        # Projection layer to match dimensions
        self.projection = nn.Linear(
            self.vision_encoder.feature_dim,
            self.code_decoder.model.config.d_model
        )
        
        # Tokenizer for convenience
        self.tokenizer = self.code_decoder.tokenizer
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        # Encode images
        vision_features = self.vision_encoder(images)
        
        # Project to decoder dimension
        projected_features = self.projection(vision_features)
        
        # Decode code
        outputs = self.code_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=projected_features,
            labels=labels
        )
        
        return outputs
    
    def generate(
        self,
        images: torch.Tensor,
        max_length: Optional[int] = None,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate code from images."""
        # Encode images
        vision_features = self.vision_encoder(images)
        
        # Project to decoder dimension
        projected_features = self.projection(vision_features)
        
        # Generate code
        generated_ids = self.code_decoder.generate(
            encoder_hidden_states=projected_features,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            **kwargs
        )
        
        return generated_ids
    
    def generate_text(
        self,
        images: torch.Tensor,
        max_length: Optional[int] = None,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        **kwargs
    ) -> List[str]:
        """Generate code text from images."""
        generated_ids = self.generate(
            images=images,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            **kwargs
        )
        
        # Decode to text
        generated_texts = []
        for ids in generated_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


def create_baseline_model(
    vision_model_name: str = "vit_base_patch16_224",
    code_model_name: str = "Salesforce/codet5-small",
    image_size: int = 224,
    max_code_length: int = 512,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    freeze_vision: bool = True,
    unfreeze_last_n_blocks: int = 2
) -> CadQueryBaselineModel:
    """Factory function to create baseline model."""
    return CadQueryBaselineModel(
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


if __name__ == "__main__":
    # Test the baseline model
    print("Testing baseline model...")
    
    # Create model
    model = create_baseline_model()
    
    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Frozen parameters: {param_counts['frozen']:,}")
    
    # Test forward pass
    batch_size = 2
    image_size = 224
    seq_length = 100
    
    images = torch.randn(batch_size, 3, image_size, image_size)
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 1000, (batch_size, seq_length))
    
    outputs = model(images, input_ids, attention_mask, labels)
    print(f"Loss: {outputs.loss.item():.4f}")
    
    # Test generation
    generated_ids = model.generate(images, max_length=50)
    print(f"Generated shape: {generated_ids.shape}")
    
    print("Baseline model test completed!")
