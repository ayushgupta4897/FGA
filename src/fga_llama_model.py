"""
FGA-enhanced Llama model with hard constraints at output
Integration with Llama 3.2 3B for smartphone specifications POC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from typing import Optional, Tuple, Dict, Any, List
from .fga_attention import FGALlamaAttention
from .knowledge_db import KnowledgeDatabase


class FGALlamaModel(nn.Module):
    """
    Llama model with FGA attention layers
    Replaces select attention layers with FGA-enhanced versions
    """
    
    def __init__(
        self,
        base_model_path: str,
        knowledge_db: KnowledgeDatabase,
        fga_layers: List[int] = None,
        constraint_threshold: float = 0.8,
        per_head_grounding: bool = False,
        device: str = "mps"
    ):
        super().__init__()
        
        # Load base Llama model
        self.device = device if torch.backends.mps.is_available() else "cpu"
        self.base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            device_map=self.device
        )
        
        self.config = self.base_model.config
        self.knowledge_db = knowledge_db
        self.constraint_threshold = constraint_threshold
        
        # Determine which layers to enhance with FGA
        # Default: top 8 layers for generation focus
        if fga_layers is None:
            total_layers = self.config.num_hidden_layers  # 28 for Llama 3.2 3B
            fga_layers = list(range(total_layers - 8, total_layers))
        
        self.fga_layers = fga_layers
        
        # Replace attention layers with FGA versions
        for layer_idx in self.fga_layers:
            original_attn = self.base_model.model.layers[layer_idx].self_attn
            
            # Create FGA attention with same config
            fga_attn = FGALlamaAttention(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                num_key_value_heads=self.config.num_key_value_heads,
                knowledge_db=knowledge_db,
                constraint_threshold=constraint_threshold,
                per_head_grounding=per_head_grounding,
                entity_stride=16
            )
            
            # Copy weights from original attention
            with torch.no_grad():
                fga_attn.q_proj.weight.copy_(original_attn.q_proj.weight)
                fga_attn.k_proj.weight.copy_(original_attn.k_proj.weight)
                fga_attn.v_proj.weight.copy_(original_attn.v_proj.weight)
                fga_attn.o_proj.weight.copy_(original_attn.o_proj.weight)
            
            # Replace layer
            self.base_model.model.layers[layer_idx].self_attn = fga_attn
        
        # Build constraint vocabularies for hard constraints
        self.constraint_vocabs = self._build_constraint_vocabs()
        
        # Track FGA info for analysis
        self.last_fga_info = {}
        
        print(f"✓ FGA Model initialized on {self.device.upper()}")
        print(f"  Enhanced layers: {self.fga_layers}")
    
    def _build_constraint_vocabs(self) -> Dict[str, List[int]]:
        """Build vocabulary indices for different field types"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model.config._name_or_path)
        
        constraint_vocabs = {}
        
        # Numeric tokens for specifications
        numeric_tokens = []
        for num in range(10000):  # Cover common spec numbers
            tokens = tokenizer.encode(str(num), add_special_tokens=False)
            numeric_tokens.extend(tokens)
        
        # Common units
        for unit in ['GB', 'TB', 'MP', 'mAh', 'Hz', 'W', 'mm', 'inch', 'inches']:
            tokens = tokenizer.encode(unit, add_special_tokens=False)
            numeric_tokens.extend(tokens)
        
        constraint_vocabs['numeric'] = list(set(numeric_tokens))
        
        # Date tokens
        date_tokens = []
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
        for month in months:
            tokens = tokenizer.encode(month, add_special_tokens=False)
            date_tokens.extend(tokens)
        
        constraint_vocabs['date'] = list(set(date_tokens))
        
        return constraint_vocabs
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        apply_constraints: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass with FGA enhancements and optional hard constraints
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=True,
            return_dict=True
        )
        
        # Apply hard constraints at logits if needed
        if apply_constraints and self.last_fga_info.get('alpha') is not None:
            alpha_max = self.last_fga_info['alpha'].max().item()
            
            if alpha_max > self.constraint_threshold:
                entities = self.last_fga_info.get('entities', [])
                metadata = self.last_fga_info.get('metadata', [])
                
                if entities and metadata:
                    # Apply vocabulary constraints for numeric fields
                    constraint_mask = self._create_vocab_constraint_mask(entities, metadata)
                    
                    if constraint_mask is not None:
                        outputs.logits = outputs.logits + constraint_mask.to(outputs.logits.device)
        
        return outputs
    
    def _create_vocab_constraint_mask(
        self, 
        entities: List[str], 
        metadata: List[Dict]
    ) -> Optional[torch.Tensor]:
        """Create vocabulary constraint mask for hard constraints"""
        vocab_size = self.config.vocab_size
        constraint_mask = torch.zeros(vocab_size)
        
        # Check if we're constraining numeric fields
        constrain_numeric = False
        for meta in metadata:
            if any(key in meta for key in ['battery_capacity', 'main_camera_mp', 'ram']):
                constrain_numeric = True
                break
        
        if constrain_numeric and 'numeric' in self.constraint_vocabs:
            # Apply numeric constraints
            allowed_tokens = self.constraint_vocabs['numeric']
            mask = torch.full_like(constraint_mask, -float('inf'))
            mask[allowed_tokens] = 0.0
            constraint_mask = mask
        
        return constraint_mask if torch.any(torch.isfinite(constraint_mask)) else None
    
    @torch.no_grad()
    def generate_with_fga(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 0.1,  # Low temp for factual accuracy
        do_sample: bool = False,
        apply_constraints: bool = True,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """
        Generate text with FGA grounding
        Returns generated tokens and FGA analysis info
        """
        self.eval()
        
        # Track FGA activations
        fga_activations = []
        
        # Generate
        outputs = self.base_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Collect FGA info
        fga_info = {
            'entities_detected': self.last_fga_info.get('entities', []),
            'max_alpha': self.last_fga_info.get('alpha', torch.tensor(0.0)).max().item() if self.last_fga_info.get('alpha') is not None else 0.0,
            'constraints_applied': apply_constraints and self.last_fga_info.get('alpha', torch.tensor(0.0)).max().item() > self.constraint_threshold if self.last_fga_info.get('alpha') is not None else False
        }
        
        return outputs.sequences, fga_info
