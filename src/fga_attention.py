"""
FGA Attention Layer for Llama models
Minimal implementation with dimensional correctness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List, Any


class ChunkedEntityRecognizer(nn.Module):
    """Efficient chunked entity recognition to reduce per-token latency"""
    
    def __init__(self, stride: int = 16, tokenizer=None):
        super().__init__()
        self.stride = stride
        self.entity_cache = {}
        self.cache_size = 1000
        self.tokenizer = tokenizer
        
        # Real entity patterns for smartphone detection
        self.phone_patterns = {
            'iphone 15 pro max': 'phone:iphone_15_pro_max',
            'iphone 15 pro': 'phone:iphone_15_pro',
            'iphone 15': 'phone:iphone_15',
            'galaxy s24 ultra': 'phone:galaxy_s24_ultra',
            'galaxy s24': 'phone:galaxy_s24',
            'pixel 8 pro': 'phone:pixel_8_pro', 
            'pixel 8': 'phone:pixel_8',
            's24 ultra': 'phone:galaxy_s24_ultra',
            's24': 'phone:galaxy_s24',
        }
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache: Optional[Dict] = None
    ) -> Tuple[List[str], List[List[int]]]:
        """Process entities every N tokens for efficiency"""
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Only process at stride boundaries
        if position_ids is None:
            position_ids = torch.arange(seq_len)
            
        current_pos = position_ids[0].item() if len(position_ids) > 0 else 0
        
        if current_pos % self.stride != 0 and cache:
            # Return cached entities
            return cache.get('entities', []), cache.get('positions', [])
        
        entities = []
        positions = []
        
        # Real entity detection using input tokens
        if input_ids is not None and self.tokenizer is not None:
            # Decode tokens to text
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).lower()
            
            # Search for phone entities
            for pattern, entity_id in self.phone_patterns.items():
                if pattern in text:
                    # Find token positions for this entity
                    pattern_tokens = self.tokenizer.encode(pattern, add_special_tokens=False)
                    
                    # Simple sliding window search for token positions
                    for i in range(seq_len - len(pattern_tokens) + 1):
                        if input_ids[0, i:i+len(pattern_tokens)].tolist() == pattern_tokens:
                            entities.append(entity_id)
                            positions.append(list(range(i, i + len(pattern_tokens))))
                            break
        
        if cache is not None:
            cache['entities'] = entities
            cache['positions'] = positions
            
        return entities, positions


class FactGate(nn.Module):
    """Learnable gate for fact grounding activation"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, query_states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute gate activation value"""
        # Combine query and context information
        gate_input = torch.cat([
            query_states.mean(dim=1),  # [batch, hidden]
            context.mean(dim=1) if len(context.shape) > 2 else context  # [batch, hidden]
        ], dim=-1)
        
        # Compute gate value with sigmoid activation
        alpha = torch.sigmoid(self.gate_proj(gate_input))  # [batch, 1]
        return alpha


class FGALlamaAttention(nn.Module):
    """
    FGA-enhanced Llama attention with dimensional correctness
    Replaces standard Llama attention layer
    """
    
    def __init__(
        self,
        hidden_size: int = 3072,
        num_heads: int = 24,
        num_key_value_heads: int = 8,
        knowledge_db = None,
        constraint_threshold: float = 0.8,
        per_head_grounding: bool = False,
        entity_stride: int = 16
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        
        # Standard Llama attention projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # FGA components
        self.knowledge_db = knowledge_db
        self.entity_recognizer = ChunkedEntityRecognizer(stride=entity_stride)
        self.fact_gate = FactGate(hidden_size)
        
        # Separate projectors for facts to key/query space
        self.fact_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fact_q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.constraint_threshold = constraint_threshold
        self.per_head_grounding = per_head_grounding
        
        if per_head_grounding:
            self.head_gates = nn.ModuleList([
                nn.Linear(hidden_size * 2, 1) for _ in range(num_heads)
            ])
        
        print(f"✓ FGA Attention initialized: {num_heads} heads, stride={entity_stride}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Dict[str, Any]]:
        """
        Forward pass with FGA modifications
        Returns: (attn_output, past_key_value, fga_info)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Standard Llama attention projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Repeat k/v for grouped query attention
        key_states = self._repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
        value_states = self._repeat_kv(value_states, self.num_heads // self.num_key_value_heads)
        
        # Standard attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # FGA: Entity recognition and fact grounding
        entities, entity_positions = self.entity_recognizer(
            hidden_states, position_ids, cache=getattr(self, 'entity_cache', {})
        )
        
        fga_info = {'entities': entities, 'alpha': None, 'metadata': None}
        
        if entities and self.knowledge_db:
            # Retrieve fact vectors
            fact_vectors, metadata = self.knowledge_db.batch_lookup_with_metadata(entities)
            fga_info['metadata'] = metadata
            
            if fact_vectors is not None:
                # Project facts to key space (dimensional fix)
                K_fact = self.fact_k_proj(fact_vectors)  # [M, hidden_size]
                K_fact = K_fact.view(len(entities), self.num_heads, self.head_dim)  # [M, heads, head_dim]
                
                # Compute query-fact affinities
                Q_fact_bias = torch.matmul(query_states, K_fact.transpose(-2, -1)) / math.sqrt(self.head_dim)
                # Q_fact_bias shape: [batch, heads, seq_len, M]
                
                # Create entity assignment matrix
                A = self._create_assignment_matrix(entity_positions, seq_len).to(query_states.device)
                A = A.unsqueeze(0).unsqueeze(0)  # [1, 1, M, seq_len]
                
                # Compute grounding scores (dimensional correction)
                G = torch.matmul(Q_fact_bias, A)  # [batch, heads, seq_len, seq_len]
                
                # Compute fact gate
                alpha = self.fact_gate(hidden_states, hidden_states)  # [batch, 1]
                alpha = alpha.unsqueeze(1).unsqueeze(-1)  # [batch, 1, 1, 1]
                fga_info['alpha'] = alpha.squeeze()
                
                # Apply confidence weighting if available
                if metadata and 'confidence' in metadata[0]:
                    conf = torch.tensor([m.get('confidence', 1.0) for m in metadata]).mean()
                    alpha = alpha * conf
                
                # Add grounding scores to attention weights
                attn_weights = attn_weights + alpha * G
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # Handle past key values for caching
        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None
        
        return attn_output, past_key_value, fga_info
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value states for grouped query attention"""
        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)
    
    def _create_assignment_matrix(
        self, 
        entity_positions: List[List[int]], 
        seq_len: int
    ) -> torch.Tensor:
        """Create entity assignment matrix A ∈ {0,1}^(M×L)"""
        if not entity_positions:
            return torch.zeros(1, seq_len)
        
        M = len(entity_positions)
        A = torch.zeros(M, seq_len)
        
        for i, positions in enumerate(entity_positions):
            for pos in positions:
                if 0 <= pos < seq_len:
                    A[i, pos] = 1.0
        
        return A
