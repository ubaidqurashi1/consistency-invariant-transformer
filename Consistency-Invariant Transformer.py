"""
Consistency-Invariant Transformer (CIT)
A self-improving language model that learns by minimizing violation of theoretical invariants
without any labeled data or human supervision.

Author: AI Research
License: MIT
Hugging Face Model: consistency-invariant-transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

class ConsistencyInvariantConfig:
    """Configuration for Consistency-Invariant Transformer"""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 2048,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        
        # Invariant-specific configurations
        use_temporal_invariant: bool = True,
        use_causal_invariant: bool = True,
        use_math_invariant: bool = True,
        use_logical_invariant: bool = True,
        invariant_loss_weights: Optional[Dict[str, float]] = None,
        
        # Training configurations
        learning_rate: float = 2e-4,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.95,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Invariant configurations
        self.use_temporal_invariant = use_temporal_invariant
        self.use_causal_invariant = use_causal_invariant
        self.use_math_invariant = use_math_invariant
        self.use_logical_invariant = use_logical_invariant
        
        # Default loss weights if not provided
        if invariant_loss_weights is None:
            self.invariant_loss_weights = {
                'temporal': 0.8,
                'causal': 0.8,
                'math': 1.2,
                'logical': 0.6,
                'lexical': 1.0,
                'factual': 0.7
            }
        else:
            self.invariant_loss_weights = invariant_loss_weights
            
        # Training configurations
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def save_pretrained(self, save_directory: str):
        """Save configuration to file"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        """Load configuration from file"""
        import os
        config_path = os.path.join(model_name_or_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better sequence modeling"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute sinusoidal frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: int):
        """
        x: [batch_size, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape
        device = x.device
        
        # Create position indices
        pos = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        
        # Apply rotation
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated_x = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated_x.flatten(-2, -1)

class MultiHeadAttention(nn.Module):
    """Multi-head attention with rotary position embeddings"""
    
    def __init__(self, config: ConsistencyInvariantConfig):
        super().__init__()
        self.config = config
        
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.rotary_emb = RotaryPositionEmbedding(self.head_dim, config.max_position_embeddings)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Linear projections
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embeddings
        q = self.rotary_emb(q.transpose(1, 2), seq_len).transpose(1, 2)
        k = self.rotary_emb(k.transpose(1, 2), seq_len).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Attention probabilities
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        
        # Context layer
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        return self.out(context)

class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization"""
    
    def __init__(self, config: ConsistencyInvariantConfig):
        super().__init__()
        self.config = config
        
        # Pre-norm architecture
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = MultiHeadAttention(config)
        
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attention_output
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states

class ConsistencyInvariantTransformer(nn.Module):
    """Main model implementing the consistency-invariant framework"""
    
    def __init__(self, config: ConsistencyInvariantConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Specialized invariant heads
        self.temporal_head = None
        self.causal_head = None
        self.math_head = None
        
        if config.use_temporal_invariant:
            self.temporal_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(config.hidden_size // 2, 3)  # BEFORE, AFTER, SIMULTANEOUS
            )
        
        if config.use_causal_invariant:
            self.causal_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(config.hidden_size // 2, 2)  # CAUSES, NO_CAUSE
            )
        
        if config.use_math_invariant:
            self.math_head = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, 1)  # Consistency score
            )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        # Create causal mask
        device = input_ids.device
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
        causal_mask = causal_mask.expand(batch_size, self.config.num_attention_heads, seq_len, seq_len)
        
        # Combine with attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.expand(-1, self.config.num_attention_heads, seq_len, -1)
            attention_mask = causal_mask * attention_mask
            attention_mask = (1.0 - attention_mask) * -1e9
        
        # Embeddings
        hidden_states = self.embedding(input_ids)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Prepare output dictionary
        output = {
            "logits": logits,
            "hidden_states": hidden_states
        }
        
        # Invariant-specific outputs
        if self.temporal_head is not None:
            output["temporal_logits"] = self.temporal_head(hidden_states)
        
        if self.causal_head is not None:
            output["causal_logits"] = self.causal_head(hidden_states)
        
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Generate text using the model"""
        self.eval()
        
        generated = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        
        for _ in range(max_length):
            # Forward pass
            outputs = self(generated, attention_mask=attention_mask)
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token in set(generated[0].tolist()):
                    next_token_logits[:, token] /= repetition_penalty
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift indices to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = -float('Inf')
            
            # Sample or take argmax
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            
            # Stop if EOS token
            if next_token.item() == self.config.eos_token_id:
                break
        
        self.train()
        return generated

class InvariantTrainer:
    """Trainer for consistency-invariant learning"""
    
    def __init__(self, model: ConsistencyInvariantTransformer, config: ConsistencyInvariantConfig):
        self.model = model
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,  # Will be updated during training
            eta_min=config.learning_rate * 0.1
        )
        
        # Track training history
        self.history = {
            'total_loss': [],
            'invariant_losses': {},
            'learning_rates': []
        }
    
    def compute_lexical_invariant_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Lexical consistency: paraphrases should have similar outputs"""
        batch_size, seq_len = batch.shape
        
        # Create paraphrased batch (simplified: random token swaps)
        paraphrased = batch.clone()
        for i in range(batch_size):
            # Randomly swap some tokens (10% chance per position)
            for j in range(1, seq_len - 1):  # Skip BOS and EOS
                if random.random() < 0.1:
                    # Swap with random token from vocabulary
                    paraphrased[i, j] = random.randint(10, min(1000, self.config.vocab_size - 1))
        
        # Get model outputs
        orig_output = self.model(batch)
        para_output = self.model(paraphrased)
        
        # KL divergence between distributions
        orig_probs = F.log_softmax(orig_output["logits"][:, :-1], dim=-1)
        para_probs = F.softmax(para_output["logits"][:, :-1], dim=-1)
        
        kl_loss = F.kl_div(orig_probs, para_probs, reduction='batchmean')
        
        return kl_loss
    
    def compute_temporal_invariant_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Temporal consistency: temporal relations should be logically consistent"""
        if not self.config.use_temporal_invariant:
            return torch.tensor(0.0, device=batch.device)
        
        outputs = self.model(batch)
        temporal_logits = outputs["temporal_logits"]
        
        # Temporal consistency loss: predictions should be stable across sequence
        temporal_variance = temporal_logits.var(dim=1).mean()
        
        return temporal_variance
    
    def compute_causal_invariant_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Causal consistency: cause-effect relations should be asymmetric"""
        if not self.config.use_causal_invariant:
            return torch.tensor(0.0, device=batch.device)
        
        outputs = self.model(batch)
        causal_logits = outputs["causal_logits"]
        
        # Causal asymmetry loss: if A causes B, then B should not cause A
        # Simplified: encourage low variance in causal predictions
        causal_variance = causal_logits.var(dim=1).mean()
        
        return causal_variance
    
    def compute_math_invariant_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Mathematical consistency: equations should be internally consistent"""
        if not self.config.use_math_invariant or self.model.math_head is None:
            return torch.tensor(0.0, device=batch.device)
        
        outputs = self.model(batch)
        hidden_states = outputs["hidden_states"]
        
        # Find positions with mathematical tokens (simplified: positions 1000-1100)
        math_positions = []
        for b in range(batch.shape[0]):
            positions = []
            for i, token in enumerate(batch[b]):
                if 1000 <= token < 1100:  # Assuming math tokens in this range
                    positions.append(i)
            if len(positions) >= 2:
                math_positions.append((b, positions))
        
        if not math_positions:
            return torch.tensor(0.0, device=batch.device)
        
        # Compute mathematical consistency scores
        math_losses = []
        for b, positions in math_positions:
            # Get pairs of math-related positions
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    pair = torch.cat([
                        hidden_states[b, positions[i]],
                        hidden_states[b, positions[j]]
                    ])
                    # Consistency score should be high for mathematically related positions
                    consistency = self.model.math_head(pair.unsqueeze(0))
                    math_losses.append(1.0 - consistency)  # We want consistency ~1
        
        return torch.stack(math_losses).mean() if math_losses else torch.tensor(0.0, device=batch.device)
    
    def compute_logical_invariant_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Logical consistency: logical statements should be valid"""
        if not self.config.use_logical_invariant:
            return torch.tensor(0.0, device=batch.device)
        
        outputs = self.model(batch)
        hidden_states = outputs["hidden_states"]
        
        # Logical consistency: hidden states of logical operators should be stable
        # Find logical operator positions (assuming special tokens for logical ops)
        logical_positions = []
        for b in range(batch.shape[0]):
            positions = []
            for i, token in enumerate(batch[b]):
                # Simplified: assume tokens 2000-2100 are logical operators
                if 2000 <= token < 2100:
                    positions.append(i)
            if len(positions) >= 2:
                logical_positions.append((b, positions))
        
        if not logical_positions:
            return torch.tensor(0.0, device=batch.device)
        
        # Compute variance of logical operator representations
        logical_variances = []
        for b, positions in logical_positions:
            logical_hidden = hidden_states[b, positions]
            variance = logical_hidden.var(dim=0).mean()
            logical_variances.append(variance)
        
        return torch.stack(logical_variances).mean() if logical_variances else torch.tensor(0.0, device=batch.device)
    
    def compute_factual_invariant_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Factual consistency: entities should have consistent properties"""
        # Simplified: use perplexity as proxy for factual consistency
        outputs = self.model(batch)
        logits = outputs["logits"]
        
        # Compute perplexity (lower is better for factual consistency)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Perform a single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        device = batch.device
        
        # Compute all invariant losses
        losses = {}
        
        # Lexical invariant
        losses['lexical'] = self.compute_lexical_invariant_loss(batch)
        
        # Temporal invariant
        losses['temporal'] = self.compute_temporal_invariant_loss(batch)
        
        # Causal invariant
        losses['causal'] = self.compute_causal_invariant_loss(batch)
        
        # Mathematical invariant
        losses['math'] = self.compute_math_invariant_loss(batch)
        
        # Logical invariant
        losses['logical'] = self.compute_logical_invariant_loss(batch)
        
        # Factual invariant (perplexity)
        losses['factual'] = self.compute_factual_invariant_loss(batch)
        
        # Weight losses according to configuration
        total_loss = torch.tensor(0.0, device=device)
        for key, loss in losses.items():
            weight = self.config.invariant_loss_weights.get(key, 1.0)
            total_loss = total_loss + weight * loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Record history
        self.history['total_loss'].append(total_loss.item())
        self.history['learning_rates'].append(self.scheduler.get_last_lr()[0])
        
        for key, loss in losses.items():
            if key not in self.history['invariant_losses']:
                self.history['invariant_losses'][key] = []
            self.history['invariant_losses'][key].append(loss.item())
        
        # Return loss values
        result = {'total_loss': total_loss.item()}
        result.update({f'{key}_loss': loss.item() for key, loss in losses.items()})
        result['learning_rate'] = self.scheduler.get_last_lr()[0]
        
        return result
    
    def save_pretrained(self, save_directory: str):
        """Save model and trainer state"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)
        
        # Save optimizer state
        optimizer_path = os.path.join(save_directory, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # Save scheduler state
        scheduler_path = os.path.join(save_directory, "scheduler.pt")
        torch.save(self.scheduler.state_dict(), scheduler_path)
        
        # Save training history
        history_path = os.path.join(save_directory, "history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_pretrained(self, save_directory: str):
        """Load model and trainer state"""
        import os
        
        # Load model
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        # Load optimizer state
        optimizer_path = os.path.join(save_directory, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
        
        # Load scheduler state
        scheduler_path = os.path.join(save_directory, "scheduler.pt")
        if os.path.exists(scheduler_path):
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location='cpu'))
        
        # Load training history
        history_path = os.path.join(save_directory, "history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.history = json.load(f)

class DataGenerator:
    """Generate synthetic training data with various reasoning patterns"""
    
    def __init__(self, config: ConsistencyInvariantConfig):
        self.config = config
        
        # Vocabulary subsets for different reasoning types
        self.temporal_words = list(range(3000, 3100))
        self.causal_words = list(range(3100, 3200))
        self.math_words = list(range(3200, 3300))
        self.logical_words = list(range(3300, 3400))
        
    def generate_batch(self, batch_size: int, seq_len: int = 64) -> torch.Tensor:
        """Generate a batch of synthetic sequences"""
        batch = []
        
        for _ in range(batch_size):
            # Start with BOS token
            sequence = [self.config.bos_token_id]
            
            # Decide reasoning type for this sequence
            reasoning_type = random.choice(['temporal', 'causal', 'math', 'logical', 'mixed'])
            
            if reasoning_type == 'temporal':
                # Temporal reasoning pattern
                sequence.extend([
                    random.choice(self.temporal_words),  # Event 1
                    random.choice([3001, 3002]),  # Temporal relation (before/after)
                    random.choice(self.temporal_words),  # Event 2
                    self.config.eos_token_id
                ])
            
            elif reasoning_type == 'causal':
                # Causal reasoning pattern
                sequence.extend([
                    random.choice(self.causal_words),  # Cause
                    random.choice([3101, 3102]),  # Causal relation
                    random.choice(self.causal_words),  # Effect
                    self.config.eos_token_id
                ])
            
            elif reasoning_type == 'math':
                # Mathematical reasoning pattern
                sequence.extend([
                    random.choice(self.math_words),  # Number/variable
                    random.choice([3201, 3202, 3203, 3204]),  # Operator
                    random.choice(self.math_words),  # Number/variable
                    3205,  # Equals
                    random.choice(self.math_words),  # Result
                    self.config.eos_token_id
                ])
            
            elif reasoning_type == 'logical':
                # Logical reasoning pattern
                sequence.extend([
                    random.choice(self.logical_words),  # Proposition A
                    random.choice([3301, 3302, 3303]),  # Logical connective
                    random.choice(self.logical_words),  # Proposition B
                    self.config.eos_token_id
                ])
            
            else:  # mixed
                # Mixed reasoning pattern
                sequence.extend([
                    random.choice(self.temporal_words + self.causal_words + self.math_words + self.logical_words)
                    for _ in range(random.randint(3, 8))
                ])
                sequence.append(self.config.eos_token_id)
            
            # Pad or truncate
            if len(sequence) < seq_len:
                sequence = sequence + [self.config.pad_token_id] * (seq_len - len(sequence))
            else:
                sequence = sequence[:seq_len]
                sequence[-1] = self.config.eos_token_id
            
            batch.append(sequence)
        
        return torch.tensor(batch)

# Example usage and demonstration
def main():
    """Demonstrate the consistency-invariant transformer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Consistency-Invariant Transformer")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--generate", action="store_true", help="Generate text")
    parser.add_argument("--save_dir", type=str, default="./cit_model", help="Directory to save model")
    parser.add_argument("--load_dir", type=str, help="Directory to load model from")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch")
    
    args = parser.parse_args()
    
    # Configuration
    config = ConsistencyInvariantConfig(
        vocab_size=10000,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=1024,
        
        # Enable all invariants
        use_temporal_invariant=True,
        use_causal_invariant=True,
        use_math_invariant=True,
        use_logical_invariant=True,
        
        # Loss weights
        invariant_loss_weights={
            'temporal': 0.8,
            'causal': 0.8,
            'math': 1.2,
            'logical': 0.6,
            'lexical': 1.0,
            'factual': 0.7
        }
    )
    
    # Initialize model
    model = ConsistencyInvariantTransformer(config)
    
    # Initialize trainer
    trainer = InvariantTrainer(model, config)
    
    # Data generator
    data_gen = DataGenerator(config)
    
    if args.train:
        print("Training Consistency-Invariant Transformer...")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training loop
        for epoch in range(args.epochs):
            epoch_losses = []
            
            for step in range(args.steps_per_epoch):
                # Generate batch
                batch = data_gen.generate_batch(args.batch_size, seq_len=64)
                
                # Train step
                losses = trainer.train_step(batch)
                epoch_losses.append(losses['total_loss'])
                
                # Print progress
                if (step + 1) % 10 == 0:
                    avg_loss = np.mean(epoch_losses[-10:])
                    print(f"Epoch {epoch+1}, Step {step+1}/{args.steps_per_epoch}, Loss: {avg_loss:.4f}")
            
            # End of epoch
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save model
        config.save_pretrained(args.save_dir)
        trainer.save_pretrained(args.save_dir)
        print(f"Model saved to {args.save_dir}")
    
    if args.generate:
        # Load model if specified
        if args.load_dir:
            config = ConsistencyInvariantConfig.from_pretrained(args.load_dir)
            model = ConsistencyInvariantTransformer(config)
            trainer = InvariantTrainer(model, config)
            trainer.load_pretrained(args.load_dir)
            print(f"Model loaded from {args.load_dir}")
        
        # Generate text
        prompt = torch.tensor([[config.bos_token_id, 3000, 3001, 3002]])  # Temporal prompt
        generated = model.generate(prompt, max_length=20, temperature=0.8)
        
        print("\nGenerated text (decoded):")
        print(f"Prompt: {prompt.tolist()[0]}")
        print(f"Generated: {generated.tolist()[0]}")
        
        # Demonstrate invariant computation
        print("\nComputing invariant losses for a sample batch...")
        sample_batch = data_gen.generate_batch(4, seq_len=32)
        
        with torch.no_grad():
            lexical_loss = trainer.compute_lexical_invariant_loss(sample_batch)
            temporal_loss = trainer.compute_temporal_invariant_loss(sample_batch)
            causal_loss = trainer.compute_causal_invariant_loss(sample_batch)
            math_loss = trainer.compute_math_invariant_loss(sample_batch)
            
            print(f"Lexical invariant loss: {lexical_loss.item():.4f}")
            print(f"Temporal invariant loss: {temporal_loss.item():.4f}")
            print(f"Causal invariant loss: {causal_loss.item():.4f}")
            print(f"Mathematical invariant loss: {math_loss.item():.4f}")
    
    if not args.train and not args.generate:
        print("Please specify --train or --generate flag")
        print("\nExample usage:")
        print("  python consistency_invariant.py --train --save_dir ./my_model")
        print("  python consistency_invariant.py --generate --load_dir ./my_model")

if __name__ == "__main__":
    main()