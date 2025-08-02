import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM

class VectorInjectedAttention(LlamaAttention):
    """
    A custom attention layer that accepts external keys and values, and is aware of the
    number of embeddings each key represents.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        external_keys: torch.Tensor | None = None,
        external_values: torch.Tensor | None = None,
        num_embeddings_per_key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        
        bsz, q_len, _ = hidden_states.size()

        # Standard query projection
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Use externally provided keys and values
        key_states = external_keys
        value_states = external_values

        # Standard attention calculation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # --- Cluster-Aware Attention Logic ---
        if num_embeddings_per_key is not None:
            # Scale the attention weights by the number of embeddings per key
            attn_weights = attn_weights * num_embeddings_per_key.view(1, 1, 1, -1)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class VectorInjectedModel(LlamaForCausalLM):
    """
    A custom model that replaces the standard attention layers with VectorInjectedAttention.
    """
    def __init__(self, config):
        super().__init__(config)
        # Replace all attention layers with our custom one
        for i, layer in enumerate(self.model.layers):
            layer.self_attn = VectorInjectedAttention(config=config, layer_idx=i)