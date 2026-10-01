from torch import nn
import torch
import math
import warnings
from .configuration_penguinvl_encoder import PenguinVLVisionEncoderConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3Attention, rotate_half, Qwen3DecoderLayer
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging, is_flash_attn_greater_or_equal_2_10, is_flash_attn_2_available
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from torch.nn.init import _calculate_fan_in_and_fan_out
import torch.nn.functional as F
if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    from flash_attn import flash_attn_varlen_func

logger = logging.get_logger(__name__)

class PenguinVLVisionEncoderEmbeddings(nn.Module):

    def __init__(self, config: PenguinVLVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1, self.config.num_channels, self.patch_size, self.patch_size
        )
        patch_embeds = self.patch_embedding(hidden_states)
        embeddings = patch_embeds.view(-1, self.embed_dim)

        return embeddings
    

# Adapted from Qwen2VLRotaryEmbedding in transformers/models/qwen2/modeling_qwen2.py
class VisualRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2VLRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(2, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (2, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    rope_section = [cos.shape[-1] // 2, cos.shape[-1] // 2]
    cos = torch.cat([m[i % 2] for i, m in enumerate(cos.split(rope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 2] for i, m in enumerate(sin.split(rope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PenguinVLAttention(Qwen3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # This is before the transpose
        seq_len = query_states.shape[2]

        # Restore the attention dtype if normalization promoted hidden states to FP32.
        target_dtype = None
        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = next(layer for layer in self.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

        # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
        kwargs.pop("is_causal", None)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2).squeeze(0)
        key_states = key_states.transpose(1, 2).squeeze(0)
        value_states = value_states.transpose(1, 2).squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            causal=self.is_causal
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    

class PenguinVLDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: PenguinVLVisionEncoderConfig, layer_idx: int):
        super(PenguinVLDecoderLayer, self).__init__(config, layer_idx)
        self.self_attn = PenguinVLAttention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        cu_seqlens: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
    
    
class PenguinVLVisionEncoderFromQwen3Model(Qwen3Model):
    def __init__(self, config: PenguinVLVisionEncoderConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [PenguinVLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = VisualRotaryEmbedding(config=config)
        del self.embed_tokens

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: PenguinVLVisionEncoderConfig,
        past_key_values: Cache,
    ):
        """
        Override the original causal mask method to create full attention mask instead.
        Creates a full attention 4D mask of shape `(batch_size, 1, query_length, key_value_length)` 
        from a 2D mask of shape `(batch_size, key_value_length)`.
        
        For vision encoding, we want full attention between all patches, not causal attention.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            full_attention_mask = attention_mask
        else:
            # Create full attention mask (all zeros, meaning attend to all positions)
            # We only mask based on the provided attention_mask for padding
            if attention_mask is not None:
                # Use the provided attention_mask to handle padding
                min_dtype = torch.finfo(dtype).min
                full_attention_mask = torch.zeros(
                    (sequence_length, target_length), dtype=dtype, device=device
                )
                # Expand to 4D
                full_attention_mask = full_attention_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
                
                # Apply padding mask if provided
                full_attention_mask = full_attention_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = attention_mask[:, None, None, :] == 0
                full_attention_mask[:, :, :, :mask_length] = full_attention_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            else:
                # No attention mask provided, create all-zeros mask (full attention)
                full_attention_mask = torch.zeros(
                    (batch_size, 1, sequence_length, target_length), dtype=dtype, device=device
                )
        return full_attention_mask

    def get_rope_index(self, grid_sizes, merge_sizes, position_ids):
        position_ids = position_ids.contiguous()
        batch_size = grid_sizes.shape[0]

        # Vision Part: Generate 2D position indices for vision tokens
        vision_pos_ids = []
        for (t, h, w), merge_size in zip(grid_sizes, merge_sizes):
            # Generate height position indices
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w).to(position_ids.device)
            hpos_ids = hpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            # Generate width position indices  
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1).to(position_ids.device)
            wpos_ids = wpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            
            # Stack height and width to create 2D positions
            vision_pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        num_start_idx = 0
        for batch_idx in range(batch_size):
            pos_len = vision_pos_ids[batch_idx].shape[0]
            position_ids[:, 0, num_start_idx: num_start_idx+pos_len] = vision_pos_ids[batch_idx].permute(1, 0)
            num_start_idx += pos_len
        
        return position_ids


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        grid_sizes: Optional[torch.Tensor] = None,
        merge_sizes: Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `2` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(2, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(2, position_ids.shape[0], -1)
        position_ids = self.get_rope_index(grid_sizes, merge_sizes, position_ids)

        causal_mask = None

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Calculate cumulative sequence lengths for the grid sizes
        cu_seqlens = torch.repeat_interleave(grid_sizes[:, 1] * grid_sizes[:, 2], grid_sizes[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class PenguinVLVisionEncoderModel(PreTrainedModel):

    config_class = PenguinVLVisionEncoderConfig
    base_model_prefix = "penguinvl_vision_encoder"
    main_input_name = "pixel_values"
    _no_split_modules = [
        "PenguinVLVisionEncoderEmbeddings",
    ]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: PenguinVLVisionEncoderConfig):
        super().__init__(config=config)
        self.embeddings = PenguinVLVisionEncoderEmbeddings(config)
        self.encoder = PenguinVLVisionEncoderFromQwen3Model(config)

        self.post_init()


    def forward(self, pixel_values, grid_sizes, merge_sizes=None) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        encoder_output = self.encoder(
            inputs_embeds=hidden_states[None, ...],
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
            output_hidden_states=True,
        )
        hidden_states = encoder_output.hidden_states
        hidden_states = hidden_states[-1].squeeze(0)

        hidden_states_chunks = hidden_states.split(grid_sizes.prod(dim=1).tolist(), dim=0)
        outputs = []

        for hidden_states, grid_size, merge_size in zip(hidden_states_chunks, grid_sizes, merge_sizes):
            # NOTE: previous implementation, which supports downsampling with any factor
            c = hidden_states.shape[-1]
            hidden_states = hidden_states.view(
                grid_size[0], grid_size[1] // merge_size, grid_size[2] // merge_size, merge_size, merge_size,  c
            ).permute(0, 1, 3, 2, 4, 5)
            hidden_states = hidden_states.reshape(
                grid_size[0], grid_size[1], grid_size[2], c
            ).permute(0, 3, 1, 2)
            hidden_states = torch.nn.functional.interpolate(
                hidden_states,
                size=(grid_size[1] // merge_size, grid_size[2] // merge_size),
                mode='bilinear'
            )
            hidden_states = hidden_states.permute(0, 2, 3, 1).view(-1, c)

            # NOTE: simplified implementation, which only supports downsampling with integer factor
            # NOTE: this implementation is mathematically equivalent to the previous one when merge_size is 1 or 2 but may cause slightly different results
            # hidden_states = hidden_states.view(-1, merge_size * merge_size, hidden_states.size(-1))
            # hidden_states = hidden_states.mean(dim=1)

            outputs.append(hidden_states)
        return torch.cat(outputs, dim=0)
