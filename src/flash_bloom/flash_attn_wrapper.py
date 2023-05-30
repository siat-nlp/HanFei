import torch
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from typing import Optional

from flash_attn.modules.mha import FlashSelfAttention
from transformers.models.bloom.modeling_bloom import dropout_add
from typing import Optional, Tuple


class FlashAttentionWrapper(torch.nn.Module):
    def __init__(self, attention, max_seqlen=8190):
        super().__init__()
        self.attention = attention
        self.max_seqlen = max_seqlen
        self.flash_self_attention = FlashSelfAttention(causal=True, softmax_scale=1)
        self.dropout_p = 0.0

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.attention.q_proj(hidden_states) * self.attention.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self.attention._shape(self.attention.k_proj(key_value_states), -1, bsz)
            value_states = self.attention._shape(self.attention.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self.attention._shape(self.attention.k_proj(hidden_states), -1, bsz)
            value_states = self.attention._shape(self.attention.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self.attention._shape(self.attention.k_proj(hidden_states), -1, bsz)
            value_states = self.attention._shape(self.attention.v_proj(hidden_states), -1, bsz)

        if self.attention.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz, self.attention.num_heads, -1, self.attention.head_dim)
        key_states = key_states.view(*proj_shape).permute(0, 2, 1, 3)
        query_states = self.attention._shape(query_states, tgt_len, bsz).permute(0, 2, 1, 3)
        value_states = value_states.view(*proj_shape).permute(0, 2, 1, 3)
        qkv = torch.concat([query_states.unsqueeze(2), key_states.unsqueeze(2), value_states.unsqueeze(2)],
                           dim=2).half()
        attn_output = self.flash_self_attention(qkv)
        attn_weights_reshaped = None

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.attention.embed_dim)

        attn_output = self.attention.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class FlashAttentionWrapperWithRotary(torch.nn.Module):
    def __init__(self, attention, max_seqlen=8192):
        super().__init__()
        self.attention = attention
        self.max_seqlen = max_seqlen
        self.flash_self_attention = FlashSelfAttention(causal=True, softmax_scale=1 / self.attention.norm_factor)
        self.dropout_p = 0.0

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                layer_past=None,
                use_cache=False,
                output_attentions=False):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.attention.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.attention.num_attention_heads, 3 * self.attention.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.attention.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.attention.head_size: 2 * self.attention.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.attention.head_size:].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.attention.rotary_ndims]
        query_pass = query[..., self.attention.rotary_ndims:]
        key_rot = key[..., : self.attention.rotary_ndims]
        key_pass = key[..., self.attention.rotary_ndims:]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        cos, sin = self.attention.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=offset)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        # attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        qkv = torch.concat([query.unsqueeze(2), key.unsqueeze(2), value.unsqueeze(2)], dim=2).permute(0, 3, 2, 1,
                                                                                                      4).half()
        attn_output = self.flash_self_attention(qkv)
        attn_weights = None

        # Reshape outputs
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1),
                                       self.attention.num_attention_heads * self.attention.head_size)
        attn_output = self.attention.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class FlashAttentionWrapperWithAlibi(torch.nn.Module):
    def __init__(self, attention, max_seqlen=8192):
        super().__init__()
        self.attention = attention
        self.max_seqlen = max_seqlen
        self.flash_self_attention = FlashSelfAttention(causal=True, softmax_scale=None)
        self.dropout_p = 0.0

    def forward(self,
                hidden_states: torch.Tensor,
                residual: torch.Tensor,
                alibi: torch.Tensor,
                attention_mask: torch.Tensor,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                head_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                output_attentions: bool = False,
                ):
        fused_qkv = self.attention.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self.attention._split_heads(fused_qkv)
        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.attention.num_heads, q_length,
                                                          self.attention.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.attention.num_heads,
                                                          self.attention.head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.attention.num_heads, q_length,
                                                          self.attention.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        reshaped_query_layer = query_layer.reshape(batch_size, self.attention.num_heads, query_layer.shape[1],
                                                   query_layer.shape[2]).permute(0, 2, 1, 3)
        reshaped_key_layer = key_layer.reshape(batch_size, self.attention.num_heads, key_layer.shape[1],
                                               key_layer.shape[2]).permute(0, 3, 1, 2)
        reshaped_value_layer = value_layer.reshape(batch_size, self.attention.num_heads, value_layer.shape[1],
                                                   value_layer.shape[2]).permute(0, 2, 1, 3)
        offset_key_layer = self.attention.inv_norm_factor * reshaped_key_layer + self.attention.beta * (
                torch.linalg.pinv(reshaped_query_layer.permute(0, 2, 1, 3).float()) * alibi.view(batch_size,
                                                                                                 alibi.shape[
                                                                                                     0] // batch_size,
                                                                                                 alibi.shape[1],
                                                                                                 alibi.shape[
                                                                                                     2])).permute(0,
                                                                                                                  3,
                                                                                                                  1,
                                                                                                                  2).half()
        qkv = torch.concat(
            [reshaped_query_layer.unsqueeze(2), offset_key_layer.unsqueeze(2), reshaped_value_layer.unsqueeze(2)],
            dim=2).half()
        context_layer = self.flash_self_attention(qkv)
        context_layer = torch.flatten(context_layer, start_dim=2)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.attention.pretraining_tp > 1 and self.attention.slow_but_exact:
            slices = self.attention.hidden_size / self.attention.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.attention.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices): int((i + 1) * slices)],
                    self.attention.dense.weight[:, int(i * slices): int((i + 1) * slices)],
                )
        else:
            output_tensor = self.attention.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.attention.hidden_dropout, self.attention.training)

        outputs = (output_tensor, present)
        return outputs
