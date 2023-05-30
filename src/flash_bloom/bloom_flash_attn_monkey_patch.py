from typing import List, Optional, Tuple

import torch
from torch import nn

import transformers
from transformers.models.bloom.modeling_bloom import dropout_add
from flash_attn.modules.mha import FlashSelfAttention

from einops import rearrange


def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
):
    fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, q_length, _, _ = query_layer.shape

    query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
    value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
    if layer_past is not None:
        past_key, past_value = layer_past
        # concatenate along seq_length dimension:
        #  - key: [batch_size * self.num_heads, head_dim, kv_length]
        #  - value: [batch_size * self.num_heads, kv_length, head_dim]
        key_layer = torch.cat((past_key, key_layer), dim=2)
        value_layer = torch.cat((past_value, value_layer), dim=1)

    _, _, kv_length = key_layer.shape

    assert not output_attentions, "output_attentions is not supported"
    assert not use_cache, "use_cache is not supported"

    if use_cache is True:
        present = (key_layer, value_layer)
    else:
        present = None

    # [batch_size * num_heads, q_length, kv_length]
    # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
    # matmul_result = alibi.baddbmm(
    #     batch1=query_layer,
    #     batch2=key_layer,
    #     beta=self.beta,
    #     alpha=self.inv_norm_factor,
    # )
    #
    # # change view to [batch_size, num_heads, q_length, kv_length]
    # attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)
    #
    # # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
    # input_dtype = attention_scores.dtype
    # # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
    # if input_dtype == torch.float16:
    #     attention_scores = attention_scores.to(torch.float)
    # attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
    # attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)
    #
    # # [batch_size, num_heads, q_length, kv_length]
    # attention_probs = self.attention_dropout(attention_probs)
    #
    # if head_mask is not None:
    #     attention_probs = attention_probs * head_mask
    #
    # # change view [batch_size x num_heads, q_length, kv_length]
    # attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)
    #
    # # matmul: [batch_size * num_heads, q_length, head_dim]
    # context_layer = torch.bmm(attention_probs_reshaped, value_layer)
    #
    # # change view [batch_size, num_heads, q_length, head_dim]
    # context_layer = self._merge_heads(context_layer)
    #
    # # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
    # if self.pretraining_tp > 1 and self.slow_but_exact:
    #     slices = self.hidden_size / self.pretraining_tp
    #     output_tensor = torch.zeros_like(context_layer)
    #     for i in range(self.pretraining_tp):
    #         output_tensor = output_tensor + F.linear(
    #             context_layer[:, :, int(i * slices): int((i + 1) * slices)],
    #             self.dense.weight[:, int(i * slices): int((i + 1) * slices)],
    #         )
    # else:
    #     output_tensor = self.dense(context_layer)
    #
    # output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

    # ===================process to flash accept form ================#
    reshaped_query_layer = query_layer.reshape(batch_size, self.num_heads, query_layer.shape[1],
                                               query_layer.shape[2]).permute(0, 2, 1, 3)
    reshaped_key_layer = key_layer.reshape(batch_size, self.num_heads, key_layer.shape[1],
                                           key_layer.shape[2]).permute(0, 3, 1, 2)
    reshaped_value_layer = value_layer.reshape(batch_size, self.num_heads, value_layer.shape[1],
                                               value_layer.shape[2]).permute(0, 2, 1, 3)
    offset_key_layer = self.inv_norm_factor * reshaped_key_layer + self.beta * (
            torch.linalg.pinv(reshaped_query_layer.permute(0, 2, 1, 3).float()) * alibi.view(batch_size,
                                                                                             alibi.shape[
                                                                                                 0] // batch_size,
                                                                                             alibi.shape[1],
                                                                                             alibi.shape[
                                                                                                 2])).permute(0, 3,
                                                                                                              1,
                                                                                                              2).half()
    qkv = torch.concat(
        [reshaped_query_layer.unsqueeze(2), offset_key_layer.unsqueeze(2), reshaped_value_layer.unsqueeze(2)],
        dim=2).half()
    if not hasattr(self, 'flash_self_attention'):
        self.flash_self_attention = FlashSelfAttention(causal=True, softmax_scale=1)
    context_layer = self.flash_self_attention(qkv)
    context_layer = torch.flatten(context_layer, start_dim=2)

    # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
    if self.pretraining_tp > 1 and self.slow_but_exact:
        slices = self.hidden_size / self.pretraining_tp
        output_tensor = torch.zeros_like(context_layer)
        for i in range(self.pretraining_tp):
            output_tensor = output_tensor + torch.nn.functional.linear(
                context_layer[:, :, int(i * slices): int((i + 1) * slices)],
                self.dense.weight[:, int(i * slices): int((i + 1) * slices)],
            )
    else:
        output_tensor = self.dense(context_layer)

    output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

    outputs = (output_tensor, present)

    return outputs


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                    inputs_embeds, past_key_values_length):
    # [bsz, seq_len]
    return attention_mask


def replace_bloom_attn_with_flash_attn():
    # transformers.models.bloom.modeling_bloom.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    transformers.models.bloom.modeling_bloom.BloomAttention.forward = forward
