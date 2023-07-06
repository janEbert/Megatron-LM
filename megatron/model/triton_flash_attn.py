# Adapted from
# https://github.com/mosaicml/llm-foundry/blob/ef350d9e64d13cb1db35ab7941bf9039b1b499fd/llmfoundry/models/layers/attention.py

import warnings

try:
    from einops import rearrange
except ImportError:
    rearrange = None
try:
    from flash_attn.flash_attn_triton import flash_attn_func
except Exception:
    flash_attn_func = None
import torch


def _reset_is_causal(num_query_tokens: int, num_key_tokens: int,
                     original_is_causal: bool):
    # disable causal when it is not needed
    # necessary for flash & triton for generation with kv_cache
    if original_is_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError(
                'MPT does not support query and key with different number of '
                'tokens, unless number of query tokens is 1.'
            )
        else:
            return False
    return original_is_causal


def check_valid_inputs(*tensors, valid_dtypes=[torch.float16, torch.bfloat16]):
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f'{tensor.dtype=} must be in {valid_dtypes=}.')
        if not tensor.is_cuda:
            raise TypeError(
                f'Inputs must be cuda tensors ({tensor.is_cuda=}).')


def triton_flash_attn_fn(
    query,
    key,
    value,
    n_heads,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
):
    """
    Megatron-LM query.shape: (seq_len, batch, head / tp, dim / head)
    required query.shape: (batch, seq_len, head * dim)
    """
    assert flash_attn_func is not None, (
        'Requirements for Triton FlashAttention not installed. '
        'Please execute `pip install flash-attn`.'
    )
    assert rearrange is not None, \
        'Please install `einops` first, e.g., with `pip install einops`.'

    check_valid_inputs(query, key, value)

    if attn_bias is not None:
        # clamp to 0 necessary for torch 2.0 compile()
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]

    if dropout_p:
        raise NotImplementedError(
            'Dropout not implemented for Triton FlashAttention.')

    if key_padding_mask is not None:
        warnings.warn(
            'Propagating key_padding_mask to the attention module '
            'and applying it within the attention module can cause '
            'unnecessary computation/memory usage. Consider integrating '
            'into attn_bias once and passing that to each attention '
            'module instead.'
        )
        b_size, s_k = key_padding_mask.shape[:2]

        if attn_bias is None:
            attn_bias = query.new_zeros(b_size, 1, 1, s_k)

        attn_bias = attn_bias.masked_fill(
            ~key_padding_mask.view((b_size, 1, 1, s_k)),
            torch.finfo(query.dtype).min)

    if query.dim() == 3:
        query = rearrange(query, 'b s (h d) -> b s h d', h=n_heads)
        key = rearrange(key, 'b s (h d) -> b s h d',
                        h=n_heads)
        value = rearrange(value,
                          'b s (h d) -> b s h d',
                          h=n_heads)

    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    attn_output = flash_attn_func(query, key, value, attn_bias,
                                  reset_is_causal, softmax_scale)

    output = attn_output.view(*attn_output.shape[:2], -1)

    return output
