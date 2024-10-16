import math

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention as sdpa_attn_function

from .utils.rotary import apply_rotary_emb
from .utils.rms_norm import RMSNorm


def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bsz, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)
    )

def repeat_and_stack_attn_heads(qkv, strategy='repeat'):
    bsz, seq_len, num_heads, _, head_dim = qkv.shape

    # repeat along to duplicate heads and split into each separate one
    qkv = qkv.repeat(1, 1, 1, 2, 1).contiguous()
    qkv = qkv.view(bsz, seq_len, num_heads, 4, head_dim)
    qkv_11, qkv_12, qkv_21, qkv_22 = torch.unbind(qkv, dim=-2)

    # head pattern based repetition (2 separate heads, repeated twice each)
    # repeat == 1,2,1,2
    if strategy == 'repeat':
        return (
            torch.cat(
                (qkv_11, qkv_12, qkv_21, qkv_22),
                dim=-2)
            .transpose(1, 2)
            .contiguous()
        )
    # interleave == 1,1,2,2
    elif strategy == 'interleave':
        return (
            torch.cat(
                (qkv_11, qkv_21, qkv_12, qkv_22),
                dim=-2)
            .transpose(1, 2)
            .contiguous()
        )
    else:
        raise ValueError(f'Requested strategy {strategy} is not supported!')

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadSdpaDiff3(nn.Module):
    """
    DiffAttn implemented with SDPA Attention, using one attention pass but more concatenating and chunking/unbinding
    """
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        num_kv_heads=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # Defaulting to half the size
        self.num_heads = num_heads // 2
        self.num_kv_heads = num_kv_heads // 2 if num_kv_heads is not None else num_heads // 2
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)

    def forward(
        self,
        x,
        rel_pos=None,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads*2, self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads*2, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, self.head_dim*2)

        # optional RoPE
        if rel_pos:
            q = apply_rotary_emb(q, *rel_pos, interleaved=True)
            k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_heads, 2, self.head_dim)
        v = v.reshape(bsz, src_len, self.num_heads, 2, self.head_dim)

        # [bsz, num_heads*4, seq_len, head_dim]
        q = repeat_and_stack_attn_heads(q, strategy='interleave')
        k = repeat_and_stack_attn_heads(k, strategy='interleave')
        v = repeat_and_stack_attn_heads(v, strategy='repeat')

        if attn_mask is not None:
            attn_mask = attn_mask[:, :, :, : src_len.shape[-2]]
        is_causal = True if attn_mask is None and tgt_len > 1 else False

        # pass into one attn with appropriate heads and reconstruct correct diff attn scores
        attn1122 = sdpa_attn_function(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        attn11, attn12, attn21, attn22 = attn1122.chunk(4, dim=1)
        attn1 = torch.cat((attn11, attn12), dim=-1)
        attn2 = torch.cat((attn21, attn22), dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(bsz, tgt_len, -1)

        attn = self.out_proj(attn)
        return attn
