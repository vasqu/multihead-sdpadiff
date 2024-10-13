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

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadSdpaDiff4(nn.Module):
    """
    DiffAttn implemented with SDPA Attention, using one attention pass based on
    @ https://github.com/microsoft/unilm/pull/1633#issuecomment-2407941437

    Credit to [MarktHart](https://github.com/MarktHart)
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

        q = q.transpose(1, 2).contiguous()
        k = repeat_kv(k, n_rep=self.n_rep).transpose(1, 2).contiguous()
        v = repeat_kv(v, n_rep=self.n_rep*2).transpose(1, 2).contiguous()

        if attn_mask is not None:
            attn_mask = attn_mask[:, :, :, : src_len.shape[-2]]
        is_causal = True if attn_mask is None and tgt_len > 1 else False

        # utilizes cyclic pattern of head calculation
        attn_weights = sdpa_attn_function(query=q, key=k, value=v, attn_mask=attn_mask, is_causal=is_causal)
        every_other_mask = torch.arange(attn_weights.size(1)) % 2 == 0
        attn1 = attn_weights[:, every_other_mask, :, :]
        attn2 = attn_weights[:, ~every_other_mask, :, :]

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
