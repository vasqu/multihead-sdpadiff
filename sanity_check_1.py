import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention as SDPA


def repeat_and_stack_attn_heads(qkv, strategy='repeat'):
    bsz, seq_len, num_heads, _, head_dim = qkv.shape

    qkv = qkv.repeat(1, 1, 1, 2, 1)
    qkv = qkv.view(bsz, seq_len, num_heads, 4, head_dim)
    qkv_11, qkv_12, qkv_21, qkv_22 = torch.unbind(qkv, dim=-2)

    if strategy == 'repeat':
        return (
            torch.cat(
                (qkv_11, qkv_12, qkv_21, qkv_22),
                dim=-2)
            .transpose(1, 2)
            .contiguous()
        )
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


bsz = 2
seq_len = 3
embed_dim = 768
head_dim = 64
num_heads = 6

q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
out_proj = nn.Linear(embed_dim, embed_dim, bias=False)


x = torch.randn(size=(bsz, seq_len, embed_dim))

q = q_proj(x)
k = k_proj(x)
v = v_proj(x)

q = q.view(bsz, seq_len, num_heads, 2, head_dim)
k = k.view(bsz, seq_len, num_heads, 2, head_dim)
v = v.view(bsz, seq_len, num_heads, 2, head_dim)

q1, q2 = q[:, :, :, 0].transpose(1, 2), q[:, :, :, 1].transpose(1, 2)
k1, k2 = k[:, :, :, 0].transpose(1, 2), k[:, :, :, 1].transpose(1, 2)
v1, v2 = v[:, :, :, 0].transpose(1, 2), v[:, :, :, 1].transpose(1, 2)

attn11 = SDPA(query=q1, key=k1, value=v1, is_causal=True)
attn12 = SDPA(query=q1, key=k1, value=v2, is_causal=True)
attn1 = torch.cat([attn11, attn12], dim=-1)

attn21 = SDPA(query=q2, key=k2, value=v1, is_causal=True)
attn22 = SDPA(query=q2, key=k2, value=v2, is_causal=True)
attn2 = torch.cat([attn21, attn22], dim=-1)


q = repeat_and_stack_attn_heads(q, strategy='interleave')
k = repeat_and_stack_attn_heads(k, strategy='interleave')
v = repeat_and_stack_attn_heads(v, strategy='repeat')

res = SDPA(query=q, key=k, value=v, is_causal=True)
attn11_, attn12_, attn21_, attn22_ = res.chunk(4, dim=1)
attn1_ = torch.cat((attn11_, attn12_), dim=-1)
attn2_ = torch.cat((attn21_, attn22_), dim=-1)

print(f'{torch.allclose(attn11, attn11_)}')
print(f'{torch.allclose(attn12, attn12_)}')
print(f'{torch.allclose(attn21, attn21_)}')
print(f'{torch.allclose(attn22, attn22_)}')

print(f'{torch.allclose(attn1, attn1_)}')
print(f'{torch.allclose(attn2, attn2_)}')
