import torch
import torch.nn as nn

from multihead_sdpadiff_1 import MultiheadSdpaDiff1
from multihead_sdpadiff_2 import MultiheadSdpaDiff2


bsz = 2
seq_len = 3
embed_dim = 768
num_heads = 12

q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
out_proj = nn.Linear(embed_dim, embed_dim, bias=False)


x = torch.randn(size=(bsz, seq_len, embed_dim))

mha_1 = MultiheadSdpaDiff1(embed_dim, 12, num_heads, num_heads)
mha_2 = MultiheadSdpaDiff2(embed_dim, 12, num_heads, num_heads)

mha_2.q_proj = mha_1.q_proj
mha_2.k_proj = mha_1.k_proj
mha_2.v_proj = mha_1.v_proj
mha_2.out_proj = mha_1.out_proj
mha_2.lambda_init = mha_1.lambda_init
mha_2.lambda_q1 = mha_1.lambda_q1
mha_2.lambda_k1 = mha_1.lambda_k1
mha_2.lambda_q2 = mha_1.lambda_q2
mha_2.lambda_k2 = mha_1.lambda_k2
mha_2.subln = mha_1.subln

res_1 = mha_1(x)
res_2 = mha_2(x)

print(torch.allclose(res_1, res_2))
