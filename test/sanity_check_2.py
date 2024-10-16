import torch

from multihead_sdpadiff import (
    MultiheadSdpaDiff1,
    MultiheadSdpaDiff2,
    MultiheadSdpaDiff3,
    MultiheadSdpaDiff4,
)


def clone_params(mha_1, mha_2):
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


# some shape values
bsz = 2
seq_len = 3
depth = 12
embed_dim = 768
num_heads = 12

# random input
x = torch.randn(size=(bsz, seq_len, embed_dim))

# using all implementations and comparing them with each other
mha_1 = MultiheadSdpaDiff1(embed_dim, depth, num_heads, num_heads // 2)
mha_2 = MultiheadSdpaDiff2(embed_dim, depth, num_heads, num_heads // 2)
mha_3 = MultiheadSdpaDiff3(embed_dim, depth, num_heads, num_heads // 2)
mha_4 = MultiheadSdpaDiff4(embed_dim, depth, num_heads, num_heads // 2)

clone_params(mha_1, mha_2)
clone_params(mha_1, mha_3)
clone_params(mha_1, mha_4)

res_1 = mha_1(x)
res_2 = mha_2(x)
res_3 = mha_3(x)
res_4 = mha_4(x)

print(torch.allclose(res_1, res_2, atol=1e-5))
print(torch.allclose(res_1, res_3, atol=1e-5))
print(torch.allclose(res_1, res_4, atol=1e-5))
