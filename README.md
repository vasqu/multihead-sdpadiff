# Differential Transformer with PyTorch Scaled Dot Product Attention

## Introduction
A set of implementations for the Differential Transformer paper [[1]](#citation) using PyTorch's
[Scaled Dot Product Attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) instead
of the provided implementations [over here](https://github.com/microsoft/unilm/tree/master/Diff-Transformer): 
- Basic manual PyTorch
- Flash Attention 2 
  - Custom kernel to handle differing `head_dim` more efficiently
  - Original kernel that is more optimized on same `head_dim`

This implementation has four variations as of now:
- Following the original Flash Attention 2 implementation more closely
- Following the custom Flash Attention 2 implementation more closely
- One forward pass to the attention calculations (transferable to original Flash Attention 2 implementation)
- One forward pass to the attention calculations based on [[2]](#citation) (utilizing SDPA different `head_dim` capability)

Note:
- RoPE is optional as I only cared about equivalency first and foremost
- Needs external proper handling of RoPE and Attention Masks
- It really needs benchmarks to see what is working better especially regarding both 
one pass versions 
  - Same `head_dim`, more `num_heads` but concatenating and chunking/unbinding
  - Different `head_dim`, less `num_heads` but possibly less utilization on Flash Attention 2


## Usage
```python
import torch

from multihead_sdpadiff_1 import MultiheadSdpaDiff1  # multiple attn passes
from multihead_sdpadiff_2 import MultiheadSdpaDiff2  # two attn passes
from multihead_sdpadiff_3 import MultiheadSdpaDiff3  # one attn pass (v1)
from multihead_sdpadiff_4 import MultiheadSdpaDiff4  # one attn pass (v2)

# some shape values
bsz = 2
seq_len = 3
depth = 12
embed_dim = 768
num_heads = 12  # this will be set to half as we double them for the diff 

# random input
x = torch.randn(size=(bsz, seq_len, embed_dim))

# choose an implementation
#sdpa_mha_diff = MultiheadSdpaDiff1(embed_dim, depth, num_heads, num_heads)
#sdpa_mha_diff = MultiheadSdpaDiff2(embed_dim, depth, num_heads, num_heads)
#sdpa_mha_diff = MultiheadSdpaDiff3(embed_dim, depth, num_heads, num_heads)
sdpa_mha_diff = MultiheadSdpaDiff4(embed_dim, depth, num_heads, num_heads)

# pass and check
res = sdpa_mha_diff(x)
assert res.shape == x.shape
```


## TODOs
- [ ] Make it a package structure
- [ ] Benchmark the speed/memory between the implementations
- [ ] Transformer style RoPE + Attn Mask


## Citation

```bibtex
[1]
@misc{ye2024differentialtransformer,
      title={Differential Transformer}, 
      author={Tianzhu Ye and Li Dong and Yuqing Xia and Yutao Sun and Yi Zhu and Gao Huang and Furu Wei},
      year={2024},
      eprint={2410.05258},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.05258}, 
}
```

[2] Thanks for [MarktHart](https://github.com/MarktHart) for providing another [version](https://github.com/microsoft/unilm/pull/1633#issuecomment-2407941437) which might be the most optimized one
