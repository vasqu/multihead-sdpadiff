# Differential Transformer with PyTorch Scaled Dot Product Attention

## Introduction
Another two implementations for the Differential Transformer paper [[1]](#citation) using PyTorch's
[Scaled Dot Product Attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) instead
of the provided implementations [over here](https://github.com/microsoft/unilm/tree/master/Diff-Transformer): 
- Flash Attention 2 
  - Custom kernel to handle differing `head_dim` more efficiently
  - Original kernel that is more optimized on same `head_dim`
- Basic manual PyTorch

This implementation has two variations as I explored:
- One forward pass to the attention calculations (transferable to Flash Attention 2 implementation)
- Following the original Flash Attention 2 implementation more closely

Note:
- RoPE is optional as I only cared about equivalency first and foremost
- Needs external proper handling of RoPE and Attention Masks 


## Usage
```python
import torch

from multihead_sdpadiff_1 import MultiheadSdpaDiff1
from multihead_sdpadiff_2 import MultiheadSdpaDiff2

# some shape values
bsz = 2
seq_len = 3
embed_dim = 768
num_heads = 12  # this will be set to half as we double them for the diff 

# random input
x = torch.randn(size=(bsz, seq_len, embed_dim))

# choose an implementation
#sdpa_mha_diff = MultiheadSdpaDiff1(embed_dim, 12, num_heads, num_heads)
sdpa_mha_diff = MultiheadSdpaDiff2(embed_dim, 12, num_heads, num_heads)

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
