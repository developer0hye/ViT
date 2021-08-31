# ViT

PyTorch implementation of [Vit](https://openreview.net/pdf?id=YicbFdNTTy)

All operations used for implementation are supported on ONNX and TensorRT.

My works already merged into https://github.com/lucidrains/vit-pytorch/pull/151.

## Usage

```python
import torch
from vit import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
```

## Acknowledgement

This project is heavily based on [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch). Thanks for their awesome works.
