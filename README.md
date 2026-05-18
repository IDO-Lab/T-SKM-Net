# T-SKM-Net: Trainable Neural Network Framework for Linear Constraint Satisfaction via Sampling Kaczmarz-Motzkin Method

[![AAAI 2026](https://img.shields.io/badge/AAAI--2026-Published-blue.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/38459)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10461-b31b1b.svg)](https://arxiv.org/abs/2512.10461)

This is the official repository for **T-SKM-Net**, a trainable neural network framework for **linear constraint satisfaction** via the Sampling Kaczmarz-Motzkin method.

- Paper (AAAI): https://ojs.aaai.org/index.php/AAAI/article/view/38459  
- Paper (arXiv, with appendix): https://arxiv.org/abs/2512.10461  
- PDF in this repository (with appendix): `paper.pdf`

**Status:** Published in **AAAI 2026**.

---

## Installation

This package does not pin a PyTorch wheel in `pyproject.toml`, because the
appropriate CPU/CUDA build depends on your machine. Install PyTorch first, then
install this package:

```bash
uv pip install --torch-backend=auto torch
uv pip install .
```

---

## Minimal Usage

`t_skm_project` maps a batch of initial points `x0` to the mixed linear
constraint set

```text
A x <= b,   C x = d.
```

```python
import torch
from tskm_net import t_skm_project

A = torch.cat([torch.eye(3), -torch.eye(3)], dim=0)  # A x <= b
b = torch.tensor([1., 1., 1., 0., 0., 0.])
C = torch.ones(1, 3)                                # C x = d
d = torch.ones(4, 1)
x0 = torch.randn(4, 3)

x = t_skm_project(
    A, b, C, d, x0,
    beta=6,
    delta=1.0,
    variant="heavy_ball_skm",
    variant_args={"momentum": 0.25},
)
```

Supported variants are `"skm"`, `"heavy_ball_skm"`, `"nesterov_skm"`, and
`"gskm"`. Shapes are strict: `A` is `(p, n)`, `b` is `(p,)`, `C` is `(q, n)`,
`d` is `(B, q)`, and `x0` is `(B, n)`. `A`, `b`, and `C` are shared across
the batch; `d` and `x0` are batched.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhuTSKMNetTrainableNeural2026,
  title   = {T-SKM-Net: Trainable Neural Network Framework for Linear Constraint Satisfaction via Sampling Kaczmarz-Motzkin Method},
  author  = {Zhu, Haoyu and Zhang, Yao and Ren, Jiashen and Hou, Qingchun},
  year    = {2026},
  month   = mar,
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {40},
  number  = {17},
  pages   = {14431--14439},
  doi     = {10.1609/aaai.v40i17.38459},
  url     = {https://doi.org/10.1609/aaai.v40i17.38459}
}
