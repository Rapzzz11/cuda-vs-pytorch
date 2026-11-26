# Custom CUDA Kernels vs. PyTorch for Neural Network Layers

Performance analysis of hand-written CUDA kernels against PyTorch's built-in
(cuBLAS / cuDNN-backed) implementations of common neural network layers.

## Layers implemented
| Layer | Custom kernel approach |
|---|---|
| ReLU | grid-stride elementwise kernel |
| MatMul v1 | 16x16 shared-memory tiling |
| MatMul v2 | 64x64 tiles with 4x4 register blocking per thread |
| Softmax | one block per row, shared-memory reductions |
| LayerNorm | one block per row, mean/variance reductions |
| Conv2d | direct convolution, one thread per output pixel |

Every kernel is checked against PyTorch with `torch.allclose` before it is timed.
Benchmarks use FP32 on both sides (TF32 disabled), CUDA-event timing and warm-up runs.

## Run it (no local GPU needed)
1. Click the badge below to open the notebook in Google Colab.
2. `Runtime -> Change runtime type -> T4 GPU`
3. `Runtime -> Run all` (first compile takes 1-2 minutes)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahul123/cuda-vs-pytorch/blob/main/notebooks/Review3_CUDA_vs_PyTorch.ipynb)

Outputs (CSV table and graphs) are written to `results/`.

## Run it locally
Requires an NVIDIA GPU, the CUDA toolkit (`nvcc`) matching your PyTorch build, and Python 3.9+.
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Review3_CUDA_vs_PyTorch.ipynb
```

## Repository layout
```
notebooks/   Review 2 and Review 3 notebooks (self-contained, compile kernels inline)
csrc/        Standalone copy of the CUDA/C++ source used by the Review 3 notebook
docs/        Review presentations
results/     Generated benchmark CSV and plots (created when you run the notebook)
```

## Troubleshooting
If you see `ImportError: ... cannot open shared object file`, the real error is a failed
compile earlier in the log. Run `!rm -rf /root/.cache/torch_extensions`, restart the runtime,
and read the first `error:` line in the build output.

## Limitations
Forward pass only (no gradients), FP32 only, and Conv2d is a direct-convolution baseline.
