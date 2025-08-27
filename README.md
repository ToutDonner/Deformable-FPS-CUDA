# What is this
An implemention of further points sampling (FPS), utlize parallel computing with CUDA. Can concat different length pointcloud to one batch.

# How to install 
1. git clone git@github.com:ToutDonner/Deformable-FPS-CUDA.git
2. cd Deformable-FPS-CUDA
3. pip install -e .

# code example
```python
import torch
import fps_batch_ext

points = torch.randn((1000, 3), device="cuda")
batch_ptr = torch.tensor([0, 500, 1000], device="cuda", dtype=torch.long)
start_idx = torch.zeros(2, dtype=torch.long, device="cuda")

out = fps_batch_ext.fps_forward(points, batch_ptr, start_idx, 128)
print(out.shape)  # [B, n_samples]
```
