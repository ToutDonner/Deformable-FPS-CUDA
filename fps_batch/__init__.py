import torch
from torch import Tensor
from typing import Optional

try:
    import fps_batch_ext
except ImportError as e:
    raise ImportError("Failed to load fps_batch_ext extension. Make sure it is built and installed correctly.") from e

def fps_batch_forward(points: Tensor, batch_ptr: Tensor, n_samples: int = 8192) -> Tensor:
    """
    Farthest Point Sampling for a batch of point clouds.

    Args:
        points: Tensor of shape (N, 3) containing point cloud data.
        batch_ptr: Tensor of shape (B+1,) indicating start and end indices of each point cloud in the batch.
        n_samples: Number of points to sample per point cloud.

    Returns:
        Tensor of shape (B, n_samples) containing indices of sampled points.
    """
    return fps_batch_ext.fps_forward(points, batch_ptr, n_samples)