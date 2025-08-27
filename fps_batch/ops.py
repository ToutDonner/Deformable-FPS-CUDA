import torch
import torch.nn.functional as F
import math

try:
    import fps_batch_ext  # compiled extension
except Exception as e:
    fps_batch_ext = None
    _err = e


def _require_extension():
    if fps_batch_ext is None:
        raise ImportError(
            f"fps_batch_ext is not built. Run `pip install -e .` or `python setup.py build_ext --inplace`. Original error: {_err}"
        )


def _make_batch_ptr_from_idx(batch_idx: torch.Tensor, B: int = None):
    """Create CSR-style batch pointer from per-point batch indices.

    Args:
        batch_idx: LongTensor [M], values in [0, B-1]
        B: optional number of batches; if None uses batch_idx.max()+1
    Returns:
        batch_ptr: LongTensor [B+1], monotonically increasing offsets
    """
    assert batch_idx.dtype == torch.long
    M = batch_idx.numel()
    if B is None:
        B = int(batch_idx.max().item()) + 1 if M > 0 else 0
    device = batch_idx.device
    # counts per batch
    counts = torch.bincount(batch_idx, minlength=B).to(device)
    batch_ptr = torch.empty(B + 1, dtype=torch.long, device=device)
    batch_ptr[0] = 0
    batch_ptr[1:] = torch.cumsum(counts, dim=0)
    return batch_ptr


def fps_batch(points: torch.Tensor,
              batch_ptr: torch.Tensor,
              nsamples: int,
              start_idx: torch.Tensor = None,
              random_start: bool = False) -> torch.Tensor:
    """Batch-aware FPS by CSR pointer.

    Args:
        points: [M, 3] float32/float16, concatenated XYZ.
        batch_ptr: [B+1] long, offsets where each cloud i is points[batch_ptr[i]:batch_ptr[i+1]].
        nsamples: int, number to sample per cloud (will clamp to Ni).
        start_idx: optional [B] long, initial seed indices **relative to each cloud**.
        random_start: if True and no start_idx given, use a random valid seed per cloud.
    Returns:
        out_idx: [B, S] long, indices INTO THE CONCATENATED POINTS (global indices).
    """
    _require_extension()
    assert points.dim() == 2 and points.size(1) == 3
    assert batch_ptr.dim() == 1 and batch_ptr.numel() >= 2
    B = batch_ptr.numel() - 1
    device = points.device
    M = points.size(0)

    if start_idx is None:
        if random_start:
            # pick a valid global seed per batch
            starts = []
            for i in range(B):
                a = int(batch_ptr[i].item())
                b = int(batch_ptr[i+1].item())
                if b > a:
                    gid = torch.randint(a, b, (1,), device=device, dtype=torch.long)
                else:
                    gid = torch.tensor([a], device=device, dtype=torch.long)
                starts.append(gid)
            start_idx_global = torch.cat(starts, dim=0)
        else:
            # default to the first point of each cloud as seed
            start_idx_global = batch_ptr[:-1].clone()
    else:
        # map per-cloud relative start to global index
        assert start_idx.numel() == B
        start_idx_global = start_idx + batch_ptr[:-1]

    # allocate output
    out = torch.empty((B, nsamples), dtype=torch.long, device=device)
    fps_batch_ext.fps_forward(points.contiguous(), batch_ptr.contiguous(), start_idx_global.contiguous(), out)
    return out


def fps_batch_idx(points: torch.Tensor,
                  batch_idx: torch.Tensor,
                  nsamples: int,
                  **kwargs) -> torch.Tensor:
    """Batch-aware FPS from per-point batch indices.

    Args:
        points: [M, 3]
        batch_idx: [M] long in [0, B-1]
        nsamples: samples per cloud
    Returns:
        [B, S] long global indices
    """
    assert batch_idx.dtype == torch.long
    batch_ptr = _make_batch_ptr_from_idx(batch_idx)
    return fps_batch(points, batch_ptr, nsamples, **kwargs)