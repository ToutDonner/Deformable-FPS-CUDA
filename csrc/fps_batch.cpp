#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) \
  TORCH_CHECK(x.dtype() == torch::kLong, #x " must be a long tensor")

// 声明 CUDA 接口
at::Tensor fps_forward_cuda(const at::Tensor& points,
                            const at::Tensor& batch_ptr,
                            const at::Tensor& start_idx, int n_samples);

// C++ wrapper
at::Tensor fps_forward(const at::Tensor& points, const at::Tensor& batch_ptr,
                       const at::Tensor& start_idx, int n_samples) {
  CHECK_CUDA(points);
  CHECK_CONTIGUOUS(points);
  CHECK_CUDA(batch_ptr);
  CHECK_CONTIGUOUS(batch_ptr);
  CHECK_LONG(batch_ptr);
  CHECK_CUDA(start_idx);
  CHECK_CONTIGUOUS(start_idx);
  CHECK_LONG(start_idx);

  return fps_forward_cuda(points, batch_ptr, start_idx, n_samples);
}

// PYBIND11_MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fps_forward", &fps_forward, "Batched Farthest Point Sampling (CUDA)");
}
