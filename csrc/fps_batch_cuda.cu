#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__device__ __forceinline__ scalar_t dist2(const scalar_t* a, const scalar_t* b) {
    return (a[0]-b[0])*(a[0]-b[0]) +
           (a[1]-b[1])*(a[1]-b[1]) +
           (a[2]-b[2])*(a[2]-b[2]);
}

// Kernel: 每个 batch 对应一个 block
template <typename scalar_t>
__global__ void fps_batch_kernel(
    const scalar_t* __restrict__ points,  // [N,3]
    const long* __restrict__ batch_ptr,   // [B+1]
    const long* __restrict__ start_idx,   // [B]
    long* __restrict__ out_idx,           // [B, S]
    scalar_t* __restrict__ temp_dists,    // [N]
    const int S)
{
    const int b = blockIdx.x;
    const long start = batch_ptr[b];
    const long end = batch_ptr[b+1];
    const long N = end - start;
    if (N <= 0) return;

    const scalar_t* pts = points + start*3;
    scalar_t* min_d2 = temp_dists + start;

    extern __shared__ unsigned char smem[];
    scalar_t* svals = reinterpret_cast<scalar_t*>(smem);
    long* sidx = reinterpret_cast<long*>(smem + blockDim.x*sizeof(scalar_t));

    // 初始化最小距离
    for (long i = threadIdx.x; i < N; i += blockDim.x)
        min_d2[i] = std::numeric_limits<scalar_t>::infinity();
    __syncthreads();

    long cur = start_idx[b];

    for (int s=0; s<S; s++) {
        if (threadIdx.x == 0)
            out_idx[b*S + s] = cur;
        __syncthreads();

        const scalar_t* pcur = points + cur*3;

        // 更新最小距离
        for (long i = threadIdx.x; i < N; i += blockDim.x) {
            scalar_t d = dist2<scalar_t>(pts + i*3, pcur);
            if (d < min_d2[i]) min_d2[i] = d;
        }
        __syncthreads();

        // 找最远点
        long best_i = -1;
        scalar_t best_d = -1.0;
        for (long i = threadIdx.x; i < N; i += blockDim.x) {
            if (min_d2[i] > best_d) {
                best_d = min_d2[i];
                best_i = i;
            }
        }

        // 共享内存归约
        svals[threadIdx.x] = best_d;
        sidx[threadIdx.x] = best_i;
        __syncthreads();

        for (int offset = blockDim.x >> 1; offset > 0; offset >>=1) {
            if (threadIdx.x < offset) {
                if (svals[threadIdx.x + offset] > svals[threadIdx.x]) {
                    svals[threadIdx.x] = svals[threadIdx.x + offset];
                    sidx[threadIdx.x] = sidx[threadIdx.x + offset];
                }
            }
            __syncthreads();
        }

        long next_local = sidx[0] < 0 ? 0 : sidx[0];
        cur = start + next_local;
        __syncthreads();
    }
}

// 返回 tensor 版本
at::Tensor fps_forward_cuda(
    const at::Tensor& points,
    const at::Tensor& batch_ptr,
    const at::Tensor& start_idx,
    int n_samples)
{
    const int B = batch_ptr.size(0) - 1;

    // 输出 tensor
    auto out_idx = at::zeros({B, n_samples}, batch_ptr.options());

    // 临时最小距离缓冲
    auto temp_dists = at::empty({points.size(0)}, points.options());

    const int threads = 512;
    size_t shmem = threads*(sizeof(float)+sizeof(long));

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "fps_forward_cuda", ([&] {
        fps_batch_kernel<scalar_t><<<B, threads, shmem>>>(
            points.data_ptr<scalar_t>(),
            batch_ptr.data_ptr<long>(),
            start_idx.data_ptr<long>(),
            out_idx.data_ptr<long>(),
            temp_dists.data_ptr<scalar_t>(),
            n_samples
        );
    }));

    return out_idx;
}
