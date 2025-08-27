from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fps_batch_ext',
    ext_modules=[
        CUDAExtension(
            name='fps_batch_ext',
            sources=['csrc/fps_batch.cpp', 'csrc/fps_batch_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math', '-arch=sm_60', '-arch=sm_70', '-arch=sm_75']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)