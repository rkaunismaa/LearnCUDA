"""
setup.py for the vector_add_scale CUDA extension.

Build and install:
    pip install -e .

Or build in place:
    python setup.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vector_add_ext',
    ext_modules=[
        CUDAExtension(
            name='vector_add_ext',
            sources=['vector_add.cpp', 'vector_add_cuda.cu'],
            extra_compile_args={
                'cxx':  ['-O2'],
                'nvcc': ['-O2', '-arch=sm_89', '--use_fast_math'],
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
