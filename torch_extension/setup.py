from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spmm_cuda',
    ext_modules=[
        CUDAExtension(
            name='spmm_cuda',
            sources=[
                'spmm_extension.cpp',
                '../src/kernel.cu',
                '../src/wrapper.cpp'
            ],
            include_dirs=['../include'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)