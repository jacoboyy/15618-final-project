from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='generate_mask_cuda',
    ext_modules=[
        CUDAExtension(
            name='generate_mask_cuda',
            sources=['generate_mask_cuda.cu',
            'host.cu',
            'prune_tree_cuda.cu',
            'tensor_parallel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
