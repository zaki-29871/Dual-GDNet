from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='gdnet_lib',
      ext_modules=[cpp_extension.CUDAExtension('gdnet_lib', ['cuda_lib.cpp', 'cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})