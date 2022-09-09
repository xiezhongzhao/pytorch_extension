#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：operators 
@File    ：setup.py
@Author  ：Xie Zhongzhao
@Date    ：2022/8/1 11:35 
'''
from setuptools import setup, Extension
from torch.utils import cpp_extension


## cpu
# setup(name='gelu',
#       ext_modules=[cpp_extension.CppExtension('gelu',
#                       ['gelu.cc'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})

## gpu
# setup(name='gelu',
#       ext_modules=[cpp_extension.CUDAExtension('gelu',
#                                                ['gelu.cc', 'gelu_kernel.cu'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})


## cpu
setup(name='upsample',
      ext_modules=[cpp_extension.CUDAExtension('upsample',
                                               ['upsample.cc'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

## gpu
# setup(name='upsample',
#       ext_modules=[cpp_extension.CUDAExtension('upsample',
#                                                ['upsample.cc', 'upsample_kernel.cu'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})


