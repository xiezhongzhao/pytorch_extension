#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：gelu 
@File    ：check.py
@Author  ：Xie Zhongzhao
@Date    ：2022/8/2 16:44 
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：operators 
@File    ：check.py
@Author  ：Xie Zhongzhao
@date    ：2022/8/1 11:38
'''

# 静态加载
# import torch
# import gelu

# # 同样可以通过 gelu = GELU.apply使用这个激活函数
# class GELU(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.input = input
#         return gelu.forward(input)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input = ctx.input
#         return gelu.backward(grad_output, input)


# 动态加载
# import torch
# from torch.utils.cpp_extension import load
# # PyTorch会进行自动编译，生成对应的模块
# gelu = load(name="gelu", sources=["gelu/gelu.cc", "gelu/gelu_kernel.cu"]) #
#
# class GELUFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.input = input
#         return gelu.forward(input)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input = ctx.input
#         return gelu.backward(grad_output, input)
#
# # a = torch.randn(4,3)
# # b = GELUFunction.apply(a)
#
# a = torch.randn(100, requires_grad=True, dtype=torch.double)
# flag = torch.autograd.gradcheck(GELUFunction.apply, a)
# print("flag: ", flag) # flag: True

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.cpp_extension import load

upsample = load(name="upsample", sources=["operators/upsample.cc", "operators/upsample_kernel.cu"])
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

class UpsampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, align_corners, scale):
        ctx.input = input
        ctx.align_corners = align_corners
        ctx.scale = scale
        return upsample.forward(input, align_corners, scale)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.input
        align_corners = ctx.align_corners
        scale = ctx.scale
        output = upsample.backward(grad_output, input, align_corners, scale)
        return output, None, None

# a = torch.randn((1, 1, 64, 64))
# b = UpsampleFunction.apply(a, False, 2)

up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).cuda()
x = torch.rand(1,1,32,32).cuda()
print("x.shape: ", x.shape)
x.requires_grad = True
x.retain_grad()
y = up(x)

y_grad = torch.rand_like(y).cuda()
y.backward(y_grad)

z = upsample.forward(x, False, 2)
print("the diff of forward: ", np.linalg.norm(y.detach().cpu().numpy() - z.detach().cpu().numpy()))
print("\n")
grad = upsample.backward(y_grad, x, False, 2)
print("the diff of backward: ", np.linalg.norm(grad.detach().cpu().numpy() - x.grad.detach().cpu().numpy()))
print("\n")

# a = torch.randn((2,2,8,8), requires_grad=True, dtype=torch.float32) #
# flag = torch.autograd.gradcheck(UpsampleFunction.apply, (a, False, 2), eps=1e-3)
# print("flag: ", flag) # flag: True


# import time
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.cpp_extension import load
# import tensorflow.compat.v1 as tf # tf 2.4.0
#
# def torch_upsample(input_array, scale=1.0, align_corners=False, use_cuda=False):
#    height, width = a.shape[2], a.shape[3]
#    data = torch.tensor(input_array).view(1,1,height,width).float() # N, C, H, W
#    data = input_array
#    if use_cuda:
#       up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=align_corners).cuda()
#       torch_forward = up(data.cuda())
#       return torch_forward
#    else:
#       up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=align_corners)
#       torch_forward = up(data)
#       return torch_forward
#
# def tf_upsample(input_array, align_corners=False, half_pixel_centers=False):
#    ### reference on the MTK board
#    data = tf.constant(input_array, dtype='float32')
#    data = tf.reshape(data, [1,3,6,1]) # N, H, W, C
#    # half_pixel_centers = False/True  this is a option.
#    out = tf.image.resize_bilinear(data,
#                                   [6,12],
#                                   align_corners=align_corners,
#                                   half_pixel_centers=half_pixel_centers)
#    out = out.numpy()
#    out = np.transpose(out, (0,3,1,2))
#    return out
#
# def my_upsample(a, scale=1.0, align_corners=False, use_cuda=False):
#    # dynamic load
#    myUpsample = load(name="upsample",
#                      sources=["gelu/upsample.cc", "gelu/upsample_kernel.cu"],
#                      verbose=True)
#    height, width = a.shape[2], a.shape[3]
#    if use_cuda:
#       forward_out = myUpsample.forward(a.cuda(), align_corners, scale).cuda()
#       return forward_out
#    else:
#       forward_out = myUpsample.forward(a, align_corners, scale)
#       return forward_out
#
# if __name__ == '__main__':
#    img_size = 16
#    a = torch.range(1, img_size*img_size).reshape(1,1,img_size,img_size).float() # cpu: 7-9 s, cuda: s
#    torch_forward_out, torch_backward_out = torch_upsample(a, scale=2, align_corners=False, use_cuda=False)
#
#    # tf_out = tf_upsample(a, align_corners=False, half_pixel_centers=False)
#    # print("tf_out: \n", tf_out.shape)
#
#    my_forward_out = my_upsample(a, scale=2, align_corners=False, use_cuda=False)
#
#    # print("the diff between tf_out and my_out: \n", tf_out-my_out)
#    print("the forward diff between torch_out and my_out: \n", torch_forward_out-my_forward_out)




