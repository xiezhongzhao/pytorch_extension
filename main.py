import numpy as np
import tensorflow.compat.v1 as tf # tf 2.4.0
import torch # torch 1.10.1+cpu
import torch.nn as nn
import numpy as np
import sys

def tf_upsample(input_array, align_corners=False, half_pixel_centers=False):
   ### reference on the MTK board
   data = tf.constant(input_array, dtype='float32')
   data = tf.reshape(data, [1,3,6,1]) # N, H, W, C
   # half_pixel_centers = False/True  this is a option.
   out = tf.image.resize_bilinear(data, [6,12], align_corners=align_corners, half_pixel_centers=half_pixel_centers)
   out = out.numpy()
   out = np.transpose(out, (0,3,1,2))
   return out


def torch_upsample(input_array, align_corners=False):
   data = torch.tensor(input_array).view(1,1,3,6).float() # N, C, H, W
   up = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=align_corners)
   out_torch_up = up(data)
   out = out_torch_up.numpy()
   return out


def my_upsample(input_array, scale=2, align_corners=False, half_pixel_centers=False):
   data = input_array # shape(3,6)
   src = torch.tensor(data).view(1,1,3,6).float() # n, c, h, w

   src_n, src_c, src_h, src_w = src.shape
   dst_n, dst_c, dst_h, dst_w = src_n, src_c, scale*src_h, scale*src_w

   if src_h == dst_h and src_w == dst_w:
      return src.copy()

   # project (dst_h, dst_w) coordinates into (src_h, src_w)
   hd = torch.arange(0, dst_h) # [6]
   wd = torch.arange(0, dst_w) # [12]

   # if align_corners: #tf
   #     h = float(src_h)/dst_h * (hd+0.5) - 0.5
   #     w = float(src_w)/dst_w * (wd+0.5) - 0.5
   # else:
   #     h = float(src_h)/(dst_h)*hd # 0.5*[0, 1, 2, 3, 4, 5]
   #     w = float(src_w)/(dst_w)*wd # 0.5*[0, 1, 2, 3, 4, 5, ....., 11]

   if align_corners: #torch
      h = float(src_h - 1) / (dst_h - 1) * hd  # 2/5*[0, 1, 2, 3, 4, 5]
      w = float(src_w - 1) / (dst_w - 1) * wd  # 2/5*[0, 1, 2, 3, 4, 5, ....., 11]
   else:
      h = float(src_h) / dst_h * (hd + 0.5) - 0.5
      w = float(src_w) / dst_w * (wd + 0.5) - 0.5

   h = torch.clamp(h, 0, src_h-1) #(input, min, max, out=None)
   w = torch.clamp(w, 0, src_w-1)

   h = h.view(dst_h, 1) #[6, 1]
   w = w.view(1, dst_w) #[1, 12]

   h = h.repeat(1, dst_w) #[6, 12]
   w = w.repeat(dst_h, 1) #[6, 12]

   #get the four coordinates
   h0 = torch.clamp(torch.floor(h), 0, src_h-2) #[6, 12]
   w0 = torch.clamp(torch.floor(w), 0, src_w-2) #[6, 12]
   h0 = h0.long()
   w0 = w0.long()

   h1 = h0 + 1
   w1 = w0 + 1

   q00 =  src[..., h0, w0] #[1,1,6,12]
   q01 =  src[..., h0, w1]
   q10 =  src[..., h1, w0]
   q11 =  src[..., h1, w1]
   print("src.shape: " , src.shape)
   print("q00.shape: ", q00.shape)

   r0 = (w1-w)*q00 + (w-w0)*q01
   r1 = (w1-w)*q10 + (w-w0)*q11

   dst = (h1-h)*r0 + (h-h0)*r1
   dst = dst.numpy()

   return dst


def bilinear_interpolation_naive(src, dst_size, align_corners=False):
   """
   双线性差值的naive实现
   :param src: 源图像
   :param dst_size: 目标图像大小H*W
   :return: 双线性差值后的图像
   """
   src = src.astype(np.float)
   (src_h, src_w, src_c) = src.shape  # 原图像大小 H*W*C
   (dst_h, dst_w), dst_c = dst_size, src_c  # 目标图像大小H*W*C

   if src_h == dst_h and src_w == dst_w:  # 如果大小不变，直接返回copy
      return src.copy()

   dst = np.zeros((dst_h, dst_w, dst_c), dtype=src.dtype)  # 目标图像初始化(6, 12)
   # 将目标图像的坐标转换到源图像中计算相应的插值
   for h_d in range(dst_h):
      for w_d in range(dst_w):
         if align_corners:
             h1r = float(src_h-1) / (dst_h-1) * h_d
             w1r = float(src_w-1) / (dst_w-1) * w_d
         else:
             h1r = float(src_h)/dst_h * (h_d + 0.5) - 0.5  # 将目标图像H坐标映射到源图像上
             w1r = float(src_w)/dst_w * (w_d + 0.5) - 0.5  # 将目标图像W坐标映射到源图像上

         # 计算h方向相邻的位置
         h1 = int(h1r)
         h1p = 1 if (h1 < src_h-1) else 0
         h1lambda = max(0, h1r - h1)
         h0lambda = 1 - h1lambda

         # 计算w方向相邻的位置
         w1 = int(w1r)
         w1p = 1 if (w1 < src_w-1) else 0
         w1lambda = max(0, w1r - w1)
         w0lambda = 1 - w1lambda

         # 根据相邻的四点坐标，计算出插值
         r0 = w0lambda*src[h1, w1, :] + w1lambda*src[h1, w1+w1p, :]
         r1 = w0lambda*src[h1+h1p, w1, :] + w1lambda*src[h1+h1p, w1+w1p, :]
         p = h0lambda*r0 + h1lambda*r1

         dst[h_d, w_d, :] = p

   return dst

if __name__ == '__main__':

   input_array = np.array(range(1,19)).reshape(3,6)
   print(input_array)

   # torch_out = torch_upsample(input_array, align_corners=False) # half_pixel=True
   # my_upsample_out = my_upsample(input_array, scale=2, align_corners=False)

   # tf_out = tf_upsample(input_array, align_corners=False, half_pixel_centers=False)
   my_out = my_upsample(input_array, scale=2, align_corners=False)
   # print("the diff between tf_out and my_out: \n", torch_out-my_out)

   naive_upsample = bilinear_interpolation_naive(input_array.reshape(3,6,1), (6, 12), align_corners=False)
   my_out = my_out.reshape(6, 12, 1)
   diff = (my_out - naive_upsample).reshape(6, 12)
   print("the diff between naive_upsample and torch_out: \n", diff)


   # tf_out = tf_upsample(input_array, align_corners=False, half_pixel_centers=True)
   # my_out = my_upsample(input_array, scale=2, align_corners=True)
   # print("the diff between tf_out and my_out: \n", torch_out-my_out)

   # tf_out = tf_upsample(input_array, align_corners=True, half_pixel_centers=False)
   # my_out = my_upsample(input_array, scale=2, align_corners=False)
   # print("the diff between tf_out and my_out: \n", tf_out-my_out)
   # print("the correct result:\n", tf_out)
   # print("the diff between torch_out and my_out: \n", torch_out-my_out)

   # print("the diff between torch_and my_out: \n", torch_out-my_out)











