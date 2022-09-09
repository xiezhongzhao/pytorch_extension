#include <torch/extension.h>
#include <ATen/ATen.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#include "ticktock.h"

#define CHECK_CUDA(x) \
    AT_CHECK(x.type().is_cuda(), #x, "must be a CUDAtensor")
#define CHECK_CPU(x) \
    AT_CHECK(!x.type().is_cuda(), #x, "must ba a CPU tensor")
#define CHECK_CONTIGUOUS(x) \
    AT_CHECK(x.is_contiguous(), #x, "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using at::Half;
using at::Tensor;
using phalf = at::Half;


float* upsample_fwd_interface(const int batch,
                              const int channels,
                              const int src_h,
                              const int src_w,
                              const int dst_h,
                              const int dst_w,
                              float* input,
                              float* output,
                              bool align_corners,
                              int n);

float* upsample_bwd_interface(const int batch,
                              const int channels,
                              const int src_h,
                              const int src_w,
                              const int dst_h,
                              const int dst_w,
                              float* grad_output,
                              float* grad_input,
                              bool align_corners,
                              int n);

at::Tensor upsample_fwd(at::Tensor& input,
                        const bool align_corners,
                        const int scale){
    if(input.device().type() == c10::DeviceType::CPU){
        int batch = input.size(0);
        int channels = input.size(1);
        int src_h = input.size(2);
        int src_w = input.size(3);
        int dst_h = scale * src_h;
        int dst_w = scale * src_w;

        if((src_h == dst_h) && (src_w == dst_w))
            return input;

        auto bench = std::chrono::steady_clock::now();
        auto dst = at::zeros({batch, channels, dst_h, dst_w});
        for(int n=0; n<batch; ++n){
            for(int c=0; c<channels; ++c){
                for(int h_d=0; h_d<dst_h; ++h_d){
                    for(int w_d=0; w_d<dst_w; ++w_d){
                        float h1r=0.0, w1r=0.0;
                        if(align_corners){
                            h1r = float(src_h-1)/(dst_h-1)*h_d;
                            w1r = float(src_w-1)/(dst_w-1)*w_d;
                        }else{
                            h1r = float(src_h)/dst_h*(h_d+0.5)-0.5;
                            w1r = float(src_w)/dst_w*(w_d+0.5)-0.5;
                        }

                        // calculate the neighbor in vertical(h) direction
                        int h1 = int(h1r);
                        int h1p = (h1<src_h-1) ? 1 : 0;
                        float  h1lambda = (h1r-h1)>0 ? h1r-h1 : 0;
                        float h0lambda = 1.0 - h1lambda;

                        // calculate the neighbor in horizontal(w) direction
                        int w1 = int(w1r);
                        int w1p = (w1<src_w-1) ? 1 : 0;
                        float w1lambda = (w1r-w1)>0 ? w1r-w1 : 0;
                        float w0lambda = 1.0 - w1lambda;

                        // calculate the interpolation from the adjacent points
                        float r0 = w0lambda*input[n][c][h1][w1].item().to<float>() + \
                                   w1lambda*input[n][c][h1][w1+w1p].item().to<float>();
                        float r1 = w0lambda*input[n][c][h1+h1p][w1].item().to<float>() + \
                                   w1lambda*input[n][c][h1+h1p][w1+w1p].item().to<float>();
                        float p = h0lambda*r0 + h1lambda*r1;

                        dst[n][c][h_d][w_d] = p;
                    }
                }
            }
        }
        printf("forward cpu: %lfs\n", std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench).count());
        return dst;
   }
   else if(input.device().type() == c10::DeviceType::CUDA){
        TORCH_CHECK(input.dtype() == torch::kFloat32, "Datatype not implemented");
        int batch = input.size(0);
        int channels = input.size(1);
        int src_h = input.size(2);
        int src_w = input.size(3);
        int dst_h = scale * src_h;
        int dst_w = scale * src_w;

        if((src_h == dst_h) && (src_w == dst_w))
            return input;

        int n_s = batch * channels * src_h * src_w; // elements of the input
        int n_d = batch * channels * dst_h * dst_w; // elements of the output

        auto output = at::zeros({batch, channels, dst_h, dst_w});
        float* input_cpu = input.data_ptr<float>();
        float* output_cpu = output.data_ptr<float>();

        auto bench = std::chrono::steady_clock::now();
        output_cpu = upsample_fwd_interface(batch, channels,
                                            src_h, src_w,
                                            dst_h, dst_w,
                                            input_cpu, output_cpu,
                                            align_corners, n_d);
        printf("forward cuda: %lfs\n",
                std::chrono::duration_cast<std::chrono::duration<double>>
                (std::chrono::steady_clock::now() - bench).count());

        for(int n=0; n<batch; ++n){
            for(int c=0; c<channels; ++c){
                for(int h=0; h<dst_h; ++h){
                    for(int w=0; w<dst_w; ++w){
                        float val = output_cpu[n*channels*dst_h*dst_w + \
                                               c*dst_h*dst_w + \
                                               h*dst_w + w];
                        output[n][c][h][w] = val;
                    }
                }
            }
        }
        return output;
   }
   AT_ERROR("No such device: ", input.device());
}

at::Tensor upsample_bwd(at::Tensor& grad_output,
                        at::Tensor& input,
                        const bool align_corners,
                        const int scale){

    if(grad_output.device().type() == c10::DeviceType::CPU){

        int batch = grad_output.size(0);
        int channels = grad_output.size(1);
        int dst_h = grad_output.size(2);
        int dst_w = grad_output.size(3);

        int src_h = dst_h / scale;
        int src_w = dst_w / scale;

        if((src_h == dst_h) && (src_w == dst_w))
            return grad_output;

        auto bench = std::chrono::steady_clock::now();
        auto grad_input = at::zeros({batch, channels, src_h, src_w});
        for(int n=0; n<batch; ++n){
            for(int c=0; c<channels; ++c){
                for(int h_d=0; h_d<dst_h; ++h_d){
                    for(int w_d=0; w_d<dst_w; ++w_d){
                        float h1r=0.0, w1r=0.0;
                        if(align_corners){
                            h1r = float(src_h-1)/(dst_h-1)*h_d;
                            w1r = float(src_w-1)/(dst_w-1)*w_d;
                        }else{
                            h1r = float(src_h)/dst_h*(h_d+0.5)-0.5;
                            w1r = float(src_w)/dst_w*(w_d+0.5)-0.5;
                        }

                        // calculate the neighbor in vertical(h) direction
                        int h1 = int(h1r);
                        int h1p = (h1<src_h-1) ? 1 : 0;
                        float  h1lambda = (h1r-h1)>0 ? h1r-h1 : 0;
                        float h0lambda = 1.0 - h1lambda;

                        // calculate the neighbor in horizontal(w) direction
                        int w1 = int(w1r);
                        int w1p = (w1<src_w-1) ? 1 : 0;
                        float w1lambda = (w1r-w1)>0 ? w1r-w1 : 0;
                        float w0lambda = 1.0 - w1lambda;

                        float val = grad_output[n][c][h_d][w_d].item().to<float>();
                        grad_input[n][c][h1][w1] += (h0lambda*w0lambda)*val;
                        grad_input[n][c][h1][w1+w1p] += (h0lambda*w1lambda)*val;
                        grad_input[n][c][h1+h1p][w1] += (h1lambda*w0lambda)*val;
                        grad_input[n][c][h1+h1p][w1+w1p] += (h1lambda*w1lambda)*val;

                    }
                }
            }
        }
        printf("backward cpu: %lfs\n", std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench).count());
        return grad_input;
    }
    else if(grad_output.device().type() == c10::DeviceType::CUDA){
        TORCH_CHECK(grad_output.dtype() == torch::kFloat32, "Datatype not implemented");

        int batch = grad_output.size(0);
        int channels = grad_output.size(1);
        int dst_h = grad_output.size(2);
        int dst_w = grad_output.size(3);

        int src_h = dst_h / scale;
        int src_w = dst_w / scale;

        if((src_h == dst_h) && (src_w == dst_w))
            return grad_output;

        int n_s = batch * channels * src_h * src_w; // elements of the output
        int n_d = batch * channels * dst_h * dst_w; // elements of the output

        auto grad_input = at::zeros({batch, channels, src_h, src_w});
        float* grad_out = grad_output.data_ptr<float>();
        float* grad_in = grad_input.data_ptr<float>();

        auto bench = std::chrono::steady_clock::now();
        grad_in = upsample_bwd_interface(batch, channels,
                                        src_h, src_w,
                                        dst_h, dst_w,
                                        grad_out, grad_in,
                                        align_corners, n_d);
        printf("backward cuda: %lfs\n",
                std::chrono::duration_cast<std::chrono::duration<double>>
                (std::chrono::steady_clock::now() - bench).count());

        for(int n=0; n<batch; ++n){
            for(int c=0; c<channels; ++c){
                for(int h=0; h<src_h; ++h){
                    for(int w=0; w<src_w; ++w){
                        float val = grad_in[n*channels*src_h*src_w + \
                                            c*src_h*src_w + \
                                            h*src_w + w];
                        grad_input[n][c][h][w] = val;
                    }
                }
            }
        }

        return grad_input;
    }
    AT_ERROR("NO such device: ", grad_output.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &upsample_fwd, "upsample forward");
    m.def("backward", &upsample_bwd, "upsample backward");
}





