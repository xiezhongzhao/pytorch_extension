#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "ticktock.h"

#define CUDA_1D_KERNEL_LOOP(i,n) \
    for(int32_t i = blockIdx.x*blockDim.x+threadIdx.x; i < (n); \
        i += blockDim.x*gridDim.x)

//#define CUDA_2D_KERNEL_LOOP(i,n,j,m) \
//    for(size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x * gridDim.x)
//        for(size_t j = blockIdx.y*blockDim.y+threadIdx.y; j<m; j += blockDim.y * gridDim.y)
//
//#define CUDA_2D_KERNEL_BLOCK_LOOP(i,n,j,m) \
    for(size_t i = blockIdx.x; i < n; i += gridDim.x) \
        for(size_t j = blockIdx.y; j < m; j += girdDim.y)

static void HandleError(cudaError_t err,
                       const char *file,
                       int line){
    if(err != cudaSuccess){
        printf("%s in %s at line %d\n",
               cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

int getThreadNum(){
    cudaDeviceProp prop;
    int count;

    // max thread num: 1024
    // max grid dimensions: 2147483647, 65535, 65535
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    //printf("gpu num %d\n", count);
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    //printf("max thread num: %d\n", prop.maxThreadsPerBlock);
    //printf("max grid dimensions: %d, %d, %d\n",
    //       prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    return prop.maxThreadsPerBlock;
}

__global__ void upsample_bilinear2d_forward(const int batch_size,
                                            const int channels,
                                            const int src_h,
                                            const int src_w,
                                            const int dst_h,
                                            const int dst_w,
                                            float* input,
                                            float* output,
                                            bool align_corners,
                                            int n_size){

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    //CUDA_1D_KERNEL_LOOP(index, n_size)
    if(index < n_size)
    {
        // index of output matrix
        const int c = index % channels;
        const int h_d = (index / channels) % dst_w;
        const int w_d = (index / channels / dst_w) % dst_h;
        const int n = index / channels / dst_w / dst_h;

        float h1r = 0.0, w1r = 0.0;
        if(align_corners){
            h1r = float(src_h-1)/(dst_h-1)*h_d;
            w1r = float(src_w-1)/(dst_w-1)*w_d;
        }
        else{
            h1r = float(src_h)/dst_h*(h_d+0.5)-0.5;
            w1r = float(src_w)/dst_w*(w_d+0.5)-0.5;
        }

        // horizontal direction
        int h1 = int(h1r);
        int h1p = (h1<src_h-1) ? 1 : 0;
        float h1lambda = (h1r-h1)>0 ? h1r-h1 : 0;
        float h0lambda = 1.0 - h1lambda;

        // vertical direction
        int w1 = int(w1r);
        int w1p = (w1<src_w-1) ? 1 : 0;
        float w1lambda = (w1r-w1)>0 ? w1r-w1 : 0;
        float w0lambda = 1.0 - w1lambda;

        // parallel
        float r0 = w0lambda * input[n*(channels*src_w*src_h) + c*(src_w*src_h) + h1*src_w + w1] + \
                   w1lambda * input[n*(channels*src_w*src_h) + c*(src_w*src_h) + h1*src_w + (w1+w1p)];
        float r1 = w0lambda * input[n*(channels*src_w*src_h) + c*(src_w*src_h) + (h1+h1p)*src_w + w1] + \
                   w1lambda * input[n*(channels*src_w*src_h) + c*(src_w*src_h) + (h1+h1p)*src_w + (w1+w1p)];
        float val = h0lambda * r0 + h1lambda * r1;

        output[n*(channels*dst_w*dst_h) + c*(dst_w*dst_h) + h_d*dst_w + w_d] = val;
    }

}

__global__ void upsample_bilinear2d_backward(const int batch,
                                             const int channels,
                                             const int src_h,
                                             const int src_w,
                                             const int dst_h,
                                             const int dst_w,
                                             float* grad_out,
                                             float* grad_in,
                                             const bool align_corners,
                                             const int n_size){

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    //CUDA_1D_KERNEL_LOOP(index, n_size)
    if(index < n_size)
    {
        // index of output matrix
        const int c = index % channels;
        const int h_d = (index / channels) % dst_w;
        const int w_d = (index / channels / dst_w) % dst_h;
        const int n = index / channels / dst_w / dst_h;

        float h1r = 0.0, w1r = 0.0;
        if(align_corners){
            h1r = float(src_h-1)/(dst_h-1)*h_d;
            w1r = float(src_w-1)/(dst_w-1)*w_d;
        }
        else{
            h1r = float(src_h)/dst_h*(h_d+0.5)-0.5;
            w1r = float(src_w)/dst_w*(w_d+0.5)-0.5;
        }

        // horizontal direction
        int h1 = int(h1r);
        int h1p = (h1<src_h-1) ? 1 : 0;
        float h1lambda = (h1r-h1)>0 ? h1r-h1 : 0;
        float h0lambda = 1.0 - h1lambda;

        // vertical direction
        int w1 = int(w1r);
        int w1p = (w1<src_w-1) ? 1 : 0;
        float w1lambda = (w1r-w1)>0 ? w1r-w1 : 0;
        float w0lambda = 1.0 - w1lambda;

        // parallel
        float val_out = grad_out[n*(channels*dst_w*dst_h) + c*(dst_w*dst_h) + h_d*dst_w + w_d];
        atomicAdd(&grad_in[n*(channels*src_w*src_h) + c*(src_w*src_h) + h1*src_w + w1],
                  h0lambda * w0lambda * val_out);

        atomicAdd(&grad_in[n*(channels*src_w*src_h) + c*(src_w*src_h) + h1*src_w + (w1+w1p)],
                  h0lambda * w1lambda * val_out);

        atomicAdd(&grad_in[n*(channels*src_w*src_h) + c*(src_w*src_h) + (h1+h1p)*src_w + w1],
                  h1lambda * w0lambda * val_out);

        atomicAdd(&grad_in[n*(channels*src_w*src_h) + c*(src_w*src_h) + (h1+h1p)*src_w + (w1+w1p)],
                  h1lambda * w1lambda * val_out);

    }
}

__host__ float* upsample_fwd_interface(const int batch,
                                    const int channels,
                                    const int src_h,
                                    const int src_w,
                                    const int dst_h,
                                    const int dst_w,
                                    float* input,
                                    float* output,
                                    bool align_corners,
                                    int n_d){

    int threadNum = getThreadNum();
    int blockNum = (n_d + threadNum - 1) / threadNum;

    int n_s = batch * channels * src_h * src_w; // elements of the input
    n_d = batch * channels * dst_h * dst_w; // elements of the output

    float* inputGPU;
    float* outputGPU;
    HANDLE_ERROR(cudaMalloc((void**)&inputGPU, n_s*sizeof(float))); // allocate the memory in cuda
    HANDLE_ERROR(cudaMalloc((void**)&outputGPU, n_d*sizeof(float)));

    // move the data from host to device(cuda)
    HANDLE_ERROR(cudaMemcpy(inputGPU, input, n_s*sizeof(float), cudaMemcpyHostToDevice));

    //TICK(upsample_bilinear2d_forward);
    upsample_bilinear2d_forward<<<blockNum, threadNum, 0, at::cuda::getCurrentCUDAStream()>>>(batch, channels,
                                                         src_h, src_w,
                                                         dst_h, dst_w,
                                                         inputGPU, outputGPU,
                                                         align_corners, n_d);
    cudaDeviceSynchronize();
    //TOCK(upsample_bilinear2d_forward);

    float* odata = new float[n_d];
    // move the result from device(cuda) to host
    HANDLE_ERROR(cudaMemcpy(odata, outputGPU, n_d*sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(inputGPU);
    cudaFree(outputGPU);

    return odata;

}

__host__ float* upsample_bwd_interface(const int batch,
                                     const int channels,
                                     const int src_h,
                                     const int src_w,
                                     const int dst_h,
                                     const int dst_w,
                                     float* grad_output,
                                     float* grad_input,
                                     bool align_corners,
                                     int n_s){
    int threadNum = getThreadNum();
    int blockNum = (n_s + threadNum - 1) / threadNum;

    n_s = batch * channels * src_h * src_w; // elements of the grad_input
    int n_d = batch * channels * dst_h * dst_w; // elements of the grad_output
    int n_weight = batch * channels * src_h * src_w * dst_h * dst_w;

    float* grad_out;
    float* grad_in;
    //float* grad_weight;
    HANDLE_ERROR(cudaMalloc((void**)&grad_out, n_d*sizeof(float))); // allocate the memory in cuda
    HANDLE_ERROR(cudaMalloc((void**)&grad_in, n_s*sizeof(float)));

    // move the data from host to device(cuda)
    HANDLE_ERROR(cudaMemcpy(grad_out, grad_output, n_d*sizeof(float), cudaMemcpyHostToDevice));

    //TICK(upsample_bilinear2d_backward);
    upsample_bilinear2d_backward<<<blockNum, threadNum, 0, at::cuda::getCurrentCUDAStream()>>>(batch, channels,
                                                          src_h, src_w,
                                                          dst_h, dst_w,
                                                          grad_out, grad_in,
                                                          align_corners, n_d);
    cudaDeviceSynchronize();
    //TOCK(upsample_bilinear2d_backward);

    float* res = new float[n_s];
    // move the result from device(cuda) to host
    HANDLE_ERROR(cudaMemcpy(res, grad_in, n_s*sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(grad_out);
    cudaFree(grad_in);
    return res;
}

























































































