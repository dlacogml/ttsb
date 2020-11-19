#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cstdio>
#include <cassert>

#define wbCheck(stmt)                                                         \
  do {                                                                        \
    cudaError_t err = stmt;                                                   \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;    \
      std::cerr << "Failed to run stmt " << #stmt << std::endl;               \
      exit(-1);                                                               \
    }                                                                         \
  } while (0)

template <size_t tile_width=16>
__global__
void
conv_forward_kernel(float *y, const float *x, const float *k, const int B,
                    const int M, const int C, const int H, const int W,
                    const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // We have some nice #defs for you below to simplify indexing.
    // Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[                                                \
                                (i3) * (M * H_out * W_out) +                  \
                                (i2) * (H_out * W_out) +                      \
                                (i1) * (W_out) +                              \
                                i0]
#define x4d(i3, i2, i1, i0) x[                                                \
                                (i3) * (C * H * W) +                          \
                                (i2) * (H * W) +                              \
                                (i1) * (W) +                                  \
                                i0]
#define k4d(i3, i2, i1, i0) k[                                                \
                                (i3) * (C * K * K) +                          \
                                (i2) * (K * K) +                              \
                                (i1) * (K) +                                  \
                                i0]
    auto yidx = [M, H_out, W_out](size_t i3, size_t i2, size_t i1, size_t i0)
        -> size_t { return i3*M*H_out*W_out + i2*H_out+W_out + i1*W_out + i2;};
    auto xidx = [C, H, W](size_t i3, size_t i2, size_t i1, size_t i0)
        -> size_t { return i3*C*H*W + i2*H*W + i1*W + i0; };
    auto kidx = [C, K](size_t i3, size_t i2, size_t i1, size_t i0)
        -> size_t { return i3*C*K*K + i2*K*K* + i1*K + i0; };

    // Insert your GPU convolution kernel code here
    // Each thread computes a single output tile
    // Each block computes 16x16 output tiles
    const int W_grid = (W_out + (tile_width-1)) / tile_width;
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int h = (blockIdx.z / W_grid) * tile_width + threadIdx.y;
    const int w = (blockIdx.z % W_grid) * tile_width + threadIdx.x;

    if (h < H_out && w < W_out) {
        float acc = 0.0f;
        for (int c = 0; c < C; ++c) {
            for (int p = 0; p < K; ++p) {
            for (int q = 0; q < K; ++q) {
                float xval = x4d(n, c, h+p, w+q);
                float kval = k4d(m, c, p, q);
                acc += xval * kval;
                // acc += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
            }}
        }
        y4d(n, m, h, w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}

__host__
void
GPUInterface::conv_forward_gpu(float* host_y, const float* host_x,
                               const float* host_k, const int B, const int M,
                               const int C, const int H, const int W,
                               const int K)
{
    // Function paramter definitions:
    // y - output
    // x - input
    // k - kernel
    // B - batch_size (number of images in x)
    // M - number of output feature maps
    // C - number of input feature maps
    // H - input height dimension
    // W - input width dimension
    // K - kernel height and width (K x K)
    // Declare relevant device pointers
    float* dev_y = nullptr;
    float* dev_x = nullptr;
    float* dev_k = nullptr;

    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const size_t size_x = B * C * H * W * sizeof(*dev_x);
    const size_t size_y = B * M * H_out * W_out * sizeof(*dev_y);
    const size_t size_k = M * C * K * K * sizeof(*dev_k);

    wbCheck(cudaMalloc(&dev_y, size_y));
    wbCheck(cudaMalloc(&dev_x, size_x));
    wbCheck(cudaMalloc(&dev_k, size_k));

    wbCheck(cudaMemcpy(dev_x, host_x, size_x, cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(dev_k, host_k, size_k, cudaMemcpyHostToDevice));

    // Set the kernel dimensions and call the kernel
    // kernel dimensions:
    static constexpr size_t tile_width = 16;
    // Block Dims --- 16 x 16.  Each thread computes a single output tile
    dim3 BlockDim(tile_width, tile_width);
    // Grid Dims: (X, Y, Z) --- (batch, output feature map, tile)
    size_t Zy = (H_out + (tile_width-1)) / tile_width;
    size_t Zx = (W_out + (tile_width-1)) / tile_width;
    dim3 GridDim(B, M, Zy * Zx);

    conv_forward_kernel<tile_width> <<<GridDim, BlockDim>>>(
            dev_y, dev_x, dev_k, B, M, C, H, W, K
    );

    // Copy the output back to host
    wbCheck(cudaMemcpy(host_y, dev_y, size_y, cudaMemcpyDeviceToHost));

    // Free device memory
    wbCheck(cudaFree(dev_y));
    wbCheck(cudaFree(dev_k));
    wbCheck(cudaFree(dev_x));


    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
