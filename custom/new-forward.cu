#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define wbCheck(stmt)                                                         \
  do {                                                                        \
    cudaError_t err = stmt;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error: %s", cudaGetErrorString(err));             \
      fprintf(stderr, "Failed to run stmt %s", #stmt);                        \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

static constexpr size_t smsize = 1 << 16;
static constexpr size_t smfloats = smsize / sizeof(float);

__constant__ float c_weights [smfloats];

float* device_k;
__global__
void
conv_forward_kernel(float* y, const float* x, const float* k, const int B,
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

    auto y4d = [&y, M, H_out, W_out](int i3, int i2, int i1, int i0) -> float&
    {
        return y[i3*M*H_out*W_out + i2*H_out*W_out + i1*W_out + i0];
    };
    auto x4d = [&x, C, H, W](int i3, int i2, int i1, int i0) -> float {
        return x[i3*C*H*W + i2*H*W + i1*W + i0];
    };
    auto k4d = [&k, C, K](int  i3, int i2, int i1, int i0) -> float {
        return c_weights[i3*C*K*K + i2*K*K + i1*K + i0];
    };
// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    const int m = blockIdx.z;
    const int h = blockIdx.y*TILE_WIDTH + threadIdx.y;
    const int w = blockIdx.x*TILE_WIDTH + threadIdx.x;

    if (h >= H_out || w >= W_out) {
        return;
    }

    for (int b = 0; b < B; ++b) {
        float acc = 0.0f;
        for (int c = 0; c < C; ++c) {
            for (int p = 0; p < K; ++p) {
                for (int q = 0; q < K; ++q) {
                    acc += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
                }
            }
        }
        y4d(b, m, h, w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}

__host__
void
GPUInterface::conv_forward_gpu(float* host_y, const float* host_x,
        const float* host_k, const int B, const int M, const int C,
        const int H, const int W, const int K)
{
    // Declare relevant device pointers
    float* device_x;
    float* device_y;

    // Allocate memory and copy over the relevant data structures to the GPU
    const unsigned int H_out = H - K + 1;
    const unsigned int W_out = W - K + 1;

    unsigned int inputArrayLength = B*C*H*W;
    unsigned int outputArrayLength = B*M*H_out*W_out;
    unsigned int kernelArrayLength = M*C*K*K;

    // check if kernel weights fit in const memory
    if (kernelArrayLength > smfloats) {
        fprintf(stderr,
                "kernelArrayLength %d does not fit into constant memory\n",
                kernelArrayLength);
        exit(1);
    }
    else {
        wbCheck(cudaMemcpyToSymbol(c_weights, host_k, kernelArrayLength *
                                   sizeof(host_k[0])));
    }

    cudaMalloc(&device_x, inputArrayLength * sizeof(*device_x));
    cudaMalloc(&device_y, outputArrayLength * sizeof(*device_y));
    cudaMalloc(&device_k, kernelArrayLength * sizeof(*device_k));

    cudaMemcpy(device_x, host_x, inputArrayLength * sizeof(*host_x), cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, host_k, kernelArrayLength * sizeof(*host_k), cudaMemcpyHostToDevice);

    // Set the kernel dimensions and call the kernel
    dim3 dimGrid(ceil((float)W_out / TILE_WIDTH), ceil((float)H_out / TILE_WIDTH), M);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    // Copy the output back to host
    cudaMemcpy(host_y, device_y, outputArrayLength * sizeof(*device_y), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_k);

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
