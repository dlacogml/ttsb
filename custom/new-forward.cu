#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define LAYER1_WIDTH 8
#define LAYER2_WIDTH 20

__global__ void conv_forward_kernel(float* y, const float* x, const float* k, const int B, const int M, const int C, const int H, const int W, const int K)
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
    extern __shared__ float s[];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int s_width = blockDim.x + K - 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define s3d(i2, i1, i0) s[(i2) * (s_width * s_width) + (i1) * (s_width) + i0]

    // Insert your GPU convolution kernel code here
    const int m = blockIdx.z;
    const int h = blockIdx.y*blockDim.y + threadIdx.y;
    const int w = blockIdx.x*blockDim.x + threadIdx.x;

    for (int b = 0; b < B; ++b) {
        // Copy input to shared memory
        for (int i = 0; i * blockDim.y < s_width; ++i) {
            for (int j = 0; j * blockDim.x < s_width; ++j) {
                int s_h = i * blockDim.y + threadIdx.y;
                int s_w = j * blockDim.x + threadIdx.x;
                if (s_h < s_width && s_w < s_width) {
                    int i_h = i * blockDim.y + h;
                    int i_w = j * blockDim.x + w;
                    for (int c = 0; c < C; ++c) {
                        if (i_h < H && i_w < W) {
                            s3d(c, s_h, s_w) = x4d(b, c, i_h, i_w);
                        } else {
                            s3d(c, s_h, s_w) = 0.0f;
                        }
                    }
                }
            }
        }

        __syncthreads();

        if (h < H_out && w < W_out) {
            float acc = 0.0f;
            for (int c = 0; c < C; ++c) {
                for (int p = 0; p < K; ++p) {
                    for (int q = 0; q < K; ++q) {
                        acc += s3d(c, threadIdx.y + p, threadIdx.x + q) * k4d(m, c, p, q);
                        // acc += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
                    }
                }
            }
            y4d(b, m, h, w) = acc;
        }

        __syncthreads();
    }


#undef y4d
#undef x4d
#undef k4d
#undef s3d
}

__host__ void GPUInterface::conv_forward_gpu(float* host_y, const float* host_x, const float* host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Declare relevant device pointers
    float* device_x;
    float* device_y;
    float* device_k;

    // Allocate memory and copy over the relevant data structures to the GPU
    const unsigned int H_out = H - K + 1;
    const unsigned int W_out = W - K + 1;

    unsigned int inputArrayLength = B*C*H*W;
    unsigned int outputArrayLength = B*M*H_out*W_out;
    unsigned int kernelArrayLength = M*C*K*K;

    cudaMalloc(&device_x, inputArrayLength * sizeof(*device_x));
    cudaMalloc(&device_y, outputArrayLength * sizeof(*device_y));
    cudaMalloc(&device_k, kernelArrayLength * sizeof(*device_k));

    cudaMemcpy(device_x, host_x, inputArrayLength * sizeof(*host_x), cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, host_k, kernelArrayLength * sizeof(*host_k), cudaMemcpyHostToDevice);

    // Set the kernel dimensions and call the kernel
    unsigned int tile_width = (C == 1) ? LAYER1_WIDTH : LAYER2_WIDTH;
    dim3 dimGrid(ceil((float)W_out / tile_width), ceil((float)H_out / tile_width), M);
    dim3 dimBlock(tile_width, tile_width, 1);

    unsigned int s_size = C * (tile_width + K - 1) * (tile_width + K - 1) * sizeof(float);
    conv_forward_kernel<<<dimGrid, dimBlock, s_size>>>(device_y, device_x, device_k, B, M, C, H, W, K);

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
