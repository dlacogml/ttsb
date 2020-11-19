#ifndef TTSB_CONV1_HH
#define TTSB_CONV1_HH
#include <cstddef>
#include <cmath>
#include <iostream>

#ifndef wbCheck
#define wbCheck(stmt)                                                         \
  do {                                                                        \
    cudaError_t err = stmt;                                                   \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;    \
      std::cerr << "Failed to run stmt " << #stmt << std::endl;               \
      exit(-1);                                                               \
    }                                                                         \
  } while (0)
#endif


struct L1Config {
    /// Batch Size
    static constexpr size_t B = 10000;
    /// Number of output channels
    static constexpr size_t M = 4;
    /// Number of input channels
    static constexpr size_t C = 1;
    /// Number of rows in input channel
    static constexpr size_t H = 86;
    /// Number of columns in input channel
    static constexpr size_t W = 86;
    /// Width of convolution kernel
    static constexpr size_t K = 7;
    /// Number of rows of output channels
    static constexpr int H_out = H-(K-1);
    /// Number of columns of output channels
    static constexpr int W_out = W-(K-1);
    using l1_t = float;
    /// Number of Kernel weights
    static constexpr size_t KernelLength = K*K*M*C;
    /// Size of Kernel weights
    static constexpr size_t KernelSize = KernelLength * sizeof(l1_t);

    // CUDA Kernel parameters
    /// Width of cuda blocks in terms of input pixels (and number launched)
    static constexpr size_t block_width = 26;
    /// Width of cuda blocks in terms of output pixels
    static constexpr size_t outblock_width = block_width - (K-1);
    /// Number of blocks needed to cover an entire output channel vertically
    static constexpr size_t blocks_per_channel_h =
        (H_out + (outblock_width-1)) / outblock_width;
    /// Number of blocks needed to cover an entire output channel horizontally
    static constexpr size_t blocks_per_channel_w =
        (W_out + (outblock_width-1)) / outblock_width;
    /// Number of blocks needed to cover an entire output channel
    static constexpr size_t blocks_per_channel =
        blocks_per_channel_h*blocks_per_channel_w;
    /// Number of inputs (partially) processed by a single thread block
    static constexpr size_t mini_batch_size = 10;
    /// Number of mini batches needed across the entire image space
    static constexpr size_t num_mini_batch = B / mini_batch_size;
    
    /// @brief Compute the index of an input pixel based on
    /// @param image --- its place in the batch
    /// @param row   --- its row in the single input channel
    /// @param col   --- its column in the single input channel
    /// @return the index into the one-dim array
    __device__
    static constexpr size_t xidx(int image, int row, int col)
    {
        return image*C*H*W + row*W + col;
    }
    /// @brief Compute the index of an output pixel based on
    /// @param image --- its place in the batch
    /// @param och   --- the pixel's channel
    /// @param row   --- its row in the output channel
    /// @param col   --- its column in the output channel
    /// @return the index into the one-dim array
    __host__ __device__
    static constexpr size_t yidx(int image, int och, int row, int col)
    {
        return image*M*H_out*W_out + och*H_out*W_out + row*W_out + col;
    }
    /// @brief Compute the index of a filter weight based on
    /// @param och --- the output channel of the filter
    /// @param row --- the row of the weight in the filter
    /// @param col --- the column of the weight in the filter
    /// @return the index into the one-dim array
    __device__
    static constexpr size_t kidx(int och, int row, int col)
    {
        return och*K*K + row*K + col;
    }
    __host__ __device__
    static constexpr int get_row(int blockIdxZ, int threadIdxY)
    {
        return (blockIdxZ / blocks_per_channel_w) * L1Config::outblock_width +
            threadIdxY;
    }
    __host__ __device__
    static constexpr int get_col(int blockIdxZ, int threadIdxX)
    {
        return (blockIdxZ % blocks_per_channel_w) * L1Config::outblock_width +
            threadIdxX;
    }
};

__constant__ L1Config::l1_t c_w1[L1Config::KernelLength];

__global__
void
conv1_forward_kernel(const float* _x, float* _y)
{
    auto y = [_y](int image, int och, int row, int col) -> float& {
        return _y[L1Config::yidx(image, och, row, col)];
    };
    auto x = [_x](int image, int row, int col) -> const float {
        return _x[L1Config::xidx(image, row, col)];
    };
    auto k = [](int och, int row, int col) -> const float {
        return c_w1[L1Config::kidx(och, row, col)];
    };

    __shared__ float sm[L1Config::mini_batch_size][L1Config::block_width]
                                                    [L1Config::block_width];

    const int image_start = blockIdx.x * L1Config::mini_batch_size;
    const int och = blockIdx.y;
    const int srow = threadIdx.y; /// Row in shared memory
    const int scol = threadIdx.x; /// Column in shared memory
    const int row = L1Config::get_row(blockIdx.z, srow);
    const int col = L1Config::get_col(blockIdx.z, scol);

    // Load the data into shared memory!
    for (int b = 0; b < L1Config::mini_batch_size; ++b) {
        const int image = image_start + b;
        sm[b][srow][scol] = x(image, row, col);
    }
    __syncthreads_and(true);

    // Data is loaded and we are ready to compute
    if ((srow < L1Config::outblock_width) &&
        (scol < L1Config::outblock_width))
    {
        for (int b = 0; b < L1Config::mini_batch_size; ++b) {
            const int image = image_start + b;
            float acc = 0.0f;
            for (int p = 0; p < L1Config::K; ++p) {
            for (int q = 0; q < L1Config::K; ++q) {
                float xval = sm[b][srow+p][scol+q];
                float kval = k(och, p, q);
                acc += xval * kval;
            }}
            y(image, och, row, col) = acc;
        }
    }
}

__host__
void
do_layer1(
        float* host_y, const float* host_x, const float* host_k, size_t& rv
)
{
    static constexpr size_t B     = L1Config::B;
    static constexpr size_t M     = L1Config::M;
    static constexpr size_t C     = L1Config::C;
    static constexpr size_t H     = L1Config::H;
    static constexpr size_t W     = L1Config::W;
    static constexpr size_t K     = L1Config::K;
    static constexpr size_t H_out = L1Config::H_out;
    static constexpr size_t W_out = L1Config::W_out;
    static constexpr size_t inputlen = H*W*C*B;
    static constexpr size_t inputsize = inputlen * sizeof(*host_x);
    static constexpr size_t outputlen = H_out*W_out*M*B;
    static constexpr size_t outputsize = outputlen * sizeof(*host_y);
    rv = outputsize;
    float* device_x;
    float* device_y;

    // Allocate device memory
    wbCheck(cudaMalloc(&device_x, inputsize));
    wbCheck(cudaMalloc(&device_y, outputsize));

    // Write device memory
    wbCheck(cudaMemcpy(device_x, host_x, inputsize, cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpyToSymbol(c_w1, host_k, L1Config::KernelSize, 0,
                               cudaMemcpyHostToDevice));
    dim3 BlockDim(L1Config::block_width, L1Config::block_width);
    dim3 GridDim(L1Config::num_mini_batch, L1Config::M,
                 L1Config::blocks_per_channel);

    conv1_forward_kernel<<<GridDim, BlockDim>>>(device_x, device_y);
    cudaDeviceSynchronize();
    wbCheck(cudaMemcpy(host_y, device_y, outputsize, cudaMemcpyDeviceToHost));
    wbCheck(cudaFree(device_x));
    wbCheck(cudaFree(device_y));
}

#endif
