#ifndef TTSB_CONV2_HH
#define TTSB_CONV2_HH
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
struct L2Config {
    /// Batch Size
    static constexpr size_t B = 10000;
    /// Number of output channels
    static constexpr size_t M = 16;
    /// Number of input channels
    static constexpr size_t C = 4;
    /// Number of rows in input channel
    static constexpr size_t H = 40;
    /// Number of columns in input channel
    static constexpr size_t W = 40;
    /// Width of convolution kernel
    static constexpr size_t K = 7;
    /// Number of rows of output channels
    static constexpr int H_out = H-(K-1);
    /// Number of columns of output channels
    static constexpr int W_out = W-(K-1);
    using l2_t = float;
    /// Number of Kernel weights
    static constexpr size_t KernelLength = K*K*M*C;
    /// Size of Kernel weights
    static constexpr size_t KernelSize = KernelLength * sizeof(l2_t);

    // CUDA Kernel parameters
    /// Width of cuda blocks in terms of input pixels (and number launched)
    static constexpr size_t block_width = 23;
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
    static constexpr size_t mini_batch_size = 5;
    /// Number of mini batches needed across the entire image space
    static constexpr size_t num_mini_batch = B / mini_batch_size;
    
    /// @brief Compute the index of an input pixel based on
    /// @param image --- its place in the batch
    /// @param ich   --- its input channel
    /// @param row   --- its row in the single input channel
    /// @param col   --- its column in the single input channel
    /// @return the index into the one-dim array
    __device__
    static constexpr size_t xidx(int image, int ich, int row, int col)
    {
        return image*C*H*W + ich*H*W + row*W + col;
    }
    /// @brief Compute the index of an output pixel based on
    /// @param image --- its place in the batch
    /// @param och   --- the pixel's channel
    /// @param row   --- its row in the output channel
    /// @param col   --- its column in the output channel
    /// @return the index into the one-dim array
    __device__
    static constexpr size_t yidx(int image, int och, int row, int col)
    {
        return image*M*H_out*W_out + och*H_out*W_out + row*W_out + col;
    }
    /// @brief Compute the index of a filter weight based on
    /// @param och --- the output channel of the filter
    /// @param ich --- the input channel of the filter
    /// @param row --- the row of the weight in the filter
    /// @param col --- the column of the weight in the filter
    /// @return the index into the one-dim array
    __device__
    static constexpr size_t kidx(int och, int ich, int row, int col)
    {
        return och*C*K*K + ich*K*K + row*K + col;
    }
    __device__
    static constexpr int get_row(int blockIdxZ, int threadIdxX)
    {
        return (blockIdxZ / blocks_per_channel_w) * L2Config::outblock_width +
            threadIdxX;
    }
    __device__
    static constexpr int get_col(int blockIdxZ, int threadIdxY)
    {
        return (blockIdxZ % blocks_per_channel_w) * L2Config::outblock_width +
            threadIdxY;
    }
};

__constant__ L2Config::l2_t c_w2[L2Config::KernelLength];

__global__
void
_conv2_forward_kernel(const float* _x, float* _y)
{
    auto y = [_y](int image, int och, int row, int col) -> float& {
        return _y[L2Config::yidx(image, och, row, col)];
    };
    auto k = [](int och, int ich, int row, int col) -> const float {
        return c_w2[L2Config::kidx(och, ich, row, col)];
    };
    auto x = [_x](int image, int ich, int row, int col) -> const float {
        return _x[L2Config::xidx(image, ich, row, col)];
    };
    const int image_start = blockIdx.x * L2Config::mini_batch_size;
    const int och  = blockIdx.y;
    const int srow = threadIdx.y; /// Row in shared memory
    const int scol = threadIdx.x; /// Column in shared memory
    const int row  = L2Config::get_row(blockIdx.z, srow);
    const int col  = L2Config::get_col(blockIdx.z, scol);
    __shared__ float sm[L2Config::mini_batch_size][L2Config::C]
                            [L2Config::block_width][L2Config::block_width];

    // Load the data into shared memory!
    {
        for (int b = 0; b < L2Config::mini_batch_size; ++b) {
        for (int ic = 0; ic < L2Config::C; ++ic) {
            const int image = image_start + b;
            sm[b][ic][srow][scol] = x(image, ic, row, col);
        }}
    }
    __syncthreads_and(true);


    if (srow < L2Config::outblock_width &&
        scol < L2Config::outblock_width)
    {
        for (int b = 0; b < L2Config::mini_batch_size; ++b) {
            const int image = b + image_start;
            float acc = 0.0f;
            for (int c = 0; c < L2Config::C; ++c) {
                for (int p = 0; p < L2Config::K; ++p) {
                for (int q = 0; q < L2Config::K; ++q) {
                    // const float xval = x(image, c, row+p, col+q);
                    const float xval = sm[b][c][srow+p][scol+q];
                    const float kval = k(och, c, p, q);
                    acc += xval * kval;
                }}
            }
            y(image, och, row, col) = acc;
        }
    }

}

__global__
void
conv2_forward_kernel(const float* _x, float* _y)
{
    auto y = [_y](int image, int och, int row, int col) -> float& {
        return _y[L2Config::yidx(image, och, row, col)];
    };
    auto k = [](int och, int ich, int row, int col) -> const float {
        return c_w2[L2Config::kidx(och, ich, row, col)];
    };
    auto x = [_x](int image, int ich, int row, int col) -> const float {
        return _x[L2Config::xidx(image, ich, row, col)];
    };

    __shared__ float sm[L2Config::mini_batch_size][L2Config::C]
                            [L2Config::block_width][L2Config::block_width];

    const int image_start = blockIdx.x * L2Config::mini_batch_size;
    const int och  = blockIdx.y;
    const int srow = threadIdx.y; /// Row in shared memory
    const int scol = threadIdx.x; /// Column in shared memory
    const int row  = L2Config::get_row(blockIdx.z, srow);
    const int col  = L2Config::get_col(blockIdx.z, scol);

    // Load the data into shared memory!
    {
        for (int b = 0; b < L2Config::mini_batch_size; ++b) {
        for (int ic = 0; ic < L2Config::C; ++ic) {
            const int image = image_start + b;
            sm[b][ic][srow][scol] = x(image, ic, row, col);
        }}
    }
    __syncthreads_and(true);

    // Data is loaded and we are ready to compute
    if ((srow < L2Config::outblock_width) &&
        (scol < L2Config::outblock_width))
    {
        for (int b = 0; b < L2Config::mini_batch_size; ++b) {
            const int image = image_start + b;
            float acc = 0.0f;
            for (int ich = 0; ich < L2Config::C; ++ich) {
            for (int p = 0; p < L2Config::K; ++p) {
            for (int q = 0; q < L2Config::K; ++q) {
                const float kval = k(och, ich, p, q);
                const float xval = sm[b][ich][srow+p][scol+q];
                // const float xval =
                //     _x[
                //         b*L2Config::C*L2Config::H*L2Config::W +
                //         ich*L2Config::H*L2Config::W +
                //         row*L2Config::W +
                //         col
                //     ];
                acc +=  xval * kval;
            }}}
            y(image, och, row, col) = acc;
        }
    }
}

__host__
void
do_layer2(
        float* host_y, const float* host_x, const float* host_k
)
{
    static constexpr size_t B     = L2Config::B;
    static constexpr size_t M     = L2Config::M;
    static constexpr size_t C     = L2Config::C;
    static constexpr size_t H     = L2Config::H;
    static constexpr size_t W     = L2Config::W;
    static constexpr size_t K     = L2Config::K;
    static constexpr size_t H_out = L2Config::H_out;
    static constexpr size_t W_out = L2Config::W_out;
    static constexpr size_t inputlen = H*W*C*B;
    static constexpr size_t inputsize = inputlen * sizeof(*host_x);
    static constexpr size_t outputlen = H_out*W_out*M*B;
    static constexpr size_t outputsize = outputlen * sizeof(*host_y);
    float* device_x;
    float* device_y;

    // Allocate device memory
    wbCheck(cudaMalloc(&device_x, inputsize));
    wbCheck(cudaMalloc(&device_y, outputsize));

    // Write device memory
    wbCheck(cudaMemcpy(device_x, host_x, inputsize, cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpyToSymbol(c_w2, host_k, L2Config::KernelSize, 0,
                               cudaMemcpyHostToDevice));
    dim3 BlockDim(L2Config::block_width, L2Config::block_width);
    dim3 GridDim(L2Config::num_mini_batch, L2Config::M,
                 L2Config::blocks_per_channel);

    _conv2_forward_kernel<<<GridDim, BlockDim>>>(device_x, device_y);
    cudaDeviceSynchronize();

    wbCheck(cudaMemcpy(host_y, device_y, outputsize, cudaMemcpyDeviceToHost));
    wbCheck(cudaFree(device_x));
    wbCheck(cudaFree(device_y));
}

#endif
