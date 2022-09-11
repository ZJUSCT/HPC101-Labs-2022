#include <cuda.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

const int alignment = 32; // 32 byte alignment
const int size = 100;
const int kernel = 3;  // odd
const int batch_size = 128;
const int in_channel = 128;
const int out_channel = 128;

#define InitRandom()                         \
  std::random_device r;                      \
  std::default_random_engine generator(r()); \
  std::uniform_int_distribution<> distribution(0, 255);

#define a(_n, _x, _y, _c) a[(_n) * size * size * in_channel + (_x) * size * in_channel + (_y) * in_channel + (_c)]
#define w(_x, _y, _ci, _co) w[(_x) * kernel * in_channel * out_channel + (_y) * in_channel * out_channel + (_ci) * out_channel + (_co)]
#define b(_n, _x, _y, _c) b[(_n) * size * size * out_channel + (_x) * size * out_channel + (_y) * out_channel + (_c)]
#define CUDA_CALL(func)                                         \
  {                                                             \
    cudaError_t e = (func);                                     \
    if (!(e == cudaSuccess || e == cudaErrorCudartUnloading))   \
    {                                                           \
        fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));   \
	      abort();                                                \
    }                                                           \
  }


/// \brief Generate [N, H, W, C] input tensor and [H, W, I, O] kernel tensor.
void Generate(uint8_t *const a, uint8_t *const w) {
#pragma omp parallel for
  // Batch dimension.
  for (int s = 0; s < batch_size; ++s) {
      InitRandom();
      // Height dimension.
      for (int i = 0; i < size; ++i)
        // Width dimension.
        for (int j = 0; j < size; ++j) {
          const int channel_lower = s * size * size * in_channel
                                  + i * size * in_channel
                                  + j * in_channel;
          const int channel_upper = channel_lower + in_channel; 
          // Channel dimension.
          for (int c = channel_lower; c < channel_upper; ++c)
            a[c] = distribution(generator);
        }
  }
#pragma omp parallel for
  for (int i = 0; i < kernel; ++i) {
    InitRandom();
    for (int j = 0; j < kernel; ++j) 
      for (int CI = 0; CI < in_channel; ++CI) {
        const int channel_lower = i * kernel * in_channel * out_channel
                                + j * in_channel * out_channel
                                + CI * out_channel;
        const int channel_upper = channel_lower + out_channel;
        for (int CO = channel_lower; CO < channel_upper; ++CO) 
          w[CO] = distribution(generator);
      }
  }
}

void conv2d_cpu_kernel(const uint8_t *__restrict__ a, 
                       const uint8_t *__restrict__ w, 
                       uint8_t *__restrict__ b) {
#pragma omp parallel for
  for (int s = 0; s < batch_size; ++s) {
    size_t output_bytes = ((out_channel * sizeof(uint8_t)) + (size_t)alignment - 1) & ~((size_t)alignment -1); 
    uint8_t *packedB = static_cast<uint8_t *>(malloc(output_bytes));

    size_t input_bytes = ((kernel * kernel * in_channel * sizeof(uint8_t)) + (size_t)alignment - 1) & ~((size_t)alignment - 1);
    uint8_t *packedA = static_cast<uint8_t *>(malloc(input_bytes));

    for (int i = 0; i < size; ++i)
      for (int j = 0; j < size; ++j) {
        // Collected needed input data,
        // Start from A[s, i - kernel / 2, j - kernel / 2, 0].
        int x = i - kernel / 2;
        int y = j - kernel / 2;
        int input_index = s * size * size * in_channel
                        + x * size * in_channel
                        + y * in_channel;
        memset(packedA, 0, input_bytes);
        int A_buffer_index = 0;
        for (int kh = 0; kh < kernel; ++kh) {
          for (int kw = 0; kw < kernel; ++ kw) {
            if (!(x < 0 || x >= size || y < 0 || y >= size)) {
              memcpy(packedA + A_buffer_index, a + input_index, in_channel * sizeof(uint8_t));
            }
            else {
              memset(packedA + A_buffer_index, 0, in_channel * sizeof(uint8_t));
            }
            y++;
            A_buffer_index += in_channel;
            input_index += in_channel;
          }
          x++;
          y -= kernel;
          input_index = input_index - kernel * in_channel + size * in_channel;
        }

        // Start from B[s, i, j, 0]
        int output_index = s * size * size * out_channel 
                         + i * size * out_channel 
                         + j * out_channel;                 
        memset(packedB, 0, output_bytes);

        // Start from W[0, 0, 0, 0]
        int kernel_index = 0;
        A_buffer_index = 0;
        // Convolution 2D computation.
        // iterate over each in_channel of input tensor,
        // and accumulate contribution to output tensor.
        for (int N = 0; N < kernel * kernel; ++N) {
          for (int CI = 0; CI < in_channel; ++CI) {
            for (int CO = 0; CO < out_channel; ++CO) {
              packedB[CO] +=  packedA[A_buffer_index] * w[kernel_index];
              kernel_index++; // move to next output channel.
            }
            A_buffer_index++;
          }
        }
        memcpy(b + output_index, packedB, sizeof(uint8_t) * out_channel);
      }
    free(packedA);
    free(packedB);
  }
}

void Check(const uint8_t *const a, const uint8_t *const w, uint8_t *const b) {
  auto b_std = new uint8_t[batch_size * size * size * out_channel];
  std::cout << "Conv2d CPU Kernel Start... \n";
  conv2d_cpu_kernel(a, w, b_std);
  std::cout << "Checking Results... \n";
  size_t N = batch_size * size * size * out_channel;
  for (size_t i = 0; i < N; ++i) {
    if (b[i] != b_std[i]) {
      std::cout << "\x1b[31m"
                   "Wrong Answer"
                   "\x1b[0m"
                   " at "
                << i << std::endl;
      std::cout << "expected " << (int)b_std[i] << " but found " << (int)b[i]
                << std::endl;
      delete[] b_std;
      return;
    }
  }
  std::cout << "\x1b[32m"
               "Correct"
               "\x1b[0m"
            << std::endl;

  delete[] b_std;
}

const int block_size = 16;
/// \brief Do Conv2d with NHWC Input with HWIO Kernel, and NHWC output 
__global__ void conv2d_cuda_kernel(const uint8_t *__restrict__ a, 
                                   const uint8_t *__restrict__ w, 
                                   uint8_t *__restrict__ b) 
{
  const int i = blockIdx.x * block_size + threadIdx.x;
  const int j = blockIdx.y * block_size + threadIdx.y;
  if (i < size && j < size) {
    for (int s = 0; s < batch_size; ++s) {
      for (int CO = 0; CO < out_channel; ++CO) {
        uint8_t conv = 0;
        // Conv2d for a single pixel, single output channel.
        for (int CI = 0; CI < in_channel; ++CI) {
          int x = i - kernel / 2, y = j - kernel / 2;
          for (int k = 0; k < kernel; ++k) {
            for (int l = 0; l < kernel; ++l) {
              if (!(x < 0 || x >= size || y < 0 || y >= size)) {
                conv += a(s, x, y, CI) * w(k, l, CI, CO);
              }
              y++;
            }
            x++;
            y -= kernel;
          }
        }
        // Write back to b.
        b(s, i, j, CO) = conv;
      }
    }
  }
}

// naive and shit
// only for testing correctness and precision
void conv_cuda(const uint8_t *const a, const uint8_t *const w, uint8_t *const b,
               cudaEvent_t *start_e, cudaEvent_t *stop_e) 
{
  uint8_t *a_kernel, *w_kernel, *b_kernel;
  CUDA_CALL(cudaMalloc(&a_kernel, batch_size * size * size * in_channel * sizeof(uint8_t)));
  CUDA_CALL(cudaMemcpy(a_kernel, a, batch_size * size * size * in_channel * sizeof(uint8_t),
             cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&w_kernel, kernel * kernel * in_channel * out_channel * sizeof(uint8_t)));
  CUDA_CALL(cudaMemcpy(w_kernel, w, kernel * kernel * in_channel * out_channel * sizeof(uint8_t),
             cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&b_kernel, batch_size * size * size * out_channel * sizeof(uint8_t)));
  // Start Timer.
  cudaEventRecord(*start_e);
  // Run Conv2d Kernel,
  // Timer for computation cuda kernel.
  dim3 grid((size + block_size - 1) / block_size,
            (size + block_size - 1) / block_size);
  dim3 block(block_size, block_size);
  // @note: you can also use CUDA API to launch a cuda kernel function,
  // __host__ cudaError_t cudaLaunchKernel;
  conv2d_cuda_kernel<<<grid, block>>>(a_kernel, w_kernel, b_kernel);
  cudaError_t kernel_err = cudaGetLastError();
  if (kernel_err != cudaSuccess) {
  	printf("CUDA Kernel: %s", cudaGetErrorString(kernel_err));
	abort();
  }
  cudaDeviceSynchronize();
  // Stop Timer
  cudaEventRecord(*stop_e);
  cudaEventSynchronize(*stop_e);

  CUDA_CALL(cudaMemcpy(b, b_kernel, batch_size * size * size * out_channel * sizeof(uint8_t),
             cudaMemcpyDeviceToHost));
  cudaFree(a_kernel);
  cudaFree(w_kernel);
  cudaFree(b_kernel);
}

int main() {
  auto a = new uint8_t[batch_size * size * size * in_channel];
  auto w = new uint8_t[kernel * kernel * in_channel * out_channel];
  auto b = new uint8_t[batch_size * size * size * out_channel];
  std::cout << "Generating input and kernel tensor... \n";
  Generate(a, w);

  cudaEvent_t start_e, stop_e;
  cudaEventCreate(&start_e);
  cudaEventCreate(&stop_e);

  // Conv(a, w, b);
  std::cout << "Conv2d Cuda Kernel Start... \n";
  conv_cuda(a, w, b, &start_e, &stop_e);

  std::cout << "Verifying... \n";
  Check(a, w, b);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_e, stop_e);
  std::cout << milliseconds << " milliseconds" << std::endl;
  cudaEventDestroy(start_e);
  cudaEventDestroy(stop_e);

  // Output(a, w, b);
  delete[] a;
  delete[] w;
  delete[] b;
  return 0;
}
