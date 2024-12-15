#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void matmul_add_kernel(
    const float *input, 
    const float *weights, 
    const float *bias, 
    float *output, 
    int input_size, 
    int output_size, 
    int batch_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Batch index
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // Output index

    if (row >= batch_size || col >= output_size) return;

    float sum = 0.0f;
    for (int i = 0; i < input_size; ++i) {
        sum += input[row * input_size + i] * weights[col * input_size + i];
    }
    output[row * output_size + col] = sum + bias[col];
}

void tensor_parallel_linear_forward(
    torch::Tensor input, 
    torch::Tensor weights, 
    torch::Tensor bias, 
    torch::Tensor output, 
    int input_size, 
    int output_size
) {
    // Ensure tensors are on CUDA
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "Weights must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "Output must be a CUDA tensor");

    int batch_size = input.size(0);

    // Define block and grid dimensions
    dim3 threads_per_block(16, 16);
    dim3 num_blocks(
        (batch_size + threads_per_block.x - 1) / threads_per_block.x,
        (output_size + threads_per_block.y - 1) / threads_per_block.y
    );

    // Launch the kernel
    matmul_add_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), 
        weights.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        input_size, 
        output_size, 
        batch_size
    );

    // Synchronize the device
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }
}
// Function declarations
void tensor_parallel_linear_forward(
    torch::Tensor input, 
    torch::Tensor weights, 
    torch::Tensor bias, 
    torch::Tensor output, 
    int input_size, 
    int output_size
);

// PyTorch module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tensor_parallel_linear_forward, "Tensor Parallel Linear Forward",
          py::arg("input"), py::arg("weights"), py::arg("bias"), py::arg("output"),
          py::arg("input_size"), py::arg("output_size"));
}