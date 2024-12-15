#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

// CUDA kernel to generate mask
__global__ void generate_mask_cuda(int *parent_indices, int num_nodes, int *mask, int mask_size) {
    int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= num_nodes) return;

    int row_start = node_id * mask_size;
    mask[row_start + node_id] = 1;  // Mark self-relation

    // Traverse ancestors and mark their relations in the mask
    int current_node = parent_indices[node_id];
    while (current_node != -1) {
        mask[row_start + current_node] = 1;
        current_node = parent_indices[current_node];
    }
}

// Function to call the CUDA kernel
torch::Tensor generate_mask(torch::Tensor parent_indices, int mask_size) {
    // Input validation
    TORCH_CHECK(parent_indices.device().is_cuda(), "parent_indices must be a CUDA tensor");
    TORCH_CHECK(parent_indices.dim() == 1, "parent_indices must be a 1D tensor");
    TORCH_CHECK(parent_indices.scalar_type() == torch::kInt32, "parent_indices must be of type int32");

    int num_nodes = parent_indices.size(0);

    // Allocate output tensor on the same device as the input
    auto mask = torch::zeros({num_nodes, mask_size}, torch::dtype(torch::kInt32).device(parent_indices.device()));

    // Get raw pointers to device memory
    int *parent_indices_ptr = parent_indices.data_ptr<int>();
    int *mask_ptr = mask.data_ptr<int>();

    // Configure CUDA kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_nodes + threads_per_block - 1) / threads_per_block;

    // Launch CUDA kernel
    generate_mask_cuda<<<blocks_per_grid, threads_per_block>>>(parent_indices_ptr, num_nodes, mask_ptr, mask_size);

    // Synchronize and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel execution failed: ") + cudaGetErrorString(err));
    }

    return mask;
}

// PyTorch extension binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_mask", &generate_mask, "Generate Mask (CUDA)",
          py::arg("parent_indices"), py::arg("mask_size"));
}
