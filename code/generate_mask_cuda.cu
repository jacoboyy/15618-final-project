#include <torch/extension.h>

__global__ void generate_mask_cuda(int *parent_indices, int num_nodes, int *mask, int mask_size) {
    int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= num_nodes) return;

    int row_start = node_id * mask_size;
    mask[row_start + node_id] = 1;

    int current_node = parent_indices[node_id];
    while (current_node != -1) {
        mask[row_start + current_node] = 1;
        current_node = parent_indices[current_node];
    }
}

torch::Tensor generate_mask(torch::Tensor parent_indices, int mask_size) {
    int num_nodes = parent_indices.size(0);
    auto mask = torch::zeros({num_nodes, mask_size}, torch::dtype(torch::kInt32).device(parent_indices.device()));

    int *parent_indices_ptr = parent_indices.data_ptr<int>();
    int *mask_ptr = mask.data_ptr<int>();

    int threads_per_block = 256;
    int blocks_per_grid = (num_nodes + threads_per_block - 1) / threads_per_block;

    generate_mask_cuda<<<blocks_per_grid, threads_per_block>>>(parent_indices_ptr, num_nodes, mask_ptr, mask_size);

    return mask;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_mask", &generate_mask, "Generate Mask (CUDA)");
} 