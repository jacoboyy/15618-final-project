#include <cuda_runtime.h>
#include <algorithm>
#include <thrust/sort.h>

__global__ void compute_subtree(
    const int* parent,        // Parent indices of nodes
    const float* node_values, // Node values
    float* subtree_values,    // Subtree values (output)
    int* topk_indices,        // Top-k indices for each subtree
    float* topk_values,       // Top-k values for each subtree
    int num_nodes,            // Number of nodes in the tree
    int k                     // Top-k branches to keep
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_nodes) return;

    // Compute subtree value by summing from root to the node
    int current = idx;
    float value = 0.0f;

    while (current != -1) { // Assuming root has parent = -1
        value += node_values[current];
        current = parent[current];
    }

    subtree_values[idx] = value;

    // Sync within the block to ensure subtree_values are ready
    __syncthreads();

    // Perform top-k selection (shared memory per block)
    extern __shared__ float shared[];
    float* shared_values = shared;             // Shared memory for values
    int* shared_indices = (int*)&shared[k];    // Shared memory for indices

    if (threadIdx.x < k) {
        shared_values[threadIdx.x] = -INFINITY; // Initialize top-k values
        shared_indices[threadIdx.x] = -1;       // Initialize top-k indices
    }

    __syncthreads();

    // Update top-k in shared memory
    if (threadIdx.x < num_nodes) {
        float val = subtree_values[idx];
        if (val > shared_values[k - 1]) {
            shared_values[k - 1] = val;
            shared_indices[k - 1] = idx;
            thrust::sort_by_key(shared_values, shared_values + k, shared_indices, thrust::greater<float>());
        }
    }

    __syncthreads();

    // Write top-k back to global memory
    if (threadIdx.x < k) {
        topk_values[blockIdx.x * k + threadIdx.x] = shared_values[threadIdx.x];
        topk_indices[blockIdx.x * k + threadIdx.x] = shared_indices[threadIdx.x];
    }
}
