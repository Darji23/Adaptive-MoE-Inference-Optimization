#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {

/**
 * MoE Router Profiling CUDA Kernel
 *
 * Accumulates activation counts of experts per layer.
 *
 * Parameters:
 *   routing_indices:   Integer tensor of shape [batch_size, seq_len, top_k]
 *                      Contains the indices of experts selected for each token.
 *   expert_counts:     Integer tensor of shape [num_layers, num_experts]
 *                      Accumulates routing frequency for each expert.
 *   num_layers:        Total layers in the network
 *   num_experts:       Total experts per MoE layer
 *   top_k:             Number of experts selected per token (e.g. 2 for Nemotron)
 *   total_tokens:      batch_size * seq_len
 */
__global__ void profile_router_decisions_kernel(
    const int* __restrict__ routing_indices,
    int*       __restrict__ expert_counts,
    int num_layers,
    int num_experts,
    int top_k,
    int total_tokens
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int layer_idx = blockIdx.y;

    if (token_idx >= total_tokens || layer_idx >= num_layers) {
        return;
    }

    // Offset in routing indices: [layer, token, top_k] or [token, top_k] if shared routing
    // Typically routing decisions are stored per layer. Let's assume indices are:
    // [num_layers, total_tokens, top_k]
    int offset = layer_idx * (total_tokens * top_k) + token_idx * top_k;

    for (int k = 0; k < top_k; ++k) {
        int expert_id = routing_indices[offset + k];
        if (expert_id >= 0 && expert_id < num_experts) {
            // Atomically increment counts for this expert in this layer
            atomicAdd(&expert_counts[layer_idx * num_experts + expert_id], 1);
        }
    }
}

// Host wrapper to launch MoE profiling kernel
void launch_moe_router_profiler(
    const int* routing_indices,
    int* expert_counts,
    int num_layers,
    int num_experts,
    int top_k,
    int total_tokens,
    cudaStream_t stream
) {
    dim3 block(256, 1, 1);
    dim3 grid((total_tokens + block.x - 1) / block.x, num_layers, 1);

    profile_router_decisions_kernel<<<grid, block, 0, stream>>>(
        routing_indices,
        expert_counts,
        num_layers,
        num_experts,
        top_k,
        total_tokens
    );
}

} // extern "C"
