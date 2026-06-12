#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {

/**
 * Expert-Conditioned KV Cache Eviction Policy Kernel
 *
 * This kernel combines attention scores (recency/relevance) with MoE routing
 * decisions. It calculates an eviction priority score for each token in the context.
 *
 * Priority Score Formula:
 *   Priority = (1.0 - expert_weight) * Mean_Attention_Score - expert_weight * Expert_Load_Signal
 *
 * Tokens with lower priority scores are evicted first.
 *
 * Parameters:
 *   attention_scores:   Float tensor of shape [batch_size, num_heads, seq_len]
 *   expert_activations: Integer tensor of shape [batch_size, seq_len, num_experts].
 *                       Contains binary indicators (0 or 1) indicating if token was routed to expert.
 *   expert_load_counts: Float tensor of shape [num_experts] containing running average load of each expert.
 *   eviction_priority:  Float output tensor of shape [batch_size, seq_len]
 *   batch_size:         Number of batches
 *   seq_len:            Sequence length of the context
 *   num_heads:          Number of attention heads
 *   num_experts:        Total number of MoE experts
 *   expert_weight:      Hyperparameter weight (0.0 to 1.0) balancing attention vs expert signal
 */
__global__ void compute_eviction_priority_kernel(
    const float* __restrict__ attention_scores,
    const int*   __restrict__ expert_activations,
    const float* __restrict__ expert_load_counts,
    float*       __restrict__ eviction_priority,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_experts,
    float expert_weight
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (token_idx >= seq_len || batch_idx >= batch_size) {
        return;
    }

    // 1. Compute mean attention score across all heads for this token
    float attn_score_sum = 0.0f;
    for (int h = 0; h < num_heads; ++h) {
        int index = batch_idx * (num_heads * seq_len) + h * seq_len + token_idx;
        attn_score_sum += attention_scores[index];
    }
    float mean_attn_score = attn_score_sum / static_cast<float>(num_heads);

    // 2. Compute expert load signal (average load of experts this token was routed to)
    float expert_load_sum = 0.0f;
    int routed_experts_count = 0;
    int token_offset = (batch_idx * seq_len + token_idx) * num_experts;

    for (int e = 0; e < num_experts; ++e) {
        if (expert_activations[token_offset + e] > 0) {
            expert_load_sum += expert_load_counts[e];
            routed_experts_count++;
        }
    }

    float expert_signal = 0.0f;
    if (routed_experts_count > 0) {
        expert_signal = expert_load_sum / static_cast<float>(routed_experts_count);
    }

    // 3. Combined score
    // Higher load signal reduces priority (evicts tokens routed to overloaded experts sooner)
    int out_index = batch_idx * seq_len + token_idx;
    eviction_priority[out_index] = (1.0f - expert_weight) * mean_attn_score - expert_weight * expert_signal;
}

// Host wrapper to call the CUDA kernel
void launch_eviction_priority(
    const float* attention_scores,
    const int* expert_activations,
    const float* expert_load_counts,
    float* eviction_priority,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_experts,
    float expert_weight,
    cudaStream_t stream
) {
    dim3 block(256, 1, 1);
    dim3 grid((seq_len + block.x - 1) / block.x, batch_size, 1);

    compute_eviction_priority_kernel<<<grid, block, 0, stream>>>(
        attention_scores,
        expert_activations,
        expert_load_counts,
        eviction_priority,
        batch_size,
        seq_len,
        num_heads,
        num_experts,
        expert_weight
    );
}

} // extern "C"
