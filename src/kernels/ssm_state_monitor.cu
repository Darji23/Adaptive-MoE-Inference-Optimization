#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C" {

/**
 * Mamba-2 SSM State Monitor CUDA Kernel
 *
 * This kernel monitors recurrent state values in Mamba-2 layers, computing
 * local L2-norms or max values to analyze activation and decay dynamics
 * without bringing large state tensors back to host.
 *
 * Parameters:
 *   ssm_states:    Float tensor of shape [batch_size, num_heads, d_state]
 *                  Contains Mamba-2 recurrent states.
 *   state_norms:   Float output tensor of shape [batch_size, num_heads]
 *                  Stores computed metrics (e.g. L2 norm) per head.
 *   batch_size:    Number of sequences
 *   num_heads:     Number of Mamba-2 heads
 *   d_state:       State dimension size (e.g. 64 or 128)
 */
__global__ void monitor_ssm_states_kernel(
    const float* __restrict__ ssm_states,
    float*       __restrict__ state_norms,
    int batch_size,
    int num_heads,
    int d_state
) {
    int head_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (head_idx >= num_heads || batch_idx >= batch_size) {
        return;
    }

    int state_offset = (batch_idx * num_heads + head_idx) * d_state;
    float sum_sq = 0.0f;

    for (int d = 0; d < d_state; ++d) {
        float val = ssm_states[state_offset + d];
        sum_sq += val * val;
    }

    int out_idx = batch_idx * num_heads + head_idx;
    state_norms[out_idx] = sqrtf(sum_sq);
}

// Host wrapper to call the monitor kernel
void launch_ssm_state_monitor(
    const float* ssm_states,
    float* state_norms,
    int batch_size,
    int num_heads,
    int d_state,
    cudaStream_t stream
) {
    dim3 block(128, 1, 1);
    dim3 grid((num_heads + block.x - 1) / block.x, batch_size, 1);

    monitor_ssm_states_kernel<<<grid, block, 0, stream>>>(
        ssm_states,
        state_norms,
        batch_size,
        num_heads,
        d_state
    );
}

} // extern "C"
