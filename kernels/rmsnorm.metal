#include <metal_stdlib>

using namespace metal;

#define THREADGROUP_SIZE 256

kernel void rmsnorm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],

    constant uint& hidden_size [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],

    uint thread_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    threadgroup float partial_sums[THREADGROUP_SIZE];

    uint row_offset = group_id * hidden_size;

    float local_sum = 0.0f;

    for (
        uint i = thread_id;
        i < hidden_size;
        i += THREADGROUP_SIZE
    ) {
        float value = input[row_offset + i];

        local_sum += value * value;
    }

    partial_sums[thread_id] =
        local_sum;

    threadgroup_barrier(
        mem_flags::mem_threadgroup
    );

    for (
        uint stride = THREADGROUP_SIZE / 2;
        stride > 0;
        stride /= 2
    ) {
        if (thread_id < stride) {
            partial_sums[thread_id] += partial_sums[thread_id + stride];
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    float mean_square = 
        partial_sums[0] / float(hidden_size);
    
    float inverse_rms = rsqrt(
            mean_square + epsilon
    );

    for(
        uint i = thread_id;
        i < hidden_size;
        i += THREADGROUP_SIZE
    ) {
        output[row_offset + i] = input[row_offset + i] * inverse_rms * weight[i];
    }
}
