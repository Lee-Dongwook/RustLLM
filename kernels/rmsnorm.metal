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

kernel void rmsnorm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],

    constant uint& hidden_size [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],

    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[256];

    float sum_sq = 0.0f;

    const uint row_offset = row * hidden_size;

    for (
        uint index = tid;
        index < hidden_size;
        index += 256
    ) {
        float value = float(input[row_offset + index]);
        sum_sq += value * value;
    }

    partial[tid] = sum_sq;

    threadgroup_barrier(
        mem_flags::mem_threadgroup
    );

    for (
        uint stride = 128;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    float inverse_rms = rsqrt(partial[0] / float(hidden_size) + epsilon);

    for (uint index = tid; index < hidden_size; index += 256) {
        float value = float(input[row_offset + index]);

        float scale = float(weight[index]);

        output[row_offset + index] = half(value * inverse_rms * scale);
    }
}
