#include <metal_stdlib>

using namespace metal;

#define THREADGROUP_SIZE 256

kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],

    constant uint& width [[buffer(2)]],

    uint thread_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    threadgroup float shared[THREADGROUP_SIZE];

    uint row_offset =
        group_id * width;

    // --------------------------------------------------
    // 1. row의 최대값 찾기
    // --------------------------------------------------

    float local_max =
        -INFINITY;

    for (
        uint i = thread_id;
        i < width;
        i += THREADGROUP_SIZE
    ) {
        local_max =
            max(
                local_max,
                input[row_offset + i]
            );
    }

    shared[thread_id] =
        local_max;

    threadgroup_barrier(
        mem_flags::mem_threadgroup
    );

    for (
        uint stride = THREADGROUP_SIZE / 2;
        stride > 0;
        stride /= 2
    ) {
        if (thread_id < stride) {
            shared[thread_id] =
                max(
                    shared[thread_id],
                    shared[
                        thread_id + stride
                    ]
                );
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    float row_max =
        shared[0];

    // --------------------------------------------------
    // 2. exp(x - max)의 합 계산
    // --------------------------------------------------

    float local_sum =
        0.0f;

    for (
        uint i = thread_id;
        i < width;
        i += THREADGROUP_SIZE
    ) {
        local_sum +=
            exp(
                input[row_offset + i]
                - row_max
            );
    }

    shared[thread_id] =
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
            shared[thread_id] +=
                shared[
                    thread_id + stride
                ];
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    float denominator =
        shared[0];

    // --------------------------------------------------
    // 3. 최종 확률 계산
    // --------------------------------------------------

    for (
        uint i = thread_id;
        i < width;
        i += THREADGROUP_SIZE
    ) {
        output[row_offset + i] =
            exp(
                input[row_offset + i]
                - row_max
            )
            / denominator;
    }
}

