#include <metal_stdlib>

using namespace metal;

constant uint GEMV_TILE_K = 256;

kernel void quantized_linear_i8_f16(
    device const half* input [[buffer(0)]],
    device const char* weight [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= n || gid.y >= m) {
        return;
    }

    float sum = 0.0f;
    for (uint index = 0; index < k; ++index) {
        sum += float(input[gid.y * k + index])
            * float(weight[index * n + gid.x]);
    }
    output[gid.y * n + gid.x] = half(sum * scales[gid.x]);
}

kernel void quantized_gemv_i8_f16(
    device const half* input [[buffer(0)]],
    device const char* weight [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device half* output [[buffer(3)]],

    constant uint& k [[buffer(4)]],
    constant uint& n [[buffer(5)]],

    uint column [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup half input_tile[GEMV_TILE_K];

    const bool valid_column = column < n;

    float sum = 0.0f;

    for (
        uint tile_start = 0;
        tile_start < k;
        tile_start += GEMV_TILE_K
    ) {
        const uint input_index = tile_start + tid;

        if (input_index < k) {
            input_tile[tid] = input[input_index];
        } else {
            input_tile[tid] = half(0.0h);
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );

        const uint tile_size = min(GEMV_TILE_K, k - tile_start);

        if (valid_column) {
            for (uint inner = 0; inner < tile_size; ++inner) {
                const uint row = tile_start + inner;
                const uint weight_index = row * n + column;

                sum += float(input_tile[inner]) * float(weight[weight_index]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_column) {
        output[column] = half(sum * scales[column]);
    }
}
